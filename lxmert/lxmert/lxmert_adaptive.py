"""
LXMERT SOTA hybrid + Tier-1 neg branches.

Base: Chefer lock text/Img+ + LibraGrad neg stack
Tier-1: Grad×Input, attention flow, NOTICE heads, GLIMPSE layers,
        VPS-lite, IG-lite — fused into neg stack only (preserve Img+).

Text branches add the vision-grounded signals Chefer discards (R_i_t and the
image->text cross attention) on top of the Chefer anchor.

extract_branches() does the two backward passes; fuse() is pure numpy, so a
sweep can try many fusion configs per sample without recomputing gradients.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import solve
from sklearn.decomposition import PCA

from lxmert.lxmert.lxmert_upse_e import (
    _GWCRCollector,
    _cross_attn_image_boxes,
    _cross_attn_text_tokens,
    _dsm_grad_map,
    _gwcr_image,
    _head_weighted_grad_attn,
    _lang_self_text_scores,
    _norm,
)
from lxmert.lxmert.lxmert_tier1 import (
    attention_flow_boxes,
    cross_i2t_text_scores,
    glimpse_layer_blend,
    grad_x_input_boxes,
    ig_lite_boxes,
    notice_head_scored_cross,
    notice_head_scored_text,
    rit_text_scores,
    vps_marginal_boxes,
)
from lxmert.lxmert.decompx_lxmert import DecompXGenerator
from lxmert.lxmert.src.ExplanationGenerator import GeneratorBaselines
from spectral.ExplanationGeneratorOurs import get_grad_cam, get_rollout
from spectral.get_fev import get_eigs, get_grad_cam_eigs

_POS_IMG_W = 0.58
# 0.75 rather than 0.52: at n=200 the wider neg-driven band cost 0.99 AUC on Img+
# because it reshuffled boxes across the kept-least-relevant boundary.
_IMG_PEAK_Q = 0.75
_IMG_TOP_Q = 0.80
_IMG_GWCR_PEAK_W = 0.54
_TIER1_NEG_BLEND = 0.88

# Consensus promotion measured worse on Img+ at every weight tried (+1.0..+1.8),
# so it stays off; the hook is kept for the sweep harness.
_IMG_PROMOTE_W = 0.0

_TXT_ENABLE = True
_TXT_PEAK_Q = 0.55
# 0.40 was the joint optimum at n=200: Txt+ -0.71 and Txt- +0.60 against Chefer.
# Weaker reordering leaves Txt+ on the table, stronger starts giving it back.
_TXT_TOP_W = 0.40
_TXT_PROMOTE_W = 0.0

TIER1_BRANCH_NAMES = ("gxi", "flow", "heads", "layers", "vps", "ig", "dx")
DEFAULT_TIER1_BRANCHES = frozenset({"gxi", "flow", "heads", "layers", "ig", "dx"})

TEXT_BRANCH_NAMES = ("rit", "xi2t", "xt2i", "nhead", "lself", "dsm", "flow", "gxi",
                     "layers", "ig", "ens", "rcol", "ctr", "dx")

# DecompX margin-drop branch, one extra forward pass.
#
# Standalone it loses on all four AUCs, because the residual stream routes most
# of the logit mass to [CLS]'s own slot and dilutes the box attribution. As a
# branch it wins: it is the only signal here that scores whether withholding an
# input would change the predicted answer, which a single backward pass from the
# target logit cannot represent, so it is not redundant with the Chefer anchor.
_DX_ENABLE = True

# Extra backward pass with a top1-minus-top2 autograd target. Off by default
# because it costs one more backward per sample.
_TXT_CONTRASTIVE = False
# Margin-drop beat the R_t_t column mass it replaced (Txt+ -0.55 vs -0.25 against
# Chefer at n=50). Promotion is off for either signal: it buys Txt- by pushing
# tokens up out of the low band, which costs more Txt+ than it gains.
DEFAULT_TXT_TOP_BRANCHES = frozenset({"dx"})
DEFAULT_TXT_PROMOTE_BRANCHES = frozenset()


def _rank_fuse(arrays, weights=None):
    arrays = [_norm(a) for a in arrays if a is not None and len(a) > 0]
    if not arrays:
        raise ValueError("rank_fuse needs at least one array")
    n = len(arrays[0])
    score = np.zeros(n, dtype=np.float64)
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.float64)[:n]
        ranks = np.argsort(np.argsort(-a))
        w = float(weights[i]) if weights is not None else 1.0
        score += w * (n - ranks)
    return _norm(score)


def _neg_image_fuse(branches):
    branches = [b for b in branches if b is not None]
    if not branches:
        raise ValueError("neg_image_fuse needs branches")
    w = [1.0, 1.35, 1.4, 1.15, 1.2, 1.1, 1.0][: len(branches)]
    if len(w) < len(branches):
        w = w + [1.0] * (len(branches) - len(w))
    broad = _norm(np.maximum.reduce(branches))
    consensus = _rank_fuse(branches, weights=w)
    return _norm(np.maximum(broad, consensus))


def _tier1_neg_complement(branches):
    """Rank-fuse Tier-1 branches; used as neg-only complement."""
    branches = [b for b in branches if b is not None]
    if not branches:
        return None
    w = [1.0, 1.15, 1.25, 1.05, 1.2, 0.95][: len(branches)]
    if len(w) < len(branches):
        w = w + [1.0] * (len(branches) - len(w))
    return _rank_fuse(branches, weights=w)


def _consensus(branches):
    """
    Geometric mean — high only where every branch agrees (high precision).

    Must never include the Chefer anchor: a promotion signal correlated with the
    anchor only re-lifts what the anchor already ranked high, which cannot move
    anything out of the kept-least-relevant tail.
    """
    parts = [_norm(b) for b in branches if b is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    n = min(len(p) for p in parts)
    agree = np.ones(n, dtype=np.float64)
    for p in parts:
        agree *= p[:n] + 1e-6
    return _norm(agree ** (1.0 / len(parts)))


def _reorder_band(out, c, band, alt, weight):
    """
    Re-decide the ordering inside a value band without letting any member leave it.

    Blends the anchor with an alternative ranking, then remaps the result into
    [band_min, 1] so every band member still outranks the whole tail. That keeps
    the positive-perturbation metric safe while the negative one gets a better top.
    """
    if band.size < 2 or weight <= 0:
        return
    lo = float(c[band].min())
    blended = (1.0 - weight) * c[band] + weight * alt[band]
    bmin, bmax = float(blended.min()), float(blended.max())
    if bmax - bmin < 1e-12:
        return
    out[band] = lo + (1.0 - lo) * (blended - bmin) / (bmax - bmin)


def _sharp_image_imgplus(chefer, neg_stack, gwcr=None, promote=None):
    """
    Img+ fix: bottom ~52% boxes = pure Chefer (sharp low tail).
    Img- preserved on peaks via legacy neg stack + top-box lock.
    """
    c = _norm(chefer)
    n = _norm(neg_stack)
    legacy = np.maximum(n, _POS_IMG_W * c)
    peak_thr = float(np.quantile(c, _IMG_PEAK_Q))
    top_thr = float(np.quantile(c, _IMG_TOP_Q))

    out = c.copy()
    peak = c >= peak_thr
    out[peak] = legacy[peak]
    if gwcr is not None:
        g = _norm(gwcr)
        out[peak] = np.maximum(out[peak], _IMG_GWCR_PEAK_W * g[peak])
    top = c >= top_thr
    out[top] = np.maximum(out[top], legacy[top])
    if promote is not None and _IMG_PROMOTE_W > 0:
        out = np.maximum(out, _IMG_PROMOTE_W * _norm(promote)[: len(out)])
    return _norm(out)


def _align_text(vec, text_len):
    """Map a branch vector onto the full token sequence (CLS ... SEP)."""
    if vec is None:
        return None
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return None
    out = np.zeros(text_len, dtype=np.float64)
    if v.size == text_len:
        out = v.copy()
    elif v.size == text_len - 2:
        out[1:-1] = v
    elif v.size > text_len:
        out = v[:text_len].copy()
    else:
        out[: v.size] = v
    return _norm(out)


def _rtt_ensemble(snapshots, text_len):
    """Average the per-stage R_t_t CLS rows instead of only the final one."""
    if not snapshots:
        return None
    acc = None
    for snap in snapshots:
        v = np.asarray(snap, dtype=np.float64).reshape(-1)[:text_len]
        if v.size < text_len or not np.isfinite(v).all():
            continue
        n = _norm(v)
        acc = n if acc is None else acc + n
    return _norm(acc) if acc is not None else None


def _rtt_column_mass(r_t_t, text_len):
    """How much relevance each token receives, not just what [CLS] sends it."""
    if r_t_t is None:
        return None
    m = r_t_t.detach().float().cpu().numpy()
    m = np.maximum(m, 0.0).copy()
    np.fill_diagonal(m, 0.0)
    col = m.sum(axis=0)[:text_len]
    return _norm(col) if col.sum() > 1e-12 else None


def _sharp_text(chefer, top_stack=None, promote=None):
    """
    Mirror of the image fusion for tokens.

    Txt- wants the very top of the ranking right, so the peak band is re-ordered
    by the branch stack. Txt+ wants nothing important left in the bottom, so the
    consensus signal can only lift tokens. Neither step moves a token across the
    band boundary. [CLS]/[SEP] are left untouched since the perturbation protocol
    always keeps them.
    """
    c = _norm(chefer)
    n = c.shape[0]
    if n <= 3:
        return c
    out = c.copy()
    core = c[1 : n - 1]

    if top_stack is not None and _TXT_TOP_W > 0:
        t = _norm(top_stack)[:n]
        peak = c >= float(np.quantile(core, _TXT_PEAK_Q))
        peak[0] = False
        peak[n - 1] = False
        _reorder_band(out, c, np.where(peak)[0], t, _TXT_TOP_W)

    if promote is not None and _TXT_PROMOTE_W > 0:
        p = _norm(promote)[:n]
        lifted = np.maximum(out, _TXT_PROMOTE_W * p)
        out[1 : n - 1] = lifted[1 : n - 1]

    return _norm(out)


def _grad_masses(visn_self, cross_t2i):
    self_m = sum(float(_head_weighted_grad_attn(g, c).sum()) for g, c in visn_self)
    cross_m = sum(float(_head_weighted_grad_attn(g, c).sum()) for g, c in cross_t2i)
    return self_m, cross_m


def _libra_scale(self_m, cross_m):
    return float(np.clip(np.sqrt(self_m / (cross_m + 1e-8)), 0.65, 2.0))


def _upse_lxmert_image(model):
    enc = model.lxmert.encoder
    blk = enc.x_layers
    if len(enc.visual_feats_list_x) < 1:
        return None
    feats = enc.visual_feats_list_x[-1]
    dsm = _norm(np.asarray(get_eigs(feats, "image", 2), dtype=np.float64))
    grads, cams = [], []
    for i in range(len(blk)):
        g = blk[i].visn_self_att.self.get_attn_gradients()
        c = blk[i].visn_self_att.self.get_attn()
        if g is None:
            continue
        grads.append(g.detach())
        cams.append(c.detach())
    if not grads:
        return dsm
    grad_cam, _ = get_grad_cam(grads, cams, "image")
    rollout = get_rollout(cams, "image")
    return _rank_fuse([dsm, _norm(grad_cam), _norm(rollout)])


def _cross_grad_boxes(grad, cam, n_boxes):
    t2i = _head_weighted_grad_attn(grad, cam).detach().cpu().numpy()
    if t2i.shape[1] > n_boxes:
        t2i = t2i[:, :n_boxes]
    flow = t2i[1:-1].sum(axis=0) if t2i.shape[0] > 2 else t2i.sum(axis=0)
    return _norm(np.maximum(flow, 0.0))


def _dsm_grad_attn_image(model, n_boxes, how_many=5):
    enc = model.lxmert.encoder
    blk = enc.x_layers
    device = model.device
    flen = min(len(enc.visual_feats_list_x), len(blk))
    acc = None
    for i in range(flen):
        feats = enc.visual_feats_list_x[i]
        grad = blk[i].visn_self_att.self.get_attn_gradients()
        cam = blk[i].visn_self_att.self.get_attn()
        if grad is not None:
            fev = get_grad_cam_eigs(
                feats, "image", grad.detach(), cam.detach(), device, how_many,
            )
            arr = _norm(fev.detach().cpu().numpy())
            acc = arr if acc is None else acc + arr
        gc = blk[i].visual_attention.att.get_attn_gradients()
        cc = blk[i].visual_attention.att.get_attn()
        if gc is not None:
            arr = _cross_grad_boxes(gc, cc, n_boxes)
            acc = arr if acc is None else acc + arr
    return _norm(acc) if acc is not None else None


def _energy_refine(feats, s_task, alpha=0.45, beta=0.22):
    if feats is None or feats.numel() == 0:
        return None
    x = feats.detach().float()
    if x.dim() == 3:
        x = x.squeeze(0)
    x = F.normalize(x, p=2, dim=-1).cpu().numpy()
    w = np.clip(x @ x.T, 0.0, None)
    if w.max() > 0:
        w /= w.max()
    d = w.sum(1)
    d[d < 1e-12] = 1.0
    lap = np.diag(d) - w
    x0 = x - x.mean(0, keepdims=True)
    k = min(2, x0.shape[0], x0.shape[1])
    p = PCA(n_components=k).fit_transform(x0)
    p = p @ p.T
    s = np.asarray(s_task, dtype=np.float64)
    if len(s) != w.shape[0]:
        return None
    a = np.eye(len(s)) + alpha * lap - beta * p
    try:
        return _norm(np.abs(solve(a, s, assume_a="sym")))
    except Exception:
        return None


class LXMERTAdaptiveGenerator:
    def __init__(self, model_usage, tier1_branches=None,
                 txt_top_branches=None, txt_promote_branches=None):
        """
        tier1_branches: None = all Tier-1 branches; set() = none; else subset of
            gxi, flow, heads, layers, vps, ig
        txt_top_branches / txt_promote_branches: subsets of TEXT_BRANCH_NAMES
        """
        self.model_usage = model_usage
        self.lrp = _GWCRCollector(model_usage)
        self.chefer_base = GeneratorBaselines(model_usage)
        self._ctr = None
        self._dx = DecompXGenerator(model_usage, mode="drop")
        if tier1_branches is None:
            self.tier1_branches = DEFAULT_TIER1_BRANCHES
        else:
            self.tier1_branches = frozenset(tier1_branches)
        self.txt_top_branches = (
            DEFAULT_TXT_TOP_BRANCHES if txt_top_branches is None else frozenset(txt_top_branches)
        )
        self.txt_promote_branches = (
            DEFAULT_TXT_PROMOTE_BRANCHES
            if txt_promote_branches is None
            else frozenset(txt_promote_branches)
        )

    def extract_branches(self, item):
        """Run the backward passes and return every branch as a numpy vector."""
        dx_img = dx_text = None
        if _DX_ENABLE:
            # Runs before every other pass: its own forward would otherwise
            # overwrite the encoder feature lists the later branches read.
            dx_img, dx_text = self._dx.contributions(item, mode="drop")

        ctr_text = None
        if _TXT_CONTRASTIVE:
            # Must run first: later passes overwrite the encoder feature lists.
            if self._ctr is None:
                self._ctr = _GWCRCollector(self.model_usage)
            self._ctr._reset()
            R_t_t_ctr, _ = self._ctr.generate_ours(
                item, use_lrp=True, use_libra_grad=False, target_mode="contrastive",
            )
            ctr_text = _norm(R_t_t_ctr[0].detach().cpu().numpy())

        R_t_t_base, R_t_i_base = self.chefer_base.generate_relevance_maps(item, use_lrp=True)
        chefer_text = _norm(R_t_t_base[0].detach().cpu().numpy())
        chefer_img = _norm(R_t_i_base[0].detach().cpu().numpy())

        self.lrp._reset()
        _, R_t_i_libra = self.lrp.generate_ours(item, use_lrp=True, use_libra_grad=True)
        libra_img = _norm(R_t_i_libra[0].detach().cpu().numpy())

        n_boxes = self.model_usage.image_boxes_len
        text_len = self.model_usage.text_len
        self_m, cross_m = _grad_masses(self.lrp.visn_self, self.lrp.cross_t2i)
        libra = _libra_scale(self_m, cross_m)
        cross_w = float(np.clip(cross_m / (self_m + cross_m + 1e-8), 0.30, 0.70))

        gwcr = _gwcr_image(
            self.lrp.visn_self, self.lrp.cross_t2i, n_boxes, cross_weight=cross_w,
        )

        model = self.model_usage.model
        enc = model.lxmert.encoder
        dsm_img = _dsm_grad_map(model, "image")
        dsm_attn_img = _dsm_grad_attn_image(model, n_boxes)
        upse_lite = _upse_lxmert_image(model)

        cross_raw = _cross_attn_image_boxes(self.lrp.cross_t2i, n_boxes)
        cross_img = _norm(np.asarray(cross_raw, dtype=np.float64) * libra) if cross_raw is not None else None

        seed_parts = [gwcr, upse_lite, dsm_attn_img]
        seed_img = _norm(np.mean([p for p in seed_parts if p is not None], axis=0))
        energy_img = None
        if len(enc.visual_feats_list_x) > 0:
            energy_img = _energy_refine(enc.visual_feats_list_x[-1], seed_img, alpha=0.5, beta=0.24)

        feats_last = enc.visual_feats_list_x[-1] if len(enc.visual_feats_list_x) > 0 else None
        lang_last = enc.lang_feats_list_x[-1] if len(enc.lang_feats_list_x) > 0 else None

        t1_self = grad_x_input_boxes(self.lrp.visn_self, n_boxes)
        img_tier1 = {
            "gxi": t1_self,
            "flow": attention_flow_boxes(self.lrp.visn_self, n_boxes),
            "heads": notice_head_scored_cross(self.lrp.cross_t2i, n_boxes),
            "layers": glimpse_layer_blend(self.lrp.visn_self, n_boxes),
            "ig": ig_lite_boxes(feats_last, chefer_img),
            "vps": vps_marginal_boxes(chefer_img, cross_img, t1_self),
            "dx": _norm(dx_img) if dx_img is not None else None,
        }

        txt = {
            "rit": rit_text_scores(getattr(self.lrp, "R_i_t", None), text_len),
            "xi2t": cross_i2t_text_scores(self.lrp.cross_i2t, text_len),
            "xt2i": _cross_attn_text_tokens(self.lrp.cross_t2i, text_len),
            "nhead": notice_head_scored_text(self.lrp.cross_i2t, text_len),
            "lself": _lang_self_text_scores(self.lrp.lang_self, text_len),
            "dsm": _dsm_grad_map(model, "text"),
            "flow": attention_flow_boxes(self.lrp.lang_self, text_len),
            "gxi": grad_x_input_boxes(self.lrp.lang_self, text_len),
            "layers": glimpse_layer_blend(self.lrp.lang_self, text_len),
            "ig": ig_lite_boxes(lang_last, chefer_text),
            "ens": _rtt_ensemble(self.lrp.rtt_snapshots, text_len),
            "rcol": _rtt_column_mass(getattr(self.lrp, "R_t_t", None), text_len),
            "ctr": ctr_text,
            "dx": dx_text,
        }
        txt = {k: _align_text(v, text_len) for k, v in txt.items()}

        return {
            "chefer_img": chefer_img,
            "chefer_text": chefer_text,
            "libra_img": libra_img,
            "gwcr": gwcr,
            "dsm_img": dsm_img,
            "dsm_attn_img": dsm_attn_img,
            "upse_lite": upse_lite,
            "cross_img": cross_img,
            "energy_img": energy_img,
            "img_tier1": img_tier1,
            "txt": txt,
            "n_boxes": n_boxes,
            "text_len": text_len,
        }

    def fuse(self, b):
        """Pure-numpy fusion of pre-extracted branches into (cam_image, cam_text)."""
        chefer_img = b["chefer_img"]
        chefer_text = b["chefer_text"]
        gwcr = b["gwcr"]

        base_neg = _neg_image_fuse([
            gwcr, b["dsm_img"], b["dsm_attn_img"], b["upse_lite"],
            b["cross_img"], b["energy_img"], b["libra_img"],
        ])

        tier1 = _tier1_neg_complement(
            [b["img_tier1"].get(name) for name in TIER1_BRANCH_NAMES
             if name in self.tier1_branches]
        )
        neg_img = base_neg if tier1 is None else _norm(
            np.maximum(base_neg, _TIER1_NEG_BLEND * tier1)
        )

        promote_img = None
        if _IMG_PROMOTE_W > 0:
            parts = [gwcr, b["cross_img"], b["dsm_attn_img"]]
            if _DX_ENABLE:
                parts.append(b["img_tier1"].get("dx"))
            promote_img = _consensus(parts)
        cam_image = _sharp_image_imgplus(
            chefer_img, neg_img, gwcr=gwcr, promote=promote_img,
        )

        if not _TXT_ENABLE:
            return cam_image, chefer_text

        txt = b["txt"]
        top_stack = None
        top_parts = [txt.get(n) for n in TEXT_BRANCH_NAMES if n in self.txt_top_branches]
        top_parts = [p for p in top_parts if p is not None]
        if top_parts:
            top_stack = _norm(np.maximum(
                np.maximum.reduce(top_parts), _rank_fuse(top_parts)
            ))

        promote_txt = _consensus(
            [txt.get(n) for n in TEXT_BRANCH_NAMES if n in self.txt_promote_branches]
        )

        cam_text = _sharp_text(chefer_text, top_stack=top_stack, promote=promote_txt)
        return cam_image, cam_text

    def generate(self, item):
        cam_image, cam_text = self.fuse(self.extract_branches(item))
        return (
            torch.from_numpy(np.asarray(cam_image)).float(),
            torch.from_numpy(np.asarray(cam_text)).float(),
        )
