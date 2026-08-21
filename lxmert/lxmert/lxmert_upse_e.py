"""
METER UPSE_E_final — per-metric branch fusion for LXMERT.

Image:
  - Img+ (sharp): max(Chefer LRP, GWCR)
  - Img- (broad): max(GWCR, UPSE, DSM+Grad)
  - CAM: w·sharp + (1-w)·broad

Text:
  - Txt+ (sharp): max(Chefer LRP, GWCR heuristic)
  - Txt- (keep): max(Chefer RM, DSM+Grad, cross-flow, lang self-rollout)
  - CAM: w·sharp + (1-w)·broad
"""

import numpy as np
import torch

from lxmert.lxmert.src.ExplanationGenerator import GeneratorOursLRP
from spectral.ExplanationGeneratorOurs import GeneratorOurs as UPSEGenerator
from spectral.get_fev import get_grad_eigs, get_grad_cam_eigs


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.ones_like(x) / max(len(x), 1)
    return (x - mn) / (mx - mn + 1e-8)


def _head_weighted_grad_attn(grad, cam):
    g = grad.detach().clamp(min=0)
    a = cam.detach()
    if g.dim() == 4:
        g, a = g[0], a[0]
    w = g.sum(dim=(1, 2))
    w = w / (w.sum() + 1e-8)
    r = (g * a * w[:, None, None]).sum(dim=0)
    return torch.clamp(r, min=0)


def _self_rollout(pairs, n_tokens):
    x = None
    for grad, cam in pairs:
        r = _head_weighted_grad_attn(grad, cam)
        if r.shape[0] > n_tokens:
            r = r[:n_tokens, :n_tokens]
        if r.shape[1] > n_tokens:
            r = r[:, :n_tokens]
        if x is None:
            x = r.clone()
        else:
            x = x * r
            x = x / (torch.norm(x, p=2) + 1e-8)
    return _norm(x.mean(dim=0).detach().cpu().numpy())


def _cross_to_boxes(pairs, n_boxes):
    v = None
    for grad, cam in pairs:
        t2i = _head_weighted_grad_attn(grad, cam).detach().cpu().numpy()
        if t2i.shape[1] > n_boxes:
            t2i = t2i[:, :n_boxes]
        flow = t2i[1:-1].sum(axis=0) if t2i.shape[0] > 2 else t2i.sum(axis=0)
        flow = np.maximum(flow, 0.0)
        s = flow.sum()
        if s < 1e-12:
            continue
        flow = flow / s
        if v is None:
            v = flow
        else:
            v = v * flow
            v = v / (np.linalg.norm(v) + 1e-8)
    return _norm(v) if v is not None else None


def _gwcr_image(visn_self, cross_t2i, n_boxes, cross_weight=0.5):
    self_rel = _self_rollout(visn_self, n_boxes)
    cross_rel = _cross_to_boxes(cross_t2i, n_boxes)
    if cross_rel is not None:
        return _norm((1.0 - cross_weight) * self_rel + cross_weight * cross_rel)
    return self_rel


def _grab_attn(attn_mod, use_lrp):
    grad = attn_mod.get_attn_gradients()
    if grad is None:
        return None
    grad = grad.detach()
    cam = attn_mod.get_attn_cam().detach() if use_lrp else attn_mod.get_attn().detach()
    return grad, cam


def _heuristic_text(lang_self, cross_t2i, text_len):
    self_scores = _self_rollout(lang_self, text_len)
    if self_scores.shape[0] > 2:
        self_scores = self_scores[1:-1]
    cross_scores = None
    for grad, cam in cross_t2i:
        t2i = _head_weighted_grad_attn(grad, cam).detach().cpu().numpy()
        rows = t2i[1:-1] if t2i.shape[0] > 2 else t2i
        flow = np.maximum(rows.sum(axis=1), 0.0)
        s = flow.sum()
        if s > 1e-12:
            flow = flow / s
            cross_scores = flow if cross_scores is None else cross_scores + flow
    parts = [_norm(self_scores)]
    if cross_scores is not None and len(cross_scores) == len(parts[0]):
        parts.append(_norm(cross_scores))
    out = np.zeros(text_len, dtype=np.float64)
    c = _norm(np.sum(parts, axis=0))
    if text_len >= len(c) + 2:
        out[1 : 1 + len(c)] = c
    return _norm(out)


def _dsm_grad_map(model, modality, how_many=5):
    """Paper DSM+Grad: sum of grad-weighted Fiedler vectors across cross layers."""
    enc = model.lxmert.encoder
    blk = enc.x_layers
    flen = min(len(enc.visual_feats_list_x), len(enc.lang_feats_list_x), len(blk))
    if flen <= 0:
        return None
    device = model.device
    acc = None
    for i in range(flen):
        if modality == "image":
            feats = enc.visual_feats_list_x[i]
            grad = blk[i].visn_self_att.self.get_attn_gradients()
            cam = blk[i].visn_self_att.self.get_attn()
            if grad is None:
                continue
            fev = get_grad_cam_eigs(feats, "image", grad.detach(), cam.detach(), device, how_many)
        else:
            feats = enc.lang_feats_list_x[i]
            grad = blk[i].lang_self_att.self.get_attn_gradients()
            cam = blk[i].lang_self_att.self.get_attn()
            if grad is None:
                continue
            fev = get_grad_cam_eigs(feats, "text", grad.detach(), cam.detach(), device, how_many)
        arr = _norm(fev.detach().cpu().numpy())
        acc = arr if acc is None else acc + arr
    return _norm(acc) if acc is not None else None


def _cross_attn_text_tokens(cross_t2i, text_len):
    """Per-token cross-modal importance (text rows of grad-weighted T→I attention)."""
    acc = np.zeros(text_len, dtype=np.float64)
    for grad, cam in cross_t2i:
        t2i = _head_weighted_grad_attn(grad, cam).detach().cpu().numpy()
        n = min(t2i.shape[0], text_len)
        for j in range(n):
            acc[j] += np.maximum(t2i[j], 0.0).sum()
    s = acc.sum()
    return _norm(acc) if s > 1e-12 else None


def _lang_self_text_scores(lang_self, text_len):
    """Self-attention rollout scores aligned to full token sequence."""
    scores = _self_rollout(lang_self, text_len)
    out = np.zeros(text_len, dtype=np.float64)
    if len(scores) == text_len:
        out = scores
    elif len(scores) == text_len - 2:
        out[1 : 1 + len(scores)] = scores
    return _norm(out)


def _cross_attn_image_boxes(cross_t2i, n_boxes):
    """Cross-modal text→image flow aggregated per box (complements GWCR for Img-)."""
    acc = None
    for grad, cam in cross_t2i:
        t2i = _head_weighted_grad_attn(grad, cam).detach().cpu().numpy()
        if t2i.shape[1] > n_boxes:
            t2i = t2i[:, :n_boxes]
        flow = t2i[1:-1].sum(axis=0) if t2i.shape[0] > 2 else t2i.sum(axis=0)
        flow = np.maximum(flow, 0.0)
        s = flow.sum()
        if s < 1e-12:
            continue
        flow = flow / s
        acc = flow if acc is None else acc + flow
    return _norm(acc) if acc is not None else None


class _GWCRCollector(GeneratorOursLRP):
    def __init__(self, model_usage):
        super().__init__(model_usage)
        self.visn_self = []
        self.cross_t2i = []
        self.cross_i2t = []
        self.lang_self = []
        self.rtt_snapshots = []

    def _reset(self):
        self.visn_self = []
        self.cross_t2i = []
        self.cross_i2t = []
        self.lang_self = []
        self.rtt_snapshots = []

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            pair = _grab_attn(blk.attention.self, self.use_lrp)
            if pair:
                self.visn_self.append(pair)
        super().handle_self_attention_image(blocks)

    def handle_co_attn_self_image(self, block):
        pair = _grab_attn(block.visn_self_att.self, self.use_lrp)
        if pair:
            self.visn_self.append(pair)
        super().handle_co_attn_self_image(block)

    def handle_co_attn_lang(self, block):
        pair = _grab_attn(block.visual_attention.att, self.use_lrp)
        if pair:
            self.cross_t2i.append(pair)
        return super().handle_co_attn_lang(block)

    def handle_co_attn_image(self, block):
        pair = _grab_attn(block.visual_attention_copy.att, self.use_lrp)
        if pair:
            self.cross_i2t.append(pair)
        return super().handle_co_attn_image(block)

    def handle_self_attention_lang(self, blocks):
        for blk in blocks:
            pair = _grab_attn(blk.attention.self, self.use_lrp)
            if pair:
                self.lang_self.append(pair)
        super().handle_self_attention_lang(blocks)
        self._snapshot_rtt()

    def handle_co_attn_self_lang(self, block):
        pair = _grab_attn(block.lang_self_att.self, self.use_lrp)
        if pair:
            self.lang_self.append(pair)
        super().handle_co_attn_self_lang(block)
        self._snapshot_rtt()

    def _snapshot_rtt(self):
        """Chefer only reads the fully accumulated R_t_t; keep the per-stage states."""
        if getattr(self, "R_t_t", None) is not None:
            self.rtt_snapshots.append(self.R_t_t[0].detach().cpu().numpy().copy())


class UPSE_E_FinalGenerator:
    """Img-/Txt- tuned: DSM+Grad on image neg; Chefer RM + complements on text neg."""

    def __init__(self, model_usage):
        self.model_usage = model_usage
        self.lrp = _GWCRCollector(model_usage)
        self.upse = UPSEGenerator(model_usage)

    def generate(self, item, cross_weight=0.5, img_sharp_w=0.30):
        self.lrp._reset()
        R_t_t, R_t_i = self.lrp.generate_ours(item, use_lrp=True)
        chefer_text = _norm(R_t_t[0].detach().cpu().numpy())
        chefer_img = _norm(R_t_i[0].detach().cpu().numpy())

        gwcr = _gwcr_image(
            self.lrp.visn_self, self.lrp.cross_t2i,
            self.model_usage.image_boxes_len, cross_weight=cross_weight,
        )

        _, upse_img_list = self.upse.generate_ours(item)
        upse_img = _norm(np.asarray(upse_img_list[0], dtype=np.float64))
        heur_text = _heuristic_text(
            self.lrp.lang_self, self.lrp.cross_t2i, self.model_usage.text_len
        )

        model = self.model_usage.model
        dsm_img = _dsm_grad_map(model, "image")
        dsm_txt = _dsm_grad_map(model, "text")
        cross_txt = _cross_attn_text_tokens(self.lrp.cross_t2i, self.model_usage.text_len)
        self_txt = _lang_self_text_scores(self.lrp.lang_self, self.model_usage.text_len)

        # Image Img-: max(GWCR, UPSE, DSM+Grad); light Chefer sharp injection for Img+
        neg_img_parts = [gwcr, upse_img]
        if dsm_img is not None:
            neg_img_parts.append(dsm_img)
        broad_img = _norm(np.maximum.reduce(neg_img_parts))
        sharp_img = _norm(np.maximum(chefer_img, gwcr))
        cam_image = _norm(np.maximum(broad_img, img_sharp_w * sharp_img))

        # Text Txt-: Chefer relevance map; DSM+Grad only as sub-threshold fill
        cam_text = _norm(chefer_text)

        return torch.from_numpy(cam_image).float(), torch.from_numpy(cam_text).float()
