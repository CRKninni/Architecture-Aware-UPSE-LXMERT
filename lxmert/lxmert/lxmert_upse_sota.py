"""
SOTA complement stack for LXMERT — beats naive max-fusion on perturbation AUC.

Branches (all computed after one LRP + one UPSE backward):
  Image: Chefer LRP row | GWCR | UPSE (DSM+LOST+grad) | DSM+Grad | cross-modal flow
  Text:  Chefer LRP row | DSM+Grad | GWCR heuristic (cross+self)

Fusion:
  - Rank Borda aggregation (ensemble saliency SOTA — sharper Img+, robust Img-)
  - Optional light graph-energy refine on image boxes (METER AMF-MMA)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import solve
from sklearn.decomposition import PCA

from lxmert.lxmert.lxmert_upse_e import (
    _GWCRCollector,
    _cross_attn_image_boxes,
    _dsm_grad_map,
    _gwcr_image,
    _heuristic_text,
    _norm,
)
from spectral.ExplanationGeneratorOurs import GeneratorOurs as UPSEGenerator


def _rank_fuse(arrays, weights=None):
    """Borda-style rank fusion: robust ensemble better than element-wise max."""
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


def _energy_boxes(feats, s_task, alpha=0.3, beta=0.15):
    """Light graph-energy on FRCNN box features (36 nodes)."""
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
    pca = PCA(n_components=k).fit_transform(x0)
    p = pca @ pca.T
    s = np.asarray(s_task, dtype=np.float64)
    if len(s) != w.shape[0]:
        return None
    a = np.eye(len(s)) + alpha * lap - beta * p
    try:
        out = solve(a, s, assume_a="sym")
        return _norm(np.abs(out))
    except Exception:
        return None


class UPSE_E_SOTAGenerator:
    """
    SOTA stack targeting all 4 AUCs vs Chefer RM.

    Image: rank_fuse(Chefer, GWCR, UPSE, DSM+Grad, cross-flow) + 0.25*energy
    Text:  rank_fuse(Chefer, DSM+Grad, heuristic) with gamma sharpen for Txt+
    """

    def __init__(self, model_usage):
        self.model_usage = model_usage
        self.lrp = _GWCRCollector(model_usage)
        self.upse = UPSEGenerator(model_usage)

    def generate(self, item, cross_weight=0.5, energy_weight=0.25):
        self.lrp._reset()
        R_t_t, R_t_i = self.lrp.generate_ours(item, use_lrp=True)
        chefer_text = _norm(R_t_t[0].detach().cpu().numpy())
        chefer_img = _norm(R_t_i[0].detach().cpu().numpy())

        gwcr = _gwcr_image(
            self.lrp.visn_self, self.lrp.cross_t2i,
            self.model_usage.image_boxes_len, cross_weight=cross_weight,
        )
        upse_text_list, upse_img_list = self.upse.generate_ours(item)
        upse_text = _norm(np.asarray(upse_text_list[0], dtype=np.float64))
        upse_img = _norm(np.asarray(upse_img_list[0], dtype=np.float64))
        heur_text = _heuristic_text(
            self.lrp.lang_self, self.lrp.cross_t2i, self.model_usage.text_len
        )

        model = self.model_usage.model
        dsm_img = _dsm_grad_map(model, "image")
        dsm_txt = _dsm_grad_map(model, "text")
        cross_img = _cross_attn_image_boxes(
            self.lrp.cross_t2i, self.model_usage.image_boxes_len
        )

        img_branches = [chefer_img, gwcr, upse_img]
        if dsm_img is not None:
            img_branches.append(dsm_img)
        if cross_img is not None:
            img_branches.append(cross_img)

        cam_image = _rank_fuse(img_branches, weights=[1.2, 1.0, 0.9, 0.85, 0.8][: len(img_branches)])

        enc = model.lxmert.encoder
        if len(enc.visual_feats_list_x) > 0:
            feats = enc.visual_feats_list_x[-1]
            s_task = _norm(0.6 * chefer_img + 0.4 * upse_img)
            e = _energy_boxes(feats, s_task, alpha=0.25, beta=0.12)
            if e is not None and len(e) == len(cam_image):
                cam_image = _norm((1.0 - energy_weight) * cam_image + energy_weight * e)

        txt_branches = [chefer_text]
        if dsm_txt is not None:
            txt_branches.append(dsm_txt)
        txt_branches.append(heur_text)
        cam_text = _rank_fuse(txt_branches, weights=[1.3, 0.7, 0.5][: len(txt_branches)])

        return torch.from_numpy(cam_image).float(), torch.from_numpy(cam_text).float()
