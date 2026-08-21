"""
Tier-1 interpretability branches (single backward pass) for LXMERT boxes/tokens.

  - Grad×Input / ε-LRP on attention maps
  - Attention flow (TAT-style rollout)
  - NOTICE head-scored cross/self attention
  - GLIMPSE-style layer importance blend
  - VPS-lite marginal box scoring (36 FRCNN boxes)
"""

import numpy as np
import torch

from lxmert.lxmert.lxmert_upse_e import _head_weighted_grad_attn, _norm


def _to_2d_cam(grad, cam):
    """Head-mean grad×attn relevance matrix."""
    g = grad.detach().clamp(min=0)
    a = cam.detach()
    if g.dim() == 4:
        g, a = g[0], a[0]
    w = g.sum(dim=(1, 2))
    w = w / (w.sum() + 1e-8)
    r = (g * a * w[:, None, None]).sum(dim=0)
    return torch.clamp(r, min=0)


def grad_x_input_boxes(pairs, n_boxes):
    """Grad×Input (ε-LRP) — column mass to boxes from self-attn stack."""
    acc = None
    for grad, cam in pairs:
        r = _to_2d_cam(grad, cam).cpu().numpy()
        if r.shape[0] > n_boxes:
            r = r[:n_boxes, :n_boxes]
        col = np.maximum(r.sum(axis=0), 0.0)
        if col.shape[0] > n_boxes:
            col = col[:n_boxes]
        acc = col if acc is None else acc + col
    return _norm(acc) if acc is not None else None


def attention_flow_boxes(pairs, n_boxes):
    """TAT-style flow: chain (0.5·I + 0.5·A) per layer, box receive mass."""
    flow = None
    for grad, cam in pairs:
        a = cam.detach().float()
        if a.dim() == 4:
            a = a[0].mean(dim=0)
        elif a.dim() == 3:
            a = a.mean(dim=0)
        n = a.shape[0]
        eye = torch.eye(n, device=a.device)
        m = 0.5 * eye + 0.5 * a
        m = m / (m.sum(dim=-1, keepdim=True) + 1e-8)
        flow = m if flow is None else m @ flow
    if flow is None:
        return None
    recv = flow.sum(dim=0).cpu().numpy()
    if recv.shape[0] > n_boxes:
        recv = recv[:n_boxes]
    return _norm(np.maximum(recv, 0.0))


def notice_head_scored_cross(cross_pairs, n_boxes, keep_frac=0.55):
    """NOTICE — keep top grad-mass cross-attn heads, aggregate text→box flow."""
    acc = None
    for grad, cam in cross_pairs:
        g = grad.detach().clamp(min=0)
        a = cam.detach()
        if g.dim() == 4:
            nh = g.shape[1]
            head_mass = g.abs().sum(dim=(0, 2, 3)).cpu().numpy()
            k = max(1, int(np.ceil(nh * keep_frac)))
            top = np.argsort(-head_mass)[:k]
            t2i = torch.zeros_like(g[0, 0])
            for h in top:
                wh = g[0, h].sum().clamp(min=1e-8)
                t2i = t2i + (g[0, h] * a[0, h]).sum(dim=0) * (wh / (g[0].sum() + 1e-8))
        else:
            t2i = _to_2d_cam(grad, cam)
        arr = t2i.cpu().numpy()
        if arr.shape[1] > n_boxes:
            arr = arr[:, :n_boxes]
        flow = arr[1:-1].sum(axis=0) if arr.shape[0] > 2 else arr.sum(axis=0)
        flow = np.maximum(flow, 0.0)
        acc = flow if acc is None else acc + flow
    return _norm(acc) if acc is not None else None


def glimpse_layer_blend(pairs, n_boxes, decay=0.82):
    """GLIMPSE-lite — later x_layers weighted more in self-attn box scores."""
    acc = None
    w_sum = 0.0
    for i, (grad, cam) in enumerate(pairs):
        w = decay ** (len(pairs) - 1 - i)
        r = _to_2d_cam(grad, cam).cpu().numpy()
        if r.shape[0] > n_boxes:
            r = r[:n_boxes, :n_boxes]
        col = np.maximum(r.sum(axis=0), 0.0)[:n_boxes]
        acc = w * col if acc is None else acc + w * col
        w_sum += w
    if acc is None:
        return None
    return _norm(acc / max(w_sum, 1e-8))


def ig_lite_boxes(feats, task_prior, steps=6):
    """IG-lite: task_prior × feature norm (proxy without extra backwards)."""
    if feats is None or feats.numel() == 0:
        return None
    x = feats.detach().float()
    if x.dim() == 3:
        x = x.squeeze(0)
    fn = torch.norm(x, p=2, dim=-1).cpu().numpy()
    fn = _norm(fn)
    tp = _norm(np.asarray(task_prior, dtype=np.float64)[: len(fn)])
    return _norm(fn * tp)


def rit_text_scores(r_i_t, text_len):
    """
    Image->text relevance column mass.

    Chefer propagates R_i_t through every cross-modal layer but only returns
    R_t_t / R_t_i, so this vision-grounded per-token signal is never used.
    """
    if r_i_t is None:
        return None
    r = r_i_t.detach().float().cpu().numpy()
    col = np.maximum(r, 0.0).sum(axis=0)
    if col.shape[0] > text_len:
        col = col[:text_len]
    return _norm(col) if col.sum() > 1e-12 else None


def cross_i2t_text_scores(cross_i2t, text_len):
    """Grad-weighted image->text cross attention aggregated per text token."""
    acc = None
    for grad, cam in cross_i2t:
        i2t = _head_weighted_grad_attn(grad, cam).detach().cpu().numpy()
        if i2t.shape[1] > text_len:
            i2t = i2t[:, :text_len]
        flow = np.maximum(i2t.sum(axis=0), 0.0)
        s = flow.sum()
        if s < 1e-12:
            continue
        flow = flow / s
        acc = flow if acc is None else acc + flow
    return _norm(acc) if acc is not None else None


def notice_head_scored_text(cross_i2t, text_len, keep_frac=0.55):
    """NOTICE head selection on the image->text direction."""
    acc = None
    for grad, cam in cross_i2t:
        g = grad.detach().clamp(min=0)
        a = cam.detach()
        if g.dim() == 4:
            nh = g.shape[1]
            head_mass = g.abs().sum(dim=(0, 2, 3)).cpu().numpy()
            k = max(1, int(np.ceil(nh * keep_frac)))
            top = np.argsort(-head_mass)[:k]
            i2t = torch.zeros_like(g[0, 0])
            for h in top:
                wh = g[0, h].sum().clamp(min=1e-8)
                i2t = i2t + (g[0, h] * a[0, h]) * (wh / (g[0].sum() + 1e-8))
        else:
            i2t = _to_2d_cam(grad, cam)
        arr = i2t.cpu().numpy()
        if arr.shape[1] > text_len:
            arr = arr[:, :text_len]
        flow = np.maximum(arr.sum(axis=0), 0.0)
        acc = flow if acc is None else acc + flow
    return _norm(acc) if acc is not None else None


def vps_marginal_boxes(chefer, cross, self_rel, collaboration=0.35):
    """
    VPS-lite marginal score on 36 boxes.
    clue = chefer; collaboration = cross vs self agreement boost.
    """
    c = _norm(chefer)
    parts = [c]
    if cross is not None:
        x = _norm(cross)
        parts.append(collaboration * _norm(np.sqrt(c * x + 1e-8)))
    if self_rel is not None:
        s = _norm(self_rel)
        parts.append(0.15 * _norm(np.sqrt(c * s + 1e-8)))
    return _norm(np.sum(parts, axis=0))
