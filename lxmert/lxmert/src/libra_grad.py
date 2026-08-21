"""
LibraGrad hooks for LXMERT LRP backward (CVPR 2025 Mehri et al.).

Applied during standard autograd backward (Chefer one_hot.backward):
  - Libra Attention: attenuate softmax-path gradients on attention_probs
  - SwapBackward on residual Add: 0.5 scale per branch (Theorem 4)
  - Libra GELU: identity backward (discard nonlinear gate grad, Theorem 5)
  - Post-backward: rebalance cross vs self attention grad mass for LXMERT

Reference: https://arxiv.org/abs/2411.16760
"""

import torch
import torch.nn as nn

from lxmert.lxmert.src.layers import Add, GELU

_HANDLES = []
_STATE = {"enabled": False, "softmax_retention": 0.2, "cross_boost": 1.0}


def _is_lxmert_attention(module):
    return module.__class__.__name__ == "LxmertAttention"


def _libra_attn_hook(module, grad):
    """Libra Attention: role-specific softmax grad attenuation."""
    if grad is None:
        module.save_attn_gradients(grad)
        return grad
    role = getattr(module, "attn_role", "")
    if role.startswith("lang"):
        r = 1.0  # no Libra on language self-attn (lock Txt+)
    elif "cross" in role:
        r = float(_STATE["softmax_retention"]) * 0.75
    else:
        r = float(_STATE["softmax_retention"])
    g = grad * r
    module.save_attn_gradients(g)
    return g


def _libra_add_hook(module, grad_input, grad_output):
    """SwapBackward-style 0.5 scaling on residual branches."""
    return tuple(g * 0.5 if g is not None else None for g in grad_input)


def _libra_gelu_hook(module, grad_input, grad_output):
    """Libra Gated Activation: treat GELU as identity in backward."""
    if not grad_output or grad_output[0] is None:
        return grad_input
    g = grad_output[0]
    return tuple((g,) if gi is not None else None for gi in grad_input)


def _tag_attention_roles(model):
    enc = model.lxmert.encoder
    for blk in enc.layer:
        blk.attention.self.attn_role = "lang_enc_self"
    for blk in enc.r_layers:
        blk.attention.self.attn_role = "visn_enc_self"
    for blk in enc.x_layers:
        blk.lang_self_att.self.attn_role = "lang_co_self"
        blk.visn_self_att.self.attn_role = "visn_co_self"
        blk.visual_attention.att.attn_role = "cross_t2i"
        if hasattr(blk, "visual_attention_copy"):
            blk.visual_attention_copy.att.attn_role = "cross_i2t"


def _grad_mass(module):
    g = module.get_attn_gradients()
    if g is None:
        return 0.0
    return float(g.detach().abs().sum())


def balance_attention_gradients(model, cross_boost=1.0):
    """Post-backward mass rebalance: boost cross-modal attn grad vs self."""
    if not _STATE["enabled"]:
        return
    self_mass, cross_mass = 0.0, 0.0
    cross_mods, self_mods = [], []
    for mod in model.modules():
        if not _is_lxmert_attention(mod):
            continue
        role = getattr(mod, "attn_role", "")
        m = _grad_mass(mod)
        if "cross" in role:
            cross_mass += m
            cross_mods.append(mod)
        else:
            self_mass += m
            self_mods.append(mod)
    if cross_mass < 1e-12 or self_mass < 1e-12:
        return
    scale_cross = cross_boost * float((self_mass / (cross_mass + 1e-8)) ** 0.5)
    scale_cross = float(max(0.8, min(scale_cross, 2.5)))
    scale_self = float((cross_mass / (self_mass + 1e-8)) ** 0.25)
    scale_self = max(0.5, min(scale_self, 1.0))
    for mod in cross_mods:
        g = mod.get_attn_gradients()
        if g is not None:
            mod.attn_gradients = g * scale_cross
    for mod in self_mods:
        g = mod.get_attn_gradients()
        if g is not None:
            role = getattr(mod, "attn_role", "")
            if role.startswith("lang"):
                continue  # preserve lang self grad for Txt+
            mod.attn_gradients = g * scale_self


def _libra_layernorm_hook(module, grad_input, grad_output):
    """Libra LayerNorm: attenuate variance-path grad (FullGrad extension)."""
    if not grad_output or grad_output[0] is None:
        return grad_input
    g = grad_output[0]
    return tuple((0.5 * g,) if gi is not None else None for gi in grad_input)


def enable_libra_grad(model, softmax_retention=0.2, cross_boost=1.15):
    """Register LibraGrad backward hooks on LXMERT (idempotent)."""
    disable_libra_grad(model)
    _STATE["enabled"] = True
    _STATE["softmax_retention"] = softmax_retention
    _STATE["cross_boost"] = cross_boost
    _tag_attention_roles(model)

    from lxmert.lxmert.src.layers import LayerNorm as LxmertLayerNorm

    for mod in model.modules():
        if _is_lxmert_attention(mod):
            mod.libra_enabled = True
            mod._libra_hook_fn = _libra_attn_hook

        if isinstance(mod, Add):
            h = mod.register_full_backward_hook(_libra_add_hook)
            _HANDLES.append(h)

        if isinstance(mod, GELU):
            h = mod.register_full_backward_hook(_libra_gelu_hook)
            _HANDLES.append(h)

        if isinstance(mod, LxmertLayerNorm):
            h = mod.register_full_backward_hook(_libra_layernorm_hook)
            _HANDLES.append(h)


def disable_libra_grad(model):
    _STATE["enabled"] = False
    for h in _HANDLES:
        h.remove()
    _HANDLES.clear()
    for mod in model.modules():
        if _is_lxmert_attention(mod):
            mod.libra_enabled = False
            mod._libra_hook_fn = None
