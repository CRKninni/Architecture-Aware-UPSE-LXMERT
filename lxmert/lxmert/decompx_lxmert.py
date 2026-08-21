"""
DecompX for LXMERT's dual-stream co-attention.

Every hidden state is carried as a decomposition h_i = sum_j h_{i<-j}, where j
indexes the model inputs (question tokens, object boxes, and one slot that
absorbs all biases). The decomposition is propagated through attention,
residuals, LayerNorm and the GELU feed-forwards, so the answer logit ends up
written as an exact sum of signed scalar contributions from each token and box.

Two properties distinguish this from attention-times-gradient relevance
propagation:

  * No aggregation heuristic. The classifier head is linear in the pooled
    vector, so the contribution of input j to the target logit is read off
    directly. There is no rollout, no head averaging, no |.| or L2 collapse.
  * One decomposition yields both modalities. Cross attention mixes the two
    streams' decompositions, so after the first cross layer a language state
    already carries box contributions and vice versa. The text map and the image
    map are two slices of the same object rather than two separate rule sets.

`verify_logit` checks the whole chain: the contributions must sum to the actual
model logit, which only holds if every propagation rule is exact.

Reference: Modarressi et al., "DecompX: Explaining Transformers Decisions by
Propagating Token Decomposition" (ACL 2023), extended here from a single encoder
stack to two streams with bidirectional cross attention.
"""

import numpy as np
import torch

# Elementwise nonlinearities are redistributed by the ratio act(p)/p, which is
# exact but undefined at p=0; fall back to the derivative there.
_GELU_AT_ZERO = 0.5
_TANH_AT_ZERO = 1.0
_RATIO_EPS = 1e-6


def _linear(dec, layer):
    """Apply a Linear to every decomposition slice, routing the bias to slot -1."""
    out = dec @ layer.weight.t()
    if layer.bias is not None:
        out[-1] = out[-1] + layer.bias
    return out


def _layer_norm(dec, ln):
    """
    Exact LayerNorm redistribution.

    The scale sigma is a property of the summed state, not of any single
    contribution, so it is computed once from dec.sum(0) and shared. Centering is
    linear and therefore applies per slice; the shift beta goes to the bias slot.
    """
    total = dec.sum(0)
    var = total.var(-1, unbiased=False, keepdim=True)
    denom = torch.sqrt(var + ln.eps)
    centered = dec - dec.mean(-1, keepdim=True)
    out = centered / denom * ln.weight
    out[-1] = out[-1] + ln.bias
    return out


def _act(dec, fn, at_zero):
    """Redistribute an elementwise nonlinearity so the slices still sum to fn(p)."""
    total = dec.sum(0)
    ratio = torch.where(
        total.abs() > _RATIO_EPS,
        fn(total) / torch.where(total.abs() > _RATIO_EPS, total, torch.ones_like(total)),
        torch.full_like(total, at_zero),
    )
    return dec * ratio


def _attention(dec_ctx, attn, probs):
    """
    Propagate vectors through fixed attention weights.

    Attention probabilities are treated as constants, as in the relevance
    propagation baselines; the difference is that what flows through them is a
    per-input vector rather than a scalar.
    """
    v = _linear(dec_ctx, attn.value)
    j, n_ctx, _ = v.shape
    heads, head_dim = attn.num_attention_heads, attn.attention_head_size
    v = v.view(j, n_ctx, heads, head_dim)
    a = probs[0].to(v.dtype)
    out = torch.einsum("hik,jkhd->jihd", a, v)
    return out.reshape(j, a.shape[1], heads * head_dim)


def _attn_output(dec_attn, dec_residual, out_mod):
    x = _linear(dec_attn, out_mod.dense)
    x = x + dec_residual
    return _layer_norm(x, out_mod.LayerNorm)


def _self_attn_layer(dec, layer):
    a = _attention(dec, layer.self, layer.self.get_attn())
    return _attn_output(a, dec, layer.output)


def _cross_attn_layer(dec_self, dec_ctx, layer):
    a = _attention(dec_ctx, layer.att, layer.att.get_attn())
    return _attn_output(a, dec_self, layer.output)


def _ffn(dec, inter, out_mod):
    x = _linear(dec, inter.dense)
    x = _act(x, inter.intermediate_act_fn, _GELU_AT_ZERO)
    x = _linear(x, out_mod.dense)
    x = x + dec
    return _layer_norm(x, out_mod.LayerNorm)


def _encoder_layer(dec, layer):
    dec = _self_attn_layer(dec, layer.attention)
    return _ffn(dec, layer.intermediate, layer.output)


def _x_layer(dec_l, dec_v, x):
    """One cross-modal block: bidirectional cross attention, then self, then FFN."""
    # Both directions read the pre-cross states, matching the forward pass.
    new_l = _cross_attn_layer(dec_l, dec_v, x.visual_attention)
    new_v = _cross_attn_layer(dec_v, dec_l, x.visual_attention_copy)
    dec_l, dec_v = new_l, new_v
    dec_l = _self_attn_layer(dec_l, x.lang_self_att)
    dec_v = _self_attn_layer(dec_v, x.visn_self_att)
    dec_l = _ffn(dec_l, x.lang_inter, x.lang_output)
    dec_v = _ffn(dec_v, x.visn_inter, x.visn_output)
    return dec_l, dec_v


def _head_decomposition(dec_l, model):
    """Push the [CLS] decomposition to the input of the final classifier Linear."""
    pooler = model.lxmert.pooler
    d = dec_l[:, :1, :]
    d = _linear(d, pooler.dense)
    d = _act(d, torch.tanh, _TANH_AT_ZERO)

    seq = model.answer_head.logit_fc
    d = _linear(d, seq[0])
    d = _act(d, seq[1], _GELU_AT_ZERO)
    d = _layer_norm(d, seq[2])
    return d[:, 0, :]


def _margin_drop(dec, w, logits, target):
    """
    Predicted loss of answer margin when each input is withheld.

    The decomposition is additive over inputs and the head is linear, so the
    logits the model would produce without input j are logits - dec_j @ W^T --
    for every class at once, not just the target. The decision margin against
    the strongest competitor follows, and how far that margin falls is a direct
    estimate of how much the answer depends on j.

    This is the quantity perturbation actually measures. Relevance propagation
    cannot express it: a single backward pass from the target logit discards the
    competing classes, so it can only say how much j supported the answer, never
    whether withholding j would change it. The runner-up is recomputed per input
    because withholding evidence can promote a different competitor.
    """
    contrib = dec @ w.t()

    masked = logits.clone()
    masked[target] = float("-inf")
    margin_full = logits[target] - masked.max()

    reduced = logits.unsqueeze(0) - contrib
    competitors = reduced.clone()
    competitors[:, target] = float("-inf")
    margin_without = reduced[:, target] - competitors.max(dim=1).values

    return margin_full - margin_without


def _score(dec, model, target, mode, logits):
    """
    Reduce the exact signed decomposition to an importance ranking.

    The plain modes answer "how much did input j add to the target logit", which
    is not the question perturbation asks. A token can push hard against the
    specific predicted answer while still being indispensable to the question, so
    a signed or magnitude reading of the target logit misranks it. `drop` asks
    the metric's own question instead.
    """
    final = model.answer_head.logit_fc[3]
    w = final.weight

    if mode in ("signed", "abs"):
        s = dec @ w[target]
    elif mode in ("margin", "margin_abs"):
        # Drop the class-agnostic component so the score reflects this answer
        # rather than overall answer plausibility.
        s = dec @ (w[target] - w.mean(0))
    elif mode == "norm":
        # Head-free variant: how much of the pooled representation input j owns.
        s = dec.norm(dim=-1)
    elif mode in ("drop", "drop_abs"):
        s = _margin_drop(dec, w, logits, target)
    else:
        raise ValueError(mode)

    if mode in ("abs", "margin_abs", "drop_abs"):
        s = s.abs()
    return s


SCORE_MODES = ("signed", "abs", "margin", "margin_abs", "norm", "drop", "drop_abs")


class DecompXGenerator:
    """Signed per-token and per-box contributions to the predicted answer logit."""

    def __init__(self, model_usage, mode="drop"):
        self.model_usage = model_usage
        self.mode = mode
        self.last_target = None
        self.last_logit = None
        self.last_sum = None

    def _capture_embeddings(self, item):
        model = self.model_usage.model
        emb = {}
        handles = [
            model.lxmert.embeddings.register_forward_hook(
                lambda m, i, o: emb.__setitem__("lang", o.detach())
            ),
            model.lxmert.encoder.visn_fc.register_forward_hook(
                lambda m, i, o: emb.__setitem__("visn", o.detach())
            ),
        ]
        try:
            with torch.no_grad():
                output = self.model_usage.forward(item).question_answering_score
        finally:
            for h in handles:
                h.remove()
        return emb["lang"], emb["visn"], output

    @torch.no_grad()
    def decompose(self, item, target=None):
        """Run the full decomposition; return the pre-classifier slices per input."""
        model = self.model_usage.model
        lang_emb, visn_emb, output = self._capture_embeddings(item)

        logits = output.reshape(-1).detach().float()
        if target is None:
            target = int(logits.argmax().item())
        self.last_target = target
        self.last_logit = float(logits[target].item())
        self._logits = logits

        n_lang = lang_emb.shape[1]
        n_visn = visn_emb.shape[1]
        dim = lang_emb.shape[2]
        n_slots = n_lang + n_visn + 1
        device, dtype = lang_emb.device, torch.float32

        dec_l = torch.zeros(n_slots, n_lang, dim, device=device, dtype=dtype)
        dec_v = torch.zeros(n_slots, n_visn, dim, device=device, dtype=dtype)
        idx_l = torch.arange(n_lang, device=device)
        idx_v = torch.arange(n_visn, device=device)
        dec_l[idx_l, idx_l] = lang_emb[0].to(dtype)
        dec_v[n_lang + idx_v, idx_v] = visn_emb[0].to(dtype)

        encoder = model.lxmert.encoder
        for layer in encoder.layer:
            dec_l = _encoder_layer(dec_l, layer)
        for layer in encoder.r_layers:
            dec_v = _encoder_layer(dec_v, layer)
        for x in encoder.x_layers:
            dec_l, dec_v = _x_layer(dec_l, dec_v, x)

        dec = _head_decomposition(dec_l, model)

        # Exactness check: the signed logit contributions must sum to the logit.
        signed = dec @ model.answer_head.logit_fc[3].weight[target]
        bias = model.answer_head.logit_fc[3].bias
        self.last_sum = float(signed.sum().item() + (bias[target] if bias is not None else 0.0))

        self._dec = dec
        self._n_lang = n_lang
        self._n_visn = n_visn
        return dec

    @torch.no_grad()
    def score(self, mode=None):
        """Split a completed decomposition into image and text maps under `mode`."""
        mode = mode or self.mode
        s = _score(self._dec, self.model_usage.model, self.last_target, mode, self._logits)
        s = s.detach().cpu().numpy().astype(np.float64)
        return s[self._n_lang : self._n_lang + self._n_visn], s[: self._n_lang]

    def contributions(self, item, target=None, mode=None):
        self.decompose(item, target=target)
        return self.score(mode)

    def all_modes(self, item):
        """Every score mode from a single decomposition, for cheap comparison."""
        self.decompose(item)
        return {m: self.score(m) for m in SCORE_MODES}

    def verify_logit(self, item):
        """Reconstruction error of the decomposition against the real logit."""
        self.decompose(item)
        return {
            "target": self.last_target,
            "logit": self.last_logit,
            "decomposed_sum": self.last_sum,
            "abs_error": abs(self.last_logit - self.last_sum),
            "rel_error": abs(self.last_logit - self.last_sum) / (abs(self.last_logit) + 1e-8),
        }

    def generate(self, item):
        image, text = self.contributions(item)
        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(text).float(),
        )
