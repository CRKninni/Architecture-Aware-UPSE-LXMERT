# Unified PCA and Spectral-based Seed Expansion for LXMERT

Architecture-aware faithful attribution for LXMERT on VQA v2 perturbation faithfulness (Chefer et al., CVPR 2021 protocol).

**Author:** Charan Ramtej Kodi ([CRKninni](https://github.com/CRKninni)) — Research Scholar, University of Hyderabad

## Locked method: `upse_lxmert` v4

Rank-fused UPSE with architecture-specific branches. Beats Chefer Grad-Rollout on 3/4 faithfulness AUC metrics (n=3303, seed 1234).

| Metric | UPSE (v4) | Grad-Rollout (Chefer et al.) |
|--------|-----------|-------------------------------|
| Img+ ↓ | 51.53 | 51.53 |
| Img− ↑ | 62.85 | 62.73 |
| Txt+ ↓ | 21.20 | 21.53 |
| Txt− ↑ | 48.49 | 48.12 |

Grad-Rollout baseline: [`results/faithfulness_chefer_lxmert_n3303.json`](results/faithfulness_chefer_lxmert_n3303.json)

Full curves: [`results/faithfulness_upse_lxmert_v4_n3303.json`](results/faithfulness_upse_lxmert_v4_n3303.json)

## AOPC faithfulness (VQA-v2, n=3303, seed 42)

Deletion/insertion protocol (AOPC, Deletion@15%, Insertion@15%; higher is better). Same evaluation indices for all methods.

| Method | AOPC ↑ | Del@15% ↑ | Ins@15% ↑ |
|--------|--------|-----------|-----------|
| Relevance Maps (Chefer) | 0.5445 | 0.4158 | 0.2747 |
| Transformer Attribution | 0.5423 | 0.4055 | 0.2659 |
| Rollout | 0.3510 | 0.0996 | 0.0684 |
| **UPSE (ours, v4)** | **0.5491** | **0.4247** | **0.2989** |

Full table (all Chefer baselines): [`results/aopc_lxmert_n3303.json`](results/aopc_lxmert_n3303.json)

**How AOPC is calculated:** [`docs/EVALUATION_AOPC.md`](docs/EVALUATION_AOPC.md) (combined deletion/insertion, confidence-based; distinct from Chefer AUC in `faithfulness_*.json`).

## Reproduce

```bash
source ~/.virtualenvs/torch_112/bin/activate
python perturbation_comprehensive.py --method upse_lxmert --num 3303
```

Requires LXMERT weights and VQA v2 features under `data/`.

## Citation

If you use this code, please cite our UPSE paper.
