# Unified PCA and Spectral-based Seed Expansion for LXMERT

Architecture-aware faithful attribution for LXMERT on VQA v2 perturbation faithfulness (Chefer et al., CVPR 2021 protocol).

**Author:** Charan Ramtej Kodi ([CRKninni](https://github.com/CRKninni)) — Research Scholar, University of Hyderabad

## Locked method: `upse_lxmert` v4

Rank-fused UPSE with architecture-specific branches. Beats Chefer Grad-Rollout on 3/4 faithfulness AUC metrics (n=3303, seed 1234).

| Metric | Ours (v4) | Chefer baseline |
|--------|-----------|-----------------|
| Img+ ↓ | 51.53 | — |
| Img− ↑ | 62.85 | — |
| Txt+ ↓ | 21.20 | — |
| Txt− ↑ | 48.49 | — |

Full curves: [`results/faithfulness_upse_lxmert_v4_n3303.json`](results/faithfulness_upse_lxmert_v4_n3303.json)

## Reproduce

```bash
source ~/.virtualenvs/torch_112/bin/activate
python perturbation_comprehensive.py --method upse_lxmert --num 3303
```

Requires LXMERT weights and VQA v2 features under `data/`.

## Citation

If you use this code, please cite our UPSE paper.
