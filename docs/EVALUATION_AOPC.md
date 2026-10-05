# AOPC evaluation protocol (LXMERT)

This document describes how **AOPC**, **Deletion@15%**, and **Insertion@15%** in [`results/aopc_lxmert_n3303.json`](../results/aopc_lxmert_n3303.json) are computed. It matches the UPSE harness in `documentation/eval_aopc/` (same logic as `eval_lxmert_upse_aopc.py`).

## What this is (and is not)

| Metric family | Protocol | Used in this repo |
|---------------|----------|-------------------|
| **AOPC / Del@15 / Ins@15** | Combined image+text deletion & insertion on target-class **confidence** | **Yes** — tables in README + `aopc_lxmert_n3303.json` |
| **Chefer AUC** (Img+/−, Txt+/−) | Accuracy vs. perturbation fraction | **Separate** — `faithfulness_*_n3303.json` via `perturbation_comprehensive.py` |

Do not mix AOPC numbers with Chefer AUC tables.

## Data and sample set

- **Dataset:** VQA v2 validation questions (`data/vqa/valid.json` or project `valid.json`).
- **Size:** `n = 3303` questions.
- **Selection:** All validation indices are shuffled with **`random.seed(42)`**, then the first 3303 indices are taken (fixed list stored in each run’s `run_metadata.json` / `summary_metrics.json` under `selected_indices`).
- **Images:** COCO **val2014** paths used by the LXMERT pipeline.

## Target score

For each question, the model predicts an answer class. Faithfulness uses the **softmax confidence of the predicted (argmax) class**:

- `baseline_conf` — confidence on the **unperturbed** input.
- Perturbations re-run the forward pass and read the same target class index.

## Attribution maps

For each method (UPSE `upse_lxmert`, Relevance Maps, Rollout, etc.), we obtain non-negative **token/region scores**:

- **Text:** one score per text token (special tokens can be zeroed in the map).
- **Image:** one score per visual region/box token.

Maps are ranked **independently** per modality (top fraction of text tokens and top fraction of image regions).

## Combined perturbation (one curve for both modalities)

At each percentage `k ∈ {5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100}`:

1. Let `fraction = k / 100`.
2. **Text:** among eligible text indices, keep the top `⌈fraction × |candidates|⌉` by attribution score.
3. **Image:** same for image region indices.

**Deletion curve (combined):** remove attributed tokens/regions **not** in the top‑k set; always retain required boundaries (e.g. text `[0, len−1]` structure as in the eval code).

**Insertion curve (combined):** start from an **empty/minimal** input and **add** the top‑k attributed tokens/regions.

- `empty_conf` — confidence on minimal input (text boundaries + minimal image placeholder).
- At step `k`, deletion confidence `del_conf`, insertion confidence `ins_conf`.

Per-point **drop** (deletion branch):

\[
\text{drop}(k) = \text{baseline\_conf} - \text{del\_conf}(k)
\]

## Reported metrics (higher is better)

Implemented in `metrics_from_curves` (UPSE `aopc_common.py`):

1. **AOPC (`combined_aopc`)** — *area over the perturbation curve* on the **combined deletion** curve:
   \[
   \text{AOPC} = \frac{1}{|K|} \sum_{k \in K} \text{drop}(k)
   \]
   where `K` is the list of `k` values above (default 12 points).

2. **Deletion@15% (`combined_drop_conf`)** — \(\text{drop}(k)\) at **`k = 15`**.

3. **Insertion@15% (`combined_increase_conf`)** — at **`k = 15`** on the **insertion** curve:
   \[
   \text{Ins@15} = \text{ins\_conf}(15) - \text{empty\_conf}
   \]
   (confidence gain over the empty baseline, not “area under”.)

Dataset-level numbers are the **mean** over all 3303 samples.

## Locked UPSE configuration

- **Attribution method:** `upse_lxmert` v4 (same maps as Chefer AUC runs in `faithfulness_upse_lxmert_v4_n3303.json`).
- **AOPC run defaults:** `seed=42`, `drop_k=15`, `k_values` as above.

## Reproduce AOPC (full harness)

Requires the UPSE monorepo `documentation/eval_aopc` (not duplicated in this GitHub repo):

```bash
source ~/.virtualenvs/torch_112/bin/activate
cd /path/to/UPSE/documentation/eval_aopc
python eval_lxmert_upse_aopc.py \
  --method upse_lxmert \
  --num-cases 3303 \
  --seed 42 \
  --gpu 0
```

Outputs: `results/lxmert/upse_lxmert_seed42_n3303_<timestamp>/summary_metrics.json` and per-sample curves under `per_sample/`.

Baselines: same script with `--method rm_with_lrp`, `rollout`, `transformer_att`, etc. (see `chefer_methods.py`).

## Citation

If you use these metrics, cite the UPSE paper and note the combined deletion/insertion AOPC protocol above.
