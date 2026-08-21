#!/usr/bin/env bash
# UPSE LXMERT n=3303 — negpush_prdx15 (best negpush sweep: Txt- +0.78, Img+ tied)
set -euo pipefail
cd /home/gen/crk/Unified-PCA-and-Spectral-based-Seed-Expansion-for-LXMERT
source ~/.virtualenvs/torch_112/bin/activate
export PYTHONPATH=.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR"

GPU="${GPU:-2}"
SEED="${SEED:-1234}"
NUM="${NUM:-3303}"
LOG="${LOG:-energy_upse_lxmert_n3303_v4.log}"
OUT="${OUT:-energy_lxmert_upse_lxmert_n3303_v4.json}"

echo "UPSE LXMERT negpush_prdx15 v4 n=${NUM} seed=${SEED} GPU=${GPU}" | tee "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" python -u perturbation_comprehensive.py \
  --COCO_path /home/gen/crk/dataset/val2014/ \
  --method upse_lxmert \
  --num-samples "$NUM" \
  --seed "$SEED" \
  --gpu 0 \
  --output "$OUT" \
  2>&1 | tee -a "$LOG"
