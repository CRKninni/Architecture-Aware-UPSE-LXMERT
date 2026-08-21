#!/usr/bin/env bash
# Chefer LXMERT baseline — same protocol as upse_lxmert n3303 seed=1234
# Method: rm_with_lrp (Chefer relevance maps + LRP)
set -euo pipefail

REPO="/home/gen/crk/Unified-PCA-and-Spectral-based-Seed-Expansion-for-LXMERT"
cd "$REPO"
source ~/.virtualenvs/torch_112/bin/activate
export PYTHONPATH="$REPO"

COCO="${COCO:-/home/gen/crk/dataset/val2014/}"
NUM_SAMPLES="${NUM_SAMPLES:-3303}"
SEED="${SEED:-1234}"
# Prefer A6000 with most free VRAM; override with GPU=<id>
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
LOG="${LOG:-energy_chefer_lxmert_n3303.log}"
OUT="${OUT:-energy_lxmert_rm_with_lrp_n3303.json}"

pick_gpu() {
  if [[ -n "${GPU:-}" ]]; then
    echo "$GPU"
    return
  fi
  # Pick 49GB card (A6000) with max free memory, skip if upse_lxmert holds the card alone
  nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits \
    | awk -F', ' '$2 >= 48000 {print $1, $3}' \
    | sort -k2 -nr | head -1 | awk '{print $1}'
}

wait_for_vram() {
  local gpu="$1"
  while true; do
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [[ -n "$free_mb" && "$free_mb" -ge "$MIN_FREE_MB" ]]; then
      echo "$(date -Iseconds) GPU${gpu} free ${free_mb} MiB >= ${MIN_FREE_MB} — starting Chefer LXMERT" | tee -a "$LOG"
      return 0
    fi
    echo "$(date -Iseconds) GPU${gpu} free ${free_mb:-?} MiB < ${MIN_FREE_MB} — waiting 120s (upse_lxmert may be on GPU0)" | tee -a "$LOG"
    sleep 120
  done
}

GPU_ID="$(pick_gpu)"
if [[ -z "$GPU_ID" ]]; then
  echo "No 49GB GPU found" >&2
  exit 1
fi

echo "Chefer LXMERT rm_with_lrp n=${NUM_SAMPLES} seed=${SEED} GPU=${GPU_ID}" | tee "$LOG"
wait_for_vram "$GPU_ID"

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup python -u perturbation_comprehensive.py \
  --COCO_path "$COCO" \
  --method rm_with_lrp \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --gpu 0 \
  --output "$OUT" \
  >> "$LOG" 2>&1 &

echo "PID=$! log=$LOG out=$OUT"
