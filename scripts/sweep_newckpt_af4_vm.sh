#!/usr/bin/env bash
# scripts/sweep_newckpt_af4_vm.sh
#
# Phase 1: evaluate the NEW GPU-trained checkpoint (dimo_cond_r4_gpu_b2/epoch_0040.pt)
# on AF4 only, cases 0/1/2 x {safe, stage2_heavier, aggressive_equal}, writing to a
# SEPARATE output root so it never collides with the old checkpoint's results.
#
# Do NOT tune schedules here. First answer one question: does epoch_0040 improve AF4
# vs the old epoch_0020? Only expand to AF8/AF10 if AF4 improves (see bottom).
#
# Usage (on the VM):
#   cd ~/T1_DDPM_Project && source .venv/bin/activate
#   bash scripts/sweep_newckpt_af4_vm.sh
#
# If CUDA is out of memory because training/other jobs hold the GPU, set DEVICE=cpu:
#   DEVICE=cpu bash scripts/sweep_newckpt_af4_vm.sh

set -euo pipefail

PROJECT="${PROJECT:-$HOME/T1_DDPM_Project}"
CKPT="${CKPT:-$PROJECT/checkpoints/dimo_cond_r4_gpu_b2/epoch_0040.pt}"
ACC_ROOT="${ACC_ROOT:-$PROJECT/ChallengeData/SingleCoil/Mapping/TrainingSet}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/outputs/sweeps_gpu_b2_alt}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT"

echo "[INFO] project : $PROJECT"
echo "[INFO] ckpt    : $CKPT"
echo "[INFO] out_root: $OUT_ROOT"
echo "[INFO] device  : $DEVICE"

python -m src.tools.sweep_alternating \
  --ckpt "$CKPT" \
  --acc_root "$ACC_ROOT" \
  --out_root "$OUT_ROOT" \
  --project "$PROJECT" \
  --acc_factors 04 \
  --indices 0 1 2 \
  --schedules safe stage2_heavier aggressive_equal \
  --cycles 3 \
  --cond_mode zf_mask \
  --dc_mode replace \
  --scale_mode auto \
  --best_metric nmse_mag \
  --device "$DEVICE"

echo
echo "[NEXT] Aggregate the new-checkpoint results into one table:"
echo "  python -m src.tools.aggregate_results --root $OUT_ROOT"
echo
echo "[NEXT] Compare against the OLD checkpoint (outputs/sweeps_cpu_alt) side by side,"
echo "       and render panels with error maps:"
echo "  python -m src.tools.make_panels --root $OUT_ROOT"
echo
echo "[ONLY IF AF4 IMPROVES] expand to higher acceleration:"
echo "  python -m src.tools.sweep_alternating --ckpt \"$CKPT\" --acc_root \"$ACC_ROOT\" \\"
echo "    --out_root \"$OUT_ROOT\" --project \"$PROJECT\" --acc_factors 08 10 --indices 0 1 2 --device $DEVICE"
