#!/usr/bin/env bash
# Robust overnight launcher for the confidence-gated fine-tune (ds_anchor_gate).
# This machine's GPU is shared with the display, so long kernels occasionally trip
# the driver watchdog ("CUDA error: the launch timed out"). We checkpoint every
# epoch (latest.pt) and auto-resume from it, so a transient kill costs <= 1 epoch.
# batch 32 (proven stable for ds_anchor_lidar's 10 epochs) keeps kernels short.
set -u
cd /home/jucpark/DeepLearning/handmesh_etr
HD="/home/jucpark/DeepLearning/Datasets/Hand Dataset"
EXP=ds_anchor_gate
CKDIR="mobrecon_ckpt/$EXP"
LOG="logs/${EXP}.log"
mkdir -p "$CKDIR"
echo $$ > "$CKDIR/wrapper.pid"

for attempt in $(seq 1 40); do
  if [ -f "$CKDIR/latest.pt" ]; then
    INIT="--resume $CKDIR/latest.pt"          # continue (restores opt+sched+epoch)
  else
    INIT="--pretrain mobrecon_ckpt/ds_anchor_lidar/best.pt"  # first start: warm-start
  fi
  echo "[wrapper] attempt $attempt  $(date '+%F %T')  $INIT" >> "$LOG"
  python train_mobrecon_dualstream.py --exp "$EXP" \
    --datasets hanco,dexycb,ho3d \
    --dexycb_root "$HD/DexYCB_full/data" --ho3d_root "$HD/HO3D_full" \
    $INIT --pose_in_chans 4 --depth_source cache \
    --lidar_sim --clip 1.0 --lr 2e-5 --cosine --epoch 20 --batch 32 --workers 12 \
    >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[wrapper] DONE (rc=0) after $attempt attempt(s)  $(date '+%F %T')" >> "$LOG"
    break
  fi
  echo "[wrapper] crash rc=$rc; resuming from latest.pt in 60s  $(date '+%F %T')" >> "$LOG"
  sleep 60
done
rm -f "$CKDIR/wrapper.pid"
