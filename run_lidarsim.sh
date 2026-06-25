#!/usr/bin/env bash
# LiDAR-sim curriculum stage (final domain adaptation toward iPhone LiDAR).
# Continues the chain ds_hybrid_ft2 -> ds_realdepth2 -> ds_lidarsim: fine-tunes the
# real-depth-adapted model with LiDAR-simulation augmentation (coarse/blocky depth +
# finger/edge dropout + ToF noise) applied to all RGB-D sets, so the depth stream
# matches the low-res, hole-prone iPhone LiDAR statistics.
#
# Stable settings carried over from the real-depth fix: --w2d 0 (predicted-root 2D
# loss diverges on a shifted domain), --lr 1e-5 cosine, --clip 1.0.
set -e
cd "$(dirname "$0")"

HD="/home/jucpark/DeepLearning/Datasets/Hand Dataset"
DEXYCB="$HD/DexYCB_full/data"
HO3D="$HD/HO3D_full"
INIT="${INIT:-mobrecon_ckpt/ds_realdepth2/best.pt}"   # warm-start from real-depth-adapted model

python -u train_mobrecon_dualstream.py \
  --exp ds_lidarsim \
  --epoch 8 \
  --datasets hanco,dexycb,ho3d \
  --dexycb_root "$DEXYCB" \
  --ho3d_root "$HO3D" \
  --pretrain "$INIT" \
  --pose_in_chans 4 \
  --lr 1e-5 --cosine \
  --w2d 0 --clip 1.0 \
  --lidar_sim --lidar_downscale 4 --lidar_edge_drop 0.5 \
  --depth_source cache \
  --batch 32 --limit 5e6
