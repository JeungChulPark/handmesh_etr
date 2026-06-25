#!/usr/bin/env bash
# Resume the rgbd_lidar_0613_ft5e6 fine-tune (lr 5e-6, LiDAR-sim) from the last
# good checkpoint. 4.pt = epoch 4 with optimizer+scheduler+best(28.45mm) state,
# so this continues epochs 5..19 toward the target of 20. Re-runnable: just point
# --resume at the newest N.pt if it stops again.
set -euo pipefail
cd /home/jucpark/DeepLearning/handmesh_etr

HD="/home/jucpark/DeepLearning/Datasets/Hand Dataset"
DEXYCB="$HD/DexYCB_full/data"
HO3D="$HD/HO3D_full"

CKPT="${CKPT:-mobrecon_ckpt/rgbd_lidar_0613_ft5e6/4.pt}"

exec python train_mobrecon_rgbd.py \
  --exp rgbd_lidar_0613_ft5e6 \
  --datasets hanco,dexycb,ho3d \
  --dexycb_root "$DEXYCB" \
  --ho3d_root "$HO3D" \
  --depth_source render \
  --resume "$CKPT" \
  --lr 5e-6 --w2d 0.1 \
  --depth_dropout 0.20 --depth_holes 4 --depth_hole_frac 0.25 --depth_sigma 0.006 \
  --lidar_sim --lidar_downscale 4 --lidar_edge_drop 0.5 \
  --batch 32 --epoch 20
