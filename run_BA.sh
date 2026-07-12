#!/usr/bin/env bash
# B then A, both domain-adapted (dexycb,ho3d,iphone), warm-started rgbd_real_0613.
#   B (locked): X,Y from MediaPipe uv (2D locked) + Z from backbone -> perfect 2D + backbone depth.
#   A (backbone + reproj loss): p_cnn + residual, with a 2D-reprojection-to-MediaPipe loss.
set -e
cd /home/jucpark/DeepLearning/handmesh_etr

echo "======== B: locked 2D (MediaPipe) + backbone Z ========"
python train_hybrid.py --exp hybrid_B --mode locked \
    --datasets dexycb,ho3d,iphone --epoch 40

echo "======== A: backbone-primary + 2D-reprojection loss ========"
python train_hybrid.py --exp hybrid_A --mode backbone --lambda_reproj 1e-6 --lambda_uv 0.01 \
    --datasets dexycb,ho3d,iphone --epoch 40
echo "======== DONE ========"
