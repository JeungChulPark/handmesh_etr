#!/usr/bin/env bash
# Real-depth curriculum for the dual-stream hybrid model.
# Fine-tunes the best HanCo model (ds_hybrid_ft2) on a HanCo + real-sensor-depth
# mix (DexYCB + HO3D), so the depth stream adapts from synthetic render to real
# sensor noise/holes -- the domain it must handle on the iPhone LiDAR target.
set -e
cd "$(dirname "$0")"

HD="/home/jucpark/DeepLearning/Datasets/Hand Dataset"
DEXYCB="$HD/DexYCB_full/data"
HO3D="$HD/HO3D_full"
INIT="${INIT:-mobrecon_ckpt/ds_hybrid_ft2/best.pt}"   # warm-start from best HanCo model

# 8 epochs (fine-tune/domain-adaptation converges fast) + early-stop monitor
# applied separately at launch (patience 3, eps 0.1mm on abs).
#
# Stability (after ds_realdepth v1 diverged): the 2D reprojection loss uses the
# PREDICTED root, which blows up on a new domain before the root adapts and drags
# the whole backbone down -> disable it (--w2d 0; 3D L_pose/L_root/L_abs already
# supervise everything). Plus conservative LR (1e-5, proven on HanCo FT) and
# grad-norm clipping (--clip 1.0) as a hard divergence guard.
python -u train_mobrecon_dualstream.py \
  --exp ds_realdepth2 \
  --epoch 8 \
  --datasets hanco,dexycb,ho3d \
  --dexycb_root "$DEXYCB" \
  --ho3d_root "$HO3D" \
  --pretrain "$INIT" \
  --pose_in_chans 4 \
  --lr 1e-5 --cosine \
  --w2d 0 --clip 1.0 \
  --depth_source cache \
  --batch 32 --limit 5e6
