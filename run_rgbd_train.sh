#!/usr/bin/env bash
# Integrated multi-dataset RGB-D hand-pose training launcher.
# All dataset paths below are the verified on-disk roots (--check passed).
# Pipeline validated end-to-end by `python smoke_rgbd.py`.
#
#   bash run_rgbd_train.sh                 # default: real RGB-D run
#   bash run_rgbd_train.sh warmup          # ToF depth-branch warm-up (MSRA only)
#   tensorboard --logdir runs --port 6006  # watch metrics/visuals
set -euo pipefail

HD="/home/jucpark/DeepLearning/Datasets/Hand Dataset"
DEXYCB="$HD/DexYCB_full/data"
HO3D="$HD/HO3D_full"
MSRA="$HD/cvpr15_MSRAHandGestureDB"
# HanCo root is hardcoded in datasets/hanco_ty.py ($HD/HanCo); synthetic depth.

PRETRAIN="pretrain/100.pt"   # RGB checkpoint -> warm 3->4 stem graft
BATCH="${BATCH:-32}"
EPOCH="${EPOCH:-50}"
# Tuned for a warm-started fine-tune on the mixed datasets (the first 1e-4/w2d=1
# run diverged: train_mpjpe rose, best stuck at epoch 1). Lower LR preserves the
# warm-start; w2d=0.1 stops the intrinsics-dependent 2D loss from dominating.
LR="${LR:-1e-5}"
W2D="${W2D:-0.1}"
# Strong depth augmentation -> robustness to ZED passive-stereo holes + breaks the
# depth over-reliance the ablation exposed (model collapsed to 106mm w/o depth).
DEPTH_DROPOUT="${DEPTH_DROPOUT:-0.20}"
DEPTH_HOLES="${DEPTH_HOLES:-4}"
DEPTH_HOLE_FRAC="${DEPTH_HOLE_FRAC:-0.25}"
DEPTH_SIGMA="${DEPTH_SIGMA:-0.006}"
AUG="--lr $LR --w2d $W2D --depth_dropout $DEPTH_DROPOUT --depth_holes $DEPTH_HOLES --depth_hole_frac $DEPTH_HOLE_FRAC --depth_sigma $DEPTH_SIGMA"
# Final target is iPhone Pro LiDAR (ToF, coarse 256x192, weak on fingers): coarsen
# the stereo/synthetic training depth to look like LiDAR. On by default; LIDAR_SIM=0 to disable.
LIDAR_SIM="${LIDAR_SIM:-1}"
if [ "$LIDAR_SIM" = "1" ]; then
  AUG="$AUG --lidar_sim --lidar_downscale ${LIDAR_DOWNSCALE:-4} --lidar_edge_drop ${LIDAR_EDGE_DROP:-0.5}"
fi

MODE="${1:-main}"
case "$MODE" in
  main)
    # Real RGB + (real|synthetic) depth. These all carry real colour, so they
    # mix cleanly with the RGB-warm-started model. MSRA is excluded here on
    # purpose: its colour channel is depth-derived (pseudo-RGB) and would fight
    # the warm-started RGB branch — use the `warmup` mode for ToF instead.
    exec python train_mobrecon_rgbd.py \
      --exp rgbd_real_$(date +%m%d) \
      --datasets hanco,dexycb,ho3d \
      --dexycb_root "$DEXYCB" \
      --ho3d_root "$HO3D" \
      --depth_source render \
      --pretrain "$PRETRAIN" \
      $AUG \
      --batch "$BATCH" --epoch "$EPOCH"
    ;;
  warmup)
    # ToF depth-branch warm-up on MSRA (Femto-Bolt domain). Depth-only -> the
    # masked loss + pseudo-RGB are expected; run this BEFORE `main` and pass the
    # resulting checkpoint via --pretrain to seed a ToF-aware depth branch.
    exec python train_mobrecon_rgbd.py \
      --exp rgbd_tofwarm_$(date +%m%d) \
      --datasets msra \
      --msra_root "$MSRA" \
      --pretrain "$PRETRAIN" \
      --batch "$BATCH" --epoch 10
    ;;
  all)
    # Everything on disk (real RGB-D + ToF). Acceptable once you are NOT warm-
    # starting from a pure-RGB checkpoint, or for a depth-dominant model.
    exec python train_mobrecon_rgbd.py \
      --exp rgbd_all_$(date +%m%d) \
      --datasets hanco,dexycb,ho3d,msra \
      --dexycb_root "$DEXYCB" --ho3d_root "$HO3D" --msra_root "$MSRA" \
      --depth_source render --pretrain "$PRETRAIN" \
      $AUG \
      --batch "$BATCH" --epoch "$EPOCH"
    ;;
  *)
    echo "usage: bash run_rgbd_train.sh [main|warmup|all]"; exit 1 ;;
esac
