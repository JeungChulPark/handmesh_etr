#!/usr/bin/env bash
# Live RGB-D hand pose from the connected ZED 2i.
#   bash run_zed_handpose.sh                 # live GUI window
#   CKPT=mobrecon_ckpt/rgbd_lidar_0613_ft5e6/best.pt bash run_zed_handpose.sh
#   bash run_zed_handpose.sh --no-gui --save out/   # headless, dump overlays
#
# Checkpoint choice: the ZED is PASSIVE STEREO, so the stereo-domain model
# (rgbd_real_0613, trained on DexYCB/HO3D stereo) matches best for a ZED test.
# The rgbd_lidar_*_ft5e6 model is tuned for iPhone-LiDAR depth (the final target)
# and may read the ZED's denser stereo depth slightly differently.
set -euo pipefail
cd /home/jucpark/DeepLearning/handmesh_etr

CKPT="${CKPT:-mobrecon_ckpt/rgbd_real_0613/best.pt}"

exec python infer_rgbd_zed.py \
  --ckpt "$CKPT" \
  --source zed \
  --zed_resolution HD720 \
  --zed_fps 30 \
  --zed_depth_mode NEURAL \
  --zed_min_depth 0.3 --zed_max_depth 2.0 \
  "$@"
