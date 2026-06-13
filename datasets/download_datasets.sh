#!/usr/bin/env bash
# Download the publicly-scriptable hand datasets (no login required).
# Usage: bash datasets/download_datasets.sh <contactpose|nyu> <dest_dir>
# The license-gated sets (DexYCB, HO3D, H2O-3D, BigHand, HANDS17, FPHA, ICVL,
# OakInk) require your own account -> see datasets/DOWNLOAD.md.
set -euo pipefail

NAME="${1:-}"
DEST="${2:-}"
if [[ -z "$NAME" || -z "$DEST" ]]; then
  echo "usage: bash datasets/download_datasets.sh <contactpose|nyu> <dest_dir>"
  exit 1
fi
mkdir -p "$DEST"

case "$NAME" in
  contactpose)
    # ~140 GB for full RGB-D. Public; no login. Uses the official toolkit script.
    echo "[contactpose] cloning toolkit into $DEST/ContactPose ..."
    if [[ ! -d "$DEST/ContactPose" ]]; then
      git clone https://github.com/facebookresearch/ContactPose "$DEST/ContactPose"
    fi
    cd "$DEST/ContactPose"
    pip install -r requirements.txt
    echo "[contactpose] downloading grasps + RGB-D images (this is large) ..."
    # grasps (small) then images; restrict with --p_nums in the loader, not here.
    python scripts/download_data.py --type grasps
    python scripts/download_data.py --type images
    echo "[contactpose] done -> point --contactpose_dir at $DEST/ContactPose/data/contactpose_data"
    ;;
  nyu)
    # ~92 GB zip. Public direct link.
    URL="https://cs.nyu.edu/~tompson/data/nyu_hand_dataset_v2.zip"
    echo "[nyu] downloading $URL ..."
    wget -c -O "$DEST/nyu_hand_dataset_v2.zip" "$URL"
    echo "[nyu] extracting ..."
    unzip -n "$DEST/nyu_hand_dataset_v2.zip" -d "$DEST"
    echo "[nyu] done -> point --nyu_root at $DEST/dataset/train"
    ;;
  *)
    echo "unknown or login-gated dataset '$NAME' — see datasets/DOWNLOAD.md"
    exit 1
    ;;
esac
