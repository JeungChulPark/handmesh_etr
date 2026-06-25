#!/usr/bin/env bash
# One-shot ZED SDK + pyzed installer for this box (Ubuntu 22.04, CUDA 12.8,
# conda env handmesh_etr / Python 3.10). The live infer_rgbd_zed.py needs the
# ZED SDK runtime (libsl_zed.so) + the pyzed Python binding.
#
# Run it as your normal user (NOT `sudo bash ...`) so pyzed lands in the conda
# env, not root's python. It will sudo only the system-level installer step and
# prompt you for your password there:
#     ! bash install_zed_sdk.sh
#
# If the download 404s, the version moved — grab the current
# "ZED SDK for Ubuntu 22 / CUDA 12" link from
#     https://www.stereolabs.com/developers/release/
# and re-run with e.g.  VER=5.0 bash install_zed_sdk.sh
set -euo pipefail

VER="${VER:-5.0}"                                   # ZED SDK major.minor (5.0 = cuda12.8 build, exact match)
URL="${URL:-https://download.stereolabs.com/zedsdk/${VER}/cu12/ubuntu22}"
RUN="/tmp/zed_sdk_${VER}.run"

if [ ! -s "$RUN" ]; then
  echo "[zed] downloading ZED SDK ${VER} (CUDA 12 / Ubuntu 22) ..."
  wget -q --show-progress -O "$RUN" "$URL"
fi
chmod +x "$RUN"

echo "[zed] installing to /usr/local/zed (sudo — enter your password) ..."
# `-- silent` auto-accepts the EULA and skips optional samples/CUDA prompts.
sudo "$RUN" -- silent skip_tools skip_cuda

echo "[zed] generating the pyzed binding into the ACTIVE (conda) env ..."
python -m pip install --upgrade pip
python /usr/local/zed/get_python_api.py || python -m pip install pyzed

echo "[zed] verifying ..."
python -c "import pyzed.sl as sl; print('pyzed OK ->', sl.Camera)"
echo "[zed] DONE.  Now run:  bash run_zed_handpose.sh"
