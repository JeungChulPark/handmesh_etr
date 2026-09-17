#!/usr/bin/env bash
# Data-scaling probe: 2x the training frames (DexYCB stride 8->4, HO3D 4->2), same
# held-out subjects/sequences, same model/loss. Tests whether MORE data moves the
# combined 37.43mm -- if it barely moves, the bottleneck is depth quality (not data).
set -e
cd /home/jucpark/DeepLearning/handmesh_etr
DEXYCB="/home/jucpark/DeepLearning/Datasets/Hand Dataset/DexYCB_full/data"
HO3D="/home/jucpark/DeepLearning/Datasets/Hand Dataset/HO3D_full"

echo "==== cache DexYCB stride 4 (2x) ===="
python scripts/capture/cache_mediapipe.py --dataset dexycb --root "$DEXYCB" --stride 4 --out cache/mp_dexycb_s4.npz
echo "==== cache HO3D stride 2 (2x) ===="
python scripts/capture/cache_mediapipe.py --dataset ho3d --root "$HO3D" --stride 2 --out cache/mp_ho3d_s2.npz
echo "==== train combined (2x data) ===="
python scripts/train/train_zlifter.py --exp zlifter_probe2x --datasets dexycb,ho3d \
    --cache cache/mp_dexycb_s4.npz --ho3d_cache cache/mp_ho3d_s2.npz --epoch 60
echo "==== DONE ===="
