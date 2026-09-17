#!/usr/bin/env bash
# Joint-training vs fine-tuning ablation on the iPhone RGB-D captures.
# All three warm-start from the same rgbd_real_0613 model, lr 1e-5, 25 epochs,
# and select best.pt on the FROZEN iPhone-eval holdout (idx%5==0, never trained on).
# Every best.pt is then re-scored by scripts/eval/eval_iphone.py so all numbers share one metric.
set -e
cd /home/jucpark/DeepLearning/handmesh_etr

PRETRAIN="mobrecon_ckpt/rgbd_real_0613/best.pt"
IPH="rgbd_captures,rgbd_captures_01,rgbd_captures_02,rgbd_captures_03,rgbd_captures_04"
DEXYCB="/home/jucpark/DeepLearning/Datasets/Hand Dataset/DexYCB_full/data"
HO3D="/home/jucpark/DeepLearning/Datasets/Hand Dataset/HO3D_full"
COMMON="--pretrain $PRETRAIN --iphone_root $IPH --iphone_split train \
        --fixed_eval_iphone --lr 1e-5 --epoch 25 --batch 32 --w2d 0.1"

echo "======== [1/3] finetune (iphone-train only) ========"
python scripts/train/train_mobrecon_rgbd.py --exp ft_fair --datasets iphone $COMMON

echo "======== [2/3] replay (iphone x8 + DexYCB 4k) ========"
python scripts/train/train_mobrecon_rgbd.py --exp replay_iphone --datasets iphone,dexycb \
    --iphone_repeat 8 --source_limit 4000 --dexycb_root "$DEXYCB" $COMMON

echo "======== [3/3] joint (iphone x8 + DexYCB 15k + HO3D 15k) ========"
python scripts/train/train_mobrecon_rgbd.py --exp joint_iphone --datasets iphone,dexycb,ho3d \
    --iphone_repeat 8 --source_limit 15000 \
    --dexycb_root "$DEXYCB" --ho3d_root "$HO3D" $COMMON

echo "======== scoring all best.pt on the frozen iPhone-eval set ========"
for exp in ft_fair replay_iphone joint_iphone; do
    python scripts/eval/eval_iphone.py --ckpt "mobrecon_ckpt/$exp/best.pt" 2>&1 | grep "eval_iphone]"
done
python scripts/eval/eval_iphone.py --ckpt mobrecon_ckpt/ef_iphone_v2/best.pt 2>&1 | grep "eval_iphone]"
python scripts/eval/eval_iphone.py --ckpt "$PRETRAIN" 2>&1 | grep "eval_iphone]"
echo "======== DONE ========"
