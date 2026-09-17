#!/usr/bin/env bash
# Session-level (leave-one-session-out) version of the joint-vs-finetune ablation.
# Train on 4 capture sessions, hold out rgbd_captures_04 ENTIRELY as eval, so there
# is no temporally-adjacent-frame leakage (unlike the idx%5 split). Everything else
# matches scripts/run/run_joint_ablation.sh: warm-start rgbd_real_0613, lr 1e-5, 25 ep, best.pt
# selected on the held-out session, all re-scored by scripts/eval/eval_iphone.py.
set -e
cd /home/jucpark/DeepLearning/handmesh_etr

PRETRAIN="mobrecon_ckpt/rgbd_real_0613/best.pt"
IPH_TRAIN="rgbd_captures,rgbd_captures_01,rgbd_captures_02,rgbd_captures_03"
IPH_EVAL="rgbd_captures_04"
DEXYCB="/home/jucpark/DeepLearning/Datasets/Hand Dataset/DexYCB_full/data"
HO3D="/home/jucpark/DeepLearning/Datasets/Hand Dataset/HO3D_full"
COMMON="--pretrain $PRETRAIN --iphone_root $IPH_TRAIN --iphone_split all \
        --iphone_eval_root $IPH_EVAL --fixed_eval_iphone \
        --lr 1e-5 --epoch 25 --batch 32 --w2d 0.1"

echo "======== [1/3] finetune-S (iphone-train only) ========"
python scripts/train/train_mobrecon_rgbd.py --exp ft_fair_s --datasets iphone $COMMON

echo "======== [2/3] replay-S (iphone x8 + DexYCB 4k) ========"
python scripts/train/train_mobrecon_rgbd.py --exp replay_s --datasets iphone,dexycb \
    --iphone_repeat 8 --source_limit 4000 --dexycb_root "$DEXYCB" $COMMON

echo "======== [3/3] joint-S (iphone x8 + DexYCB 15k + HO3D 15k) ========"
python scripts/train/train_mobrecon_rgbd.py --exp joint_s --datasets iphone,dexycb,ho3d \
    --iphone_repeat 8 --source_limit 15000 \
    --dexycb_root "$DEXYCB" --ho3d_root "$HO3D" $COMMON

echo "======== scoring all best.pt on the held-out SESSION (rgbd_captures_04) ========"
for exp in ft_fair_s replay_s joint_s; do
    python scripts/eval/eval_iphone.py --ckpt "mobrecon_ckpt/$exp/best.pt" \
        --iphone_root "$IPH_EVAL" --split all 2>&1 | grep "eval_iphone]"
done
python scripts/eval/eval_iphone.py --ckpt mobrecon_ckpt/ef_iphone_v2/best.pt \
    --iphone_root "$IPH_EVAL" --split all 2>&1 | grep "eval_iphone]"
python scripts/eval/eval_iphone.py --ckpt "$PRETRAIN" \
    --iphone_root "$IPH_EVAL" --split all 2>&1 | grep "eval_iphone]"
echo "======== DONE ========"
