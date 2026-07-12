#!/usr/bin/env bash
# Option B: MediaPipe 2D heatmaps as extra input channels (RGB+D+21 = 25ch).
# Isolated ablation vs finetune-S (25.54mm): SAME data (iPhone 4 sessions),
# warm-start, lr, epochs, and held-out-session eval (rgbd_captures_04) -- the ONLY
# difference is the 21 heatmap channels. Answers: does feeding MediaPipe 2D help?
set -e
cd /home/jucpark/DeepLearning/handmesh_etr

PRETRAIN="mobrecon_ckpt/rgbd_real_0613/best.pt"
IPH_TRAIN="rgbd_captures,rgbd_captures_01,rgbd_captures_02,rgbd_captures_03"
IPH_EVAL="rgbd_captures_04"

echo "======== B: finetune + MediaPipe heatmap (25ch) ========"
python train_mobrecon_rgbd.py --exp ft_mp_heatmap_s --datasets iphone \
    --pretrain "$PRETRAIN" --iphone_root "$IPH_TRAIN" --iphone_split all \
    --iphone_eval_root "$IPH_EVAL" --fixed_eval_iphone --mp_heatmap \
    --lr 1e-5 --epoch 25 --batch 32 --w2d 0.1

echo "======== scoring on held-out session (rgbd_captures_04) ========"
python eval_iphone.py --ckpt mobrecon_ckpt/ft_mp_heatmap_s/best.pt \
    --iphone_root "$IPH_EVAL" --split all --mp_heatmap 2>&1 | grep "eval_iphone]"
echo "======== DONE ========"
