# iPhone Hand-Pose: Joint-vs-Finetune Ablation & MediaPipe-2D Z-Lifter

Work log for the 2026-07-02/03 session. Two related threads toward accurate
**metric 3D hand pose on iPhone LiDAR**:

1. **Joint-vs-fine-tune ablation** for the early-fusion RGB-D model.
2. **MediaPipe-2D + depth "Z-only lifter"** — trust MediaPipe's 2D, solve only depth.

All MPJPE numbers are root-relative, in mm, on **held-out splits** (leakage-free
where noted). The single most important result: **the lifter's accuracy is
bottlenecked by per-joint depth-sampling quality, not by architecture or data
quantity.**

---

## 1. Joint training vs fine-tuning (early-fusion iPhone model)

**Question:** the early-fusion model `rgbd_real_0613` fine-tuned on iPhone captures
(`ef_iphone_v2`). Does mixing iPhone data *into* the source training (joint) beat
sequential fine-tuning?

**Data imbalance (the crux):** source mix (HanCo+DexYCB+HO3D) = 787,689 frames vs
iPhone = 961. Naive concat → iPhone is 0.12% of a batch. So `build_dataset` got
`--iphone_repeat` (oversample), `--source_limit` (strided cap), `--iphone_split`
(deterministic holdout), `--fixed_eval_iphone` (select best.pt on the iPhone-eval
holdout, not a random 5% of the source-dominated concat).

### 1a. idx%5 holdout (has temporal leakage)

| model | data | MPJPE |
|---|---|---|
| rgbd_real_0613 | source only (zero-shot) | 48.10 |
| ef_iphone_v2 (orig, unlogged cfg) | iPhone | 49.21 |
| finetune | iPhone-train only | 22.26 |
| replay | iPhone×8 + DexYCB 4k | **21.51** |
| joint | iPhone×8 + DexYCB15k+HO3D15k | 26.62 |

→ replay ≳ finetune > oversampled-joint. But idx%5 splits **adjacent video frames**
(near-duplicates) → 22mm is optimistic.

### 1b. Session-level holdout (leakage-free: hold out `rgbd_captures_04`)

| model | MPJPE |
|---|---|
| rgbd_real_0613 (no finetune) | 44.85 |
| ef_iphone_v2 (orig) | 44.42 |
| **finetune-S** (fair) | **25.54** |

→ Fine-tuning genuinely helps even on an unseen session (44.85 → 25.54). Session
split is stricter than idx%5 (22.3 → 25.5) — confirms the leakage. replay-S/joint-S
were interrupted (user stopped training) so the joint-vs-replay comparison on the
session split is incomplete.

**Files:** `scripts/train/train_mobrecon_rgbd.py` (new args above), `datasets/iphone_captures.py`
(`split` param), `scripts/eval/eval_iphone.py`, `scripts/run/run_joint_ablation.sh`, `scripts/run/run_joint_ablation_session.sh`.

---

## 2. MediaPipe-2D as the front-end

**Motivation:** the base RGB MobRecon (`pretrain/100.pt`) under-articulates fingers
on iPhone RGB (median shape-err ~23%, p90 66% after Procrustes alignment). MediaPipe
Hands, trained on huge in-the-wild data, localises the 21 2D joints far better
(measured **7.8px** vs GT-2D on DexYCB). Idea: offload 2D to MediaPipe, let the model
solve depth.

Two axes (kept distinct throughout):
- **Where the 2D comes from:** GT-projected (+jitter) vs **real MediaPipe** (cached).
- **How it's encoded:** heatmap image (→ CNN) vs raw coordinates (→ MLP/lifter).

### 2a. B — MediaPipe heatmap as CNN input (`--mp_heatmap`)

RGB(3)+Depth(1)+21 Gaussian heatmap channels = 25ch into the early-fusion CNN;
warm-start grafts the 4ch and zero-inits the 21 heatmap channels. Train-time 2D
jitter + joint dropout simulate MediaPipe noise.

Result (iPhone session eval): **25.46mm ≈ finetune-S 25.54mm — no gain.** Because
the iPhone GT is itself MediaPipe-derived (circular) and the error is Z-dominated
(2D doesn't help Z).

### 2b. C — 2D + depth → 3D lifter (`scripts/train/train_lifter.py`)

Discard pixels; input per joint = (u,v) + depth sampled at (u,v). 0.6M MLP → 21×3.
Result (iPhone pseudo-GT session eval): **23.42mm**, matching the full CNN at 1/100
the size — but still pseudo-GT circular.

---

## 3. Z-only lifter + bone-length prior, on REAL GT (`scripts/train/train_zlifter.py`)

The definitive, circularity-free version of the user's idea: **keep MediaPipe 2D
exactly, predict only per-joint depth.**

- **Architecture:** head predicts a per-joint **depth correction ΔZ** over the
  sampled sensor depth: `Z = z_sampled + ΔZ`. 3D is reconstructed geometrically —
  `X=(u−cx)/fx·Z, Y=(v−cy)/fy·Z` — so the 2D reprojects onto MediaPipe exactly.
  Works in the **original camera frame** (no crop). 0.58M MLP.
- **Loss:** 3D L2 (root-relative) + λ·**bone-length** consistency (vs GT bone lengths).
- **Real GT:** DexYCB (MANO/mocap) + HO3D — NOT MediaPipe-derived → no circularity.
- **Splits:** DexYCB by subject (hold out subject-09/10), HO3D by sequence (hold out
  11 seqs). Leakage-free.
- **Baseline printed every epoch:** geometric no-learning reconstruction (ΔZ=0).

### Results (real GT, held-out)

| version | DexYCB | HO3D | Combined | geo-baseline |
|---|---|---|---|---|
| unmasked (DexYCB only) | 52.58 | — | — | 118.22 |
| **hand-masked** (DexYCB only) | 46.68 | — | — | 105.40 |
| **masked + HO3D combined** | **45.2** | **30.1** | **37.43** | dex 105 / ho3d 42 |

### The bottleneck (diagnostic)

Sampled-depth vs GT joint depth: **mean 128mm** (median 22mm) with naive sampling —
the mean tail comes from silhouette/object/background pixels the window catches
(DexYCB is hand-object; MediaPipe's 7px error at silhouettes → big depth jumps).

- **Hand-masking** (DexYCB `seg==255`, or HO3D z-band) before sampling → mean 82mm →
  DexYCB 52.6→46.7mm (−11%).
- **HO3D depth is intrinsically cleaner** (geo 42 vs DexYCB 105) → lifts to 30.1mm.

→ The architecture works; **accuracy scales with depth-sampling cleanliness.** Train
≈ test (40 vs 45mm on DexYCB) → the model is signal-limited, not data-limited.

### HO3D integration gotcha (fixed)

HO3D `handJoints3D` groups fingertips at the end — **not** MediaPipe order. Correct
reorder (recovered by MediaPipe-vs-GT nearest-neighbour, fixes 81.9px→16.9px):

```
HO3D_TO_MANO = [0,13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20]
```

The stock `datasets/ho3d.py` uses identity `JOINT_PERM` — a **latent order bug** that
is only self-consistent within HO3D (would scramble a model shared with DexYCB).

---

## 4. Deploy inference (`scripts/infer/infer_zlifter.py`)

Per frame (no crop, full image):

```
RGB+LiDAR+K → MediaPipe(RGB)=21 (u,v)   [fallback if no detection]
            → sample LiDAR depth at each (u,v) → z_sampled, valid, z_ref
            → feat [21,4]=(u_norm,v_norm,z_use−z_ref,valid)
            → ZLifter → ΔZ[21]
            → Z=z_use+ΔZ,  X=(u−cx)/fx·Z, Y=(v−cy)/fy·Z  → metric 3D (camera frame)
```

Overlay draws the skeleton at MediaPipe uv with joints colour-coded by depth (near
red → far blue). On iPhone captures the best model gives physically-sane metric
values (root_z ~58cm, depth-span ~16cm, anatomically-correct depth ordering).
iPhone has no real 3D GT → deploy assessment is qualitative only.

```bash
python scripts/infer/infer_zlifter.py --dir rgbd_captures \
    --ckpt mobrecon_ckpt/zlifter_dex_ho3d/best.pt --tag zlift_v2
```

---

## 5. Reproduce

```bash
DEXYCB="/home/jucpark/DeepLearning/Datasets/Hand Dataset/DexYCB_full/data"
HO3D="/home/jucpark/DeepLearning/Datasets/Hand Dataset/HO3D_full"

# 1. cache real MediaPipe 2D (strided subset)
python scripts/capture/cache_mediapipe.py --dataset dexycb --root "$DEXYCB" --stride 8 --out cache/mp_dexycb.npz
python scripts/capture/cache_mediapipe.py --dataset ho3d   --root "$HO3D"   --stride 4 --out cache/mp_ho3d.npz

# 2. train the Z-only lifter (real GT, hand-masked, bone prior)
python scripts/train/train_zlifter.py --exp zlifter_dex_ho3d --datasets dexycb,ho3d --epoch 60

# 3. deploy inference on iPhone captures
python scripts/infer/infer_zlifter.py --dir rgbd_captures --ckpt mobrecon_ckpt/zlifter_dex_ho3d/best.pt
```

---

## 6. Artifacts

**Scripts** — `scripts/capture/cache_mediapipe.py`, `scripts/train/train_zlifter.py`, `scripts/infer/infer_zlifter.py`,
`scripts/train/train_lifter.py` (C), `scripts/infer/infer_rgb_base.py` (base-RGB shape check), `scripts/eval/eval_iphone.py`;
edits to `scripts/train/train_mobrecon_rgbd.py`, `datasets/iphone_captures.py`,
`scripts/infer/infer_rgbd_captures.py`.

**Checkpoints** — `mobrecon_ckpt/zlifter_dex_ho3d/best.pt` (**best deploy: 37.4mm
combined real-GT**), `zlifter_masked/`, `zlifter_dexycb/`, `lifter_s/` (C),
`ft_mp_heatmap_s/` (B), `ft_fair_s/` (session finetune), `ef_iphone_v2/` (early fusion).

**Caches** — `cache/mp_dexycb.npz` (62.6%/22820 detected), `cache/mp_ho3d.npz`
(74.9%/16938 detected).

**Videos** — `rgbd_captures_zlift_v2_pred.mp4` (combined lifter),
`rgbd_captures_base_vs_v2.mp4` (base-RGB vs early-fusion, MediaPipe rings),
`rgbd_captures_v2_pred.mp4`, `rgbd_captures_rgbbase_pred.mp4`.

---

## 7. Key findings

1. **Fine-tuning helps on unseen iPhone sessions** (44.9 → 25.5mm), but eval splits
   must be session-level — idx%5 leaks adjacent video frames.
2. **MediaPipe 2D ≫ MobRecon 2D** (7.8px vs ~23% shape-err). Feeding it as a
   *heatmap into the CNN* gave nothing (Z-dominated + circular GT); feeding it as
   *coordinates into a lifter* that solves Z is the right formulation.
3. **Z-only + bone-prior on real GT: 45mm (DexYCB) / 30mm (HO3D) / 37mm combined**,
   with a 0.58M model — circularity-free. The iPhone pseudo-GT 23mm was optimistic.
4. **Depth-sampling quality is the bottleneck** (128mm naive → 82mm masked; HO3D
   clean → 30mm). Not architecture, not data quantity (train ≈ test).

## 8. Open questions / next steps

- **Robust nearest-hand-surface depth sampling** — attack the 128mm tail directly
  (highest ROI, > full-data scaling).
- **Full-data scaling** — likely only 1–3mm (video near-duplicates; signal-limited).
  Cheap probe: stride 8→2 and see if it moves at all.
- **iPhone fine-tuning / deploy validation** — no real iPhone 3D GT yet; consider a
  small metrology-grade capture for a real number.
- Model capacity (0.58M → ~2M) only worth it alongside more data.
