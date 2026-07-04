# Hybrid-B on the ZED: model, A/B result, live deployment

Working log for taking the **hybrid_B** lifter to live ZED inference, plus the
A-vs-B training comparison that motivated picking B. Date: 2026-07-04.

## 1. What "model B" is

`hybrid_B` is one of two hybrid lifters trained by `run_BA.sh` (both warm-started
from `rgbd_real_0613`, datasets `dexycb,ho3d,iphone`, 40 epochs). The model class
is `HybridLifter` in `train_hybrid.py`; the two variants differ only in `--mode`:

| Variant | mode | How the 3D joints are formed |
|---|---|---|
| **B** (`hybrid_B`) | `locked` | **X,Y from MediaPipe (u,v)** (2D locked to the detector) + **Z from the MobRecon backbone's per-joint relative depth**, anchored at the sensor wrist depth: `Zabs = Z0 + dz_backbone`. |
| A (`hybrid_A`) | `backbone` | backbone `p_cnn` is primary, a residual head predicts `Δp` from `[p_cnn, p_geom, mp_feat]`, plus a 2D-reprojection-to-MediaPipe loss (`--lambda_reproj 1e-6`). |

Backbone = `LargeModel_Extra_RGBD` (4-channel RGB-D) on a 256×256 hand crop.
Loss (both) = 3D L2 + bone-length + aux(`p_cnn` vs GT) + small uv-reg.

## 2. A vs B training result

Read from each `mobrecon_ckpt/<exp>/best.pt` and `logs/run_BA.log`:

| | Best epoch | **Combined MPJPE** | DexYCB | HO3D | iPhone |
|---|---|---|---|---|---|
| **B** (locked) | 21 / 40 | **27.01 mm** ✅ | 28.19 | 26.36 | 17.15 |
| A (backbone+reproj) | 1 / 40 | **1669.57 mm** ❌ diverged | 1958 | 1426 | 1055 |

**A diverged.** Its per-epoch trace blows up monotonically — the backbone output
`cnn` error goes 61 mm → thousands, peaking `comb=23379 mm` at ep16 — and `best.pt`
is stuck at epoch 1 (already broken). The residual head + tiny reprojection loss
never stabilised the backbone-primary path.

**B converged cleanly** (28.6 → 27.0 mm by ep21, flat thereafter). Constraining
2D to MediaPipe and only letting the backbone supply *relative depth* is far more
stable and accurate. (For reference, the pure Z-lifter baseline was 37.43 mm
combined; `hybrid_da` 20.54 mm and `hybrid_bb` 21.17 mm score better than B on the
held-out combined metric but were not the target of this deployment.)

## 3. Live ZED inference — `infer_zed_hybrid.py`

New script that runs **model B live on a Stereolabs ZED** (or `.svo` replay),
reusing existing pieces so nothing about the model path changed:

* frame source: `infer_rgbd_zed.ZEDSource` — LEFT BGR + metric depth (metres,
  NaN/Inf mapped to 0), LEFT-cam intrinsics.
* per-frame inference: same path as `infer_hybrid.py` — MediaPipe uv →
  per-joint depth sample (3×3 median) → `feat` + 256² RGB-D crop →
  `HybridLifter(mode="locked")` → root placed at the ZED wrist depth → reproject.

Overlay legend: **cyan rings = MediaPipe 2D**, **grey = backbone-only** (`p_cnn`),
**colour skeleton = hybrid_B**; HUD shows `wrist_z`, `z-span`, depth-coverage, FPS.

### Run it

```bash
DISPLAY=:1 python infer_zed_hybrid.py --source zed            # live GUI, NEURAL depth
DISPLAY=:1 python infer_zed_hybrid.py --source zed --zed_depth_mode PERFORMANCE
python infer_zed_hybrid.py --no-gui --save zed_hybrid_out/    # headless: PNGs + metrics.jsonl
python infer_zed_hybrid.py --source svo --svo rec.svo2        # replay a recording
python infer_zed_hybrid.py --source selftest                 # no hardware, wiring check
```

Defaults: `--ckpt mobrecon_ckpt/hybrid_B/best.pt`, `--zed_resolution HD720`,
`--zed_fps 30`, `--zed_depth_mode NEURAL`, `--zed_min_depth 0.3 --zed_max_depth 2.0`.
Quit the GUI with `q` / `ESC`.

### Verified on hardware (2026-07-04)

* Camera: ZED 2i (S/N 30670909), HD720@30, GPU RTX 5090 Laptop. LEFT K:
  `fx=fy=526.20, cx=626.31, cy=352.50`.
* Wiring: `selftest` loads `hybrid_B/best.pt` (epoch 21, 27.01 mm) OK.
* Live NEURAL run: 826 frames, hand detected in 668 (~81%), real-time, exited on
  window close.
* **PERFORMANCE-mode FPS (headless, 540 frames): avg 27.6, min 25.0, max 30.0** —
  effectively pinned to the 30-fps camera cap, i.e. real-time. NEURAL depth is
  more accurate but heavier; PERFORMANCE is the fast option.

## 4. Files

| Path | Role |
|---|---|
| `infer_zed_hybrid.py` | **new** — live ZED / SVO inference for hybrid_B (locked). |
| `train_hybrid.py` | `HybridLifter`, both A/B modes; training entrypoint. |
| `run_BA.sh` | trains B then A. |
| `infer_hybrid.py` | offline hybrid inference on `cap_*.json` captures. |
| `infer_rgbd_zed.py` | `ZEDSource` (camera front-end, reused). |
| `mobrecon_ckpt/hybrid_B/best.pt` | deployed checkpoint (ep21, 27.01 mm). |
| `logs/infer_zed_hybrid_B.log`, `logs/infer_zed_hybrid_B_perf.log` | run logs. |
