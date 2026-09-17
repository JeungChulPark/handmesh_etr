# Dual-Stream Cross-Dataset Generalization Eval

**Date:** 2026-06-25 · **Script:** `scripts/eval/eval_pa_mpjpe.py` · **Frames:** 2016 per cell
(seed-0, 0.95/0.05 split of each set's train portion — comparable across models,
not an official test benchmark). HanCo uses `--depth_source render`.

All numbers in **mm** (lower better) except PCK-AUC(PA) (0–50mm, higher better).

---

## abs MPJPE — the deploy metric (absolute camera-space joints)

| Model \ Dataset | HanCo (synth) | DexYCB (real) | HO3D (real) |
|-----------------|--------------:|--------------:|------------:|
| `ds_hybrid_ft2` (in-domain best) | **19.88** | 238.38 | 125.87 |
| `ds_lidarsim` (LiDAR-sim aug)    | 28.28 | **80.59** | **51.22** |
| `ds_realdepth2` (real-depth ft)  | 27.71 | 86.30 | 52.33 |

## root error — global translation (the TranslationHead's job)

| Model \ Dataset | HanCo | DexYCB | HO3D |
|-----------------|------:|-------:|-----:|
| `ds_hybrid_ft2` | **18.63** | 236.86 | 133.45 |
| `ds_lidarsim`   | 26.48 | **78.38** | **48.87** |
| `ds_realdepth2` | 25.71 | 84.40 | 49.42 |

## root-rel MPJPE — articulation (root-aligned)

| Model \ Dataset | HanCo | DexYCB | HO3D |
|-----------------|------:|-------:|-----:|
| `ds_hybrid_ft2` | **14.42** | 63.93 | 84.42 |
| `ds_lidarsim`   | 15.56 | **27.25** | **29.62** |
| `ds_realdepth2` | 16.00 | 29.05 | 33.93 |

## PA-MPJPE — pose only (similarity-aligned, removes global error)

| Model \ Dataset | HanCo | DexYCB | HO3D |
|-----------------|------:|-------:|-----:|
| `ds_hybrid_ft2` | **7.99** | 18.73 | 43.23 |
| `ds_lidarsim`   | 9.52 | 10.56 | 16.24 |
| `ds_realdepth2` | 9.84 | 11.31 | 19.95 |

## PCK-AUC (PA), 0–50mm — higher better

| Model \ Dataset | HanCo | DexYCB | HO3D |
|-----------------|------:|-------:|-----:|
| `ds_hybrid_ft2` | **0.841** | 0.633 | 0.235 |
| `ds_lidarsim`   | 0.811 | **0.789** | **0.677** |
| `ds_realdepth2` | 0.804 | 0.774 | 0.606 |

---

## Findings

1. **The in-domain "best" model is the worst deployable model.** `ds_hybrid_ft2`
   wins every HanCo metric but collapses on real depth (abs 238 / 126 mm on
   DexYCB / HO3D). It memorised HanCo's synthetic-render depth statistics.

2. **LiDAR-sim augmentation is the single biggest generalization lever.**
   `ds_lidarsim` is 3× better than `ds_hybrid_ft2` on real depth (abs 80/51 vs
   238/126) for only ~8 mm of HanCo regression. It is the current best deployable
   checkpoint and the right base for the iPhone-LiDAR target.

3. **Real-depth fine-tune did NOT beat LiDAR-sim** — `ds_realdepth2` is uniformly
   a touch worse than `ds_lidarsim` on every set, despite DexYCB/HO3D being in its
   own training mix. Consistent with the `ds_realdepth` divergence (170→477 mm):
   multi-dataset real-depth training is unstable, not additive.

4. **Pose generalizes; absolute translation does not.** PA-MPJPE stays 9–20 mm
   cross-dataset for lidarsim/realdepth, but root error is 78–237 mm everywhere
   off-HanCo. The broken piece is squarely the **absolute root / `depth_med`
   mapping**, not articulation — exactly the `depth_med` per-dataset distribution
   problem. Fixing root normalization should recover most of the abs gap.

---

## Geometric anchor + robust root loss — RESULT (2026-06-26)

`ds_anchor_lidar`: dual-stream with the depth-only geometric root anchor
(`geometric_root_anchor`, root = anchor + residual) + Huber root loss with
occlusion-outlier masking (`|anchor_z − gt_z| > 80 mm` dropped). Warm-start
`pretrain/100.pt`, hanco+dexycb+ho3d, LiDAR-sim, grad-clip 1.0, cosine lr 5e-5,
10 epochs (~8.5 h). Same 2016-frame eval protocol. Epoch-10 `best.pt`:

| Model \ Dataset | HanCo abs | HanCo root | DexYCB abs | DexYCB root | HO3D abs | HO3D root |
|-----------------|----------:|-----------:|-----------:|------------:|---------:|----------:|
| `ds_lidarsim` (prev best) | 28.28 | 26.48 | 80.59 | 78.38 | 51.22 | 48.87 |
| **`ds_anchor_lidar`** | **18.09** | **13.25** | **78.69** | **73.97** | **31.19** | **29.16** |

PA-MPJPE / root-rel preserved or improved everywhere (HanCo PA 9.52=9.52,
root-rel 15.5≈15.6; HO3D PA 16.2→13.7, root-rel 29.6→23.9; DexYCB root-rel
27.3→25.6). Training converged monotonically with NO divergence (combined-val
abs 50.7→41.4), unlike the free-regression `ds_realdepth` (170→477 mm).

**Conclusion:** the anchor beats the previous best on every dataset and metric,
and HanCo abs 18.09 even beats the in-domain champion `ds_hybrid_ft2` (19.91)
while generalizing (that model was 238/126 on DexYCB/HO3D). Articulation is
preserved; only absolute position improved — as designed. Remaining gap: DexYCB
abs stays ~79 mm because at INFERENCE the ~5–10% occlusion frames feed a wrong
anchor that the model trusts (robust loss only fixes *training*). Next:
**anchor-confidence gating** so the head falls back to the RGB prior when the
anchor is unreliable. See [[dexycb-depth-anchor-outliers]].

---

## Confidence-gated anchor — RESULT (2026-06-27)

`ds_anchor_gate`: the gating fine-tune predicted as the next step above. Warm-start
`ds_anchor_lidar/best.pt`, hanco+dexycb+ho3d, LiDAR-sim, grad-clip 1.0, cosine
lr 2e-5, batch 32, 20 epochs. Best = **epoch 19** (epoch 20 regressed, 23.70→23.76
combined-val abs → converged/plateaued). `geo_anchor=True`.

**Eval note:** run on the *full* val split (HanCo 20278 / DexYCB 14584 / HO3D 4524
frames), not the 2016-frame cap of the tables above. To keep it apples-to-apples,
`ds_lidarsim` was re-evaluated on the identical full-val split — its numbers match
the 2016-frame ones within <1 mm (HanCo 28.28→28.58, DexYCB 80.59→79.87,
HO3D 51.22→51.07), confirming the two protocols are equivalent. Comparison below
uses the full-val numbers for both.

| Metric (mm ↓, PCK ↑) | Model | HanCo | DexYCB | HO3D |
|----------------------|-------|------:|-------:|-----:|
| **abs MPJPE** | `ds_lidarsim` | 28.58 | 79.87 | 51.07 |
|               | **`ds_anchor_gate`** | **15.11** | **38.36** | **23.29** |
| **root error** | `ds_lidarsim` | 26.76 | 78.19 | 48.75 |
|                | **`ds_anchor_gate`** | **9.32** | **32.94** | **18.61** |
| **root-rel MPJPE** | `ds_lidarsim` | 15.62 | 26.93 | 29.56 |
|                    | **`ds_anchor_gate`** | **14.51** | **22.64** | **20.23** |
| **PA-MPJPE** | `ds_lidarsim` | 9.58 | 10.40 | 16.20 |
|              | **`ds_anchor_gate`** | **9.12** | **8.98** | **12.01** |
| **PCK-AUC (PA)** | `ds_lidarsim` | 0.809 | 0.792 | 0.678 |
|                  | **`ds_anchor_gate`** | **0.818** | **0.821** | **0.760** |

**Conclusion:** gating wins on **every dataset × every metric**. abs MPJPE roughly
halves on real depth (DexYCB 80→38, HO3D 51→23; −52%/−54%), driven mostly by root
error (DexYCB 78→33, HO3D 49→19). Crucially this **closes the DexYCB gap that
`ds_anchor_lidar` left open** — that model was still stuck at DexYCB abs 78.69 (≈
`ds_lidarsim`) because occlusion-frame anchors fooled it at inference; gating lets
the head fall back to the RGB prior, dropping DexYCB abs to 38.36. PA-MPJPE
improves too, so articulation was not sacrificed. HanCo abs 15.11 also beats the
in-domain champion `ds_hybrid_ft2` (19.88) while generalizing.

**`ds_anchor_gate/best.pt` is now the single best deployable checkpoint** — it
should replace `ds_lidarsim` in the export/deploy path. See
[[geometric-anchor-result]], [[zed-deployment-sensor]].
