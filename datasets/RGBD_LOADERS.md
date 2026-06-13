# RGB-D hand dataset loaders — master catalog

12 dataset loaders that all emit the **same** per-sample dict as
`datasets/hanco_ty.HanCo_ETRI_jitter`, so they drop into `train_mobrecon_rgbd.py`
(`--datasets <comma-list>`) and concatenate freely. Depth-only / partial-joint
sets add a `joint_valid` mask and train through a masked loss. Full guide +
contract details: [`rgbd_datasets.md`](rgbd_datasets.md).

## Output contract (every loader)
`image[4,256,256]` (RGB[0,1] + depth[-1,1]) · `keypoints3D[21,3]` root-aligned cam
metres · `keypoints2D[21,2]` (/256) · `root[3]` · `cam[3,3]` · (+`joint_valid[21]`
on ToF sets; the trainer's `WithJointValid` injects ones elsewhere).

## The 12 loaders

| # | dataset | `--datasets` | class · file | sensor / depth | joints | depth source | key gotchas baked in |
|---|---------|--------------|--------------|----------------|--------|--------------|----------------------|
| 1 | HanCo | `hanco` | `HanCo_ETRI_jitter` · hanco_ty.py | — (synthetic) | 21 ✓ | **render** (depth_synth) | multi-view mocap, no sensor depth; capsule-mesh render |
| 2 | DexYCB | `dexycb` | `DexYCB_RGBD` · dexycb.py | RealSense (stereo) | 21 ✓ | **real**, aligned | joints cam-metres; `seg==255` hand mask; left→right flip |
| 3 | HO3D | `ho3d` | `HO3D_RGBD` · ho3d.py | RealSense (stereo) | 21 ✓ | **real**, 2-ch PNG | OpenGL coords (coordChangeMat); train-split; z-band mask |
| 4 | H2O-3D | `h2o3d` | `H2O3D_RGBD` · ho3d.py | RealSense (stereo) | 21 ✓ ×2 | **real**, 2-ch PNG | two-hand; `right/leftHandJoints3D`; reuses HO3D `_finalize` |
| 5 | MSRA15 | `msra` | `MSRA_RGBD` · tof_depth.py | **ToF** | 21 ✓ | **real**, .bin | y-up→flip; MSRA→MANO perm; bbox-cropped depth; pseudo-RGB |
| 6 | ICVL | `icvl` | `ICVL_RGBD` · tof_depth.py | **ToF** | 16 → masked | **real**, mm PNG | 3/finger → 16/21 MANO slots (5 PIP absent); pseudo-RGB |
| 7 | NYU | `nyu` | `NYU_RGBD` · tof_depth.py | structured-light | 6 → masked | **real**, G·256+B | fingertips+wrist only; needs scipy; pseudo-RGB |
| 8 | BigHand2.2M | `bighand` | `BigHand_RGBD` · tof_depth.py | SR300 | 21 ✓ | **real**, mm PNG | BIGHAND→MANO perm; ~2.2M lines → use `limit`; pseudo-RGB |
| 9 | HANDS17 | `hands17` | `HANDS17_RGBD` · tof_depth.py | SR300 | 21 ✓ | **real**, mm PNG | same format as BigHand (`training/`); pseudo-RGB |
| 10 | FPHA | `fpha` | `FPHA_RGBD` · tof_depth.py | SR300 (egocentric) | 21 ✓ | **real**, mm PNG | world→cam `FPHA_CAM_EXTR`; reorder==BIGHAND; colour/depth offset → verify viz |
| 11 | ContactPose | `contactpose` | `ContactPose_RGBD` · oakink_contactpose.py | Kinect v2 | 21 ✓ | **real**, mm PNG | toolkit-wrapped; joints obj→cam via `object_pose`; z-band mask |
| 12 | OakInk | `oakink` | `OakInk_RGBD` · oakink_contactpose.py | RealSense | 21 ✓ | **render** (no oikit depth) | toolkit-wrapped; oikit exposes no depth → synthetic; RGB+MANO |

Legend: **joints** ✓ = full 21 (all valid); "→ masked" = partial, `joint_valid` set.
**real** = genuine sensor depth; **render** = synthesised from geometry (depth_synth).

## Sensor families (for the Femto-Bolt ToF domain gap)
- **ToF** (closest to Femto Bolt): MSRA, ICVL
- **SR300 coded-light**: BigHand, HANDS17, FPHA
- **Stereo RealSense**: DexYCB, HO3D, H2O-3D, OakInk(no depth)
- **Kinect v2**: ContactPose
- **Synthetic**: HanCo, OakInk
→ Recommended: warm the depth branch on **MSRA (ToF)**, then train on real RGB-D
(DexYCB/HO3D/ContactPose). Stereo-trained depth won't transfer to ToF without
`add_sensor_noise` tuning or a ToF pretrain.

## Usage
```bash
# single
python train_mobrecon_rgbd.py --datasets dexycb --dexycb_root /data/DexYCB
# concat real-depth + ToF warm-up, masked loss handles partial-joint sets
python train_mobrecon_rgbd.py --exp rgbd_all \
    --datasets hanco,dexycb,ho3d,h2o3d,msra,fpha \
    --dexycb_root /data/DexYCB --ho3d_root /data/HO3D_v3 \
    --h2o3d_root /data/H2O3D --msra_root /data/MSRA --fpha_root /data/FPHA \
    --pretrain pretrain/100.pt
```
Per-loader smoke test (verify joint/depth alignment on real data):
```bash
python -m datasets.dexycb       --root <DexYCB> --check 8
python -m datasets.ho3d         --root <HO3D|H2O3D> [--dataset h2o3d] --check 8
python -m datasets.tof_depth    --dataset <msra|icvl|nyu|bighand|hands17|fpha> --root <p> --check 8
python -m datasets.oakink_contactpose --dataset <contactpose|oakink> [--data_dir <p>] --check 8
```

## Download / access status — see [`DOWNLOAD.md`](DOWNLOAD.md)
Most require registration or a license agreement (cannot be auto-fetched). Only
ContactPose (public script) and NYU (direct link) are scriptable without login.
