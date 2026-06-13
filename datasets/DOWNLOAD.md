# Dataset download & access status

Reality check: **most of these datasets are behind a registration form or license
agreement** and cannot be fetched without your credentials. Below is the access
path, approximate size, and the target root each loader expects. Local free disk
at last check: **~932 GB** — far less than the full set (~several TB), so download
selectively.

| dataset | access | approx size | loader root arg |
|---------|--------|-------------|-----------------|
| **HanCo** | ✅ already on disk | — | `Datasets/Hand Dataset/HanCo` |
| **FreiHAND** | ✅ already on disk | — | (RGB baseline) |
| **ContactPose** | 🟢 public script (no login) | ~140 GB (RGB-D) | `--contactpose_dir` |
| **NYU** | 🟢 direct link (no login) | ~92 GB | `--nyu_root` (train/) |
| **MSRA15** | 🟡 public mirror (Dropbox/OneDrive, link rot) | ~3 GB | `--msra_root` |
| **ICVL** | 🔴 registration (Imperial) | ~4 GB | `--icvl_root` |
| **DexYCB** | 🔴 license (NVIDIA), then direct tars | ~120 GB | `--dexycb_root` |
| **HO3D_v3** | 🔴 registration form | ~30 GB | `--ho3d_root` |
| **H2O-3D** | 🔴 registration form | ~25 GB | `--h2o3d_root` |
| **BigHand2.2M** | 🔴 registration (Imperial) | ~200+ GB | `--bighand_root` |
| **HANDS17** | 🔴 registration (challenge) | ~80 GB | `--hands17_root` |
| **FPHA** | 🔴 registration (Imperial) | ~120 GB | `--fpha_root` |
| **OakInk** | 🔴 Google form | ~100+ GB | `OAKINK_DIR` env |
| **MANO models** | 🔴 license (mano.is.tue.mpg.de) | tiny | `thirdparty/mano` (ContactPose/render) |

🟢 scriptable here · 🟡 best-effort mirror · 🔴 needs your account/license

## Scriptable downloads (no login)
Run `bash datasets/download_datasets.sh <name> <dest>`:
- `contactpose` → clones the toolkit + runs their `download_data.py` (RGB-D + grasps)
- `nyu` → direct zip from the NYU mirror

For the 🔴 sets: register on the dataset site, accept the license, then point the
loader's `--*_root` at the extracted folder. Layouts the loaders expect are
documented at the top of each loader file (`datasets/{dexycb,ho3d,tof_depth,...}.py`).

## Recommended first pull (fits comfortably in 932 GB)
1. **MSRA15** (~3 GB) — ToF warm-up, smallest real-depth set, validates the ToF path.
2. **ContactPose** (subset via `--p_nums`) — public, real Kinect depth + MANO.
3. **DexYCB** (~120 GB) — after accepting the NVIDIA license; best real RGB-D source.
Skip BigHand/FPHA/HANDS17 unless you specifically need their scale (200 GB+ each).
