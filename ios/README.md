# HandPoseLiDAR — iPhone (ARKit LiDAR + Core ML) test app

Live RGB-D hand pose on an iPhone Pro (LiDAR) using the dual-stream model
exported to Core ML (`pretrain/ds_lidarsim.mlpackage`).

> **Status: scaffold.** These Swift sources cannot be compiled/verified on this
> Linux box — they are a faithful starting point to drop into Xcode (on a Mac or
> cloud Mac). The preprocessing is ported to match the Python pipeline
> (`infer_zed_dualstream.py` / `convert_dualstream.py`); verify against a known
> frame on the Mac before trusting absolute numbers.

## Model I/O contract (from convert_dualstream.py)
- **inputs**: `image` `[1,4,256,256]` (RGB 0..1 on ch0-2 + median-centred depth in
  [-1,1] on ch3), `depth_med` `[1,1]` (raw-depth median over the hand crop, metres).
- **outputs**: `keypoints` `[1,21,3]` (root-rel m), `root` `[1,3]`, `scale` `[1,1]`,
  `keypoints_abs` `[1,21,3]` (absolute camera-frame m → project & draw this).

## Files
| file | role |
|------|------|
| `Preprocess.swift` | hand bbox → 256 crop, build 4-ch input + depth_med (ports normalize_depth) |
| `HandPoseModel.swift` | Core ML wrapper (load .mlpackage, run, read outputs) |
| `ARViewController.swift` | ARKit sceneDepth(LiDAR) + Vision hand + project + draw |
| `Info.plist.snippet` | required usage keys |

## Build & deploy WITHOUT owning a Mac (cloud Mac + TestFlight)
1. **Apple Developer Program** ($99/yr) — needed for TestFlight.
2. Rent a **cloud Mac** (MacinCloud / AWS EC2 Mac / Scaleway) with Xcode, or use a
   CI macOS runner (Codemagic / Xcode Cloud / GitHub Actions).
3. In Xcode: **File ▸ New ▸ App** (SwiftUI or Storyboard, iOS, Swift). Set the
   deployment target to iOS 16+.
4. Add these `.swift` files to the target; merge `Info.plist.snippet` keys.
5. Drag `pretrain/ds_lidarsim.mlpackage` into the project (check "Copy items",
   add to target). Xcode auto-generates a `ds_lidarsim` class.
6. Set your Team (signing) + a bundle id. Connect to App Store Connect.
7. **Product ▸ Archive ▸ Distribute ▸ TestFlight**. (A cloud Mac can't USB to your
   phone, so install via the **TestFlight app** on the iPhone — over the air.)
8. Requires a **LiDAR** iPhone (Pro / Pro Max, 12 Pro and up).

## Validate FIRST without any Mac (recommended)
Before investing in the app, confirm the model is worth deploying:
1. Capture iPhone LiDAR RGB+depth with a free app (**Record3D**, **3D Scanner App**).
2. Export per-frame RGB (`.png`) + depth. Write them as paired
   `rgb_00000.png` + `depth_00000.png` (depth = uint16 **millimetres**, 0=invalid).
3. On this Linux box:
   ```
   python infer_zed_dualstream.py --ckpt mobrecon_ckpt/ds_lidarsim/best.pt \
       --source folder --path <your_capture_dir> --no-gui --save out/iphone_test
   ```
   This runs the SAME model on real iPhone-LiDAR depth — no Mac needed — and tells
   you whether ds_lidarsim actually transfers to the iPhone domain.
