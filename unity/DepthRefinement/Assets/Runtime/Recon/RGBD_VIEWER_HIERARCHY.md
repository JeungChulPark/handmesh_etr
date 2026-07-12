# RGB-D viewer — Unity Hierarchy (build this exactly)

Target: AR Foundation 6.3.5 + Unity 6. `(menu)` = created by a menu command (comes pre-wired).
`(add)` = you add it manually. `→` = an Inspector reference you drag.

```
Hierarchy
│
├── AR Session                         (menu: GameObject ▸ XR ▸ AR Session)
│     • ARSession
│
├── XR Origin                          (menu: GameObject ▸ XR ▸ XR Origin (AR))
│     • XROrigin            → Camera = Main Camera (auto)
│     │
│     └── Camera Offset                (auto child)
│           │
│           └── Main Camera            (auto child)
│                 • Camera             (auto)
│                 • AudioListener      (auto)
│                 • TrackedPoseDriver  (auto — drives camera pose)
│                 • ARCameraManager    (auto)
│                 • ARCameraBackground (auto)  ← draws the RGB feed fullscreen
│                 • AROcclusionManager (ADD)   ← LiDAR depth
│                       Environment Depth Mode = Best (or Medium)
│                       Environment Depth Temporal Smoothing = OFF
│
├── RGBDViewer                         (add: empty GameObject)
│     • DepthImageVisualizer (ADD, script)
│           Occlusion Manager → Main Camera   (the AROcclusionManager above)
│           Target            → DepthView      (the RawImage below)
│           Status Label      → DepthLabel     (optional Text below)
│           Min Depth = 0.2   Max Depth = 3.0
│           Rotate 90 CW = ✔
│
├── Canvas                             (menu: GameObject ▸ UI ▸ Canvas)
│     • Canvas            Render Mode = Screen Space - Overlay
│     • Canvas Scaler
│     • Graphic Raycaster
│     │
│     ├── DepthView                    (add child: GameObject ▸ UI ▸ Raw Image)
│     │     • RawImage                 (Texture left empty — the script sets it at runtime)
│     │     RectTransform: Anchor = top-right, e.g. Pos (-200,-280), Size (360, 480)
│     │
│     └── DepthLabel                   (optional: GameObject ▸ UI ▸ Text - Legacy)
│           • Text                     ("depth ..." — the script fills it)
│           RectTransform: just under DepthView; Color white, Font Size ~24
│
└── EventSystem                        (auto-created with the Canvas)
      • EventSystem
      • Standard/Input System UI Module
```

## Build order (5 steps)

1. **AR Session**: `GameObject ▸ XR ▸ AR Session`.
2. **XR Origin (AR)**: `GameObject ▸ XR ▸ XR Origin (AR)` → creates Camera Offset ▸ Main
   Camera with ARCameraManager + ARCameraBackground + TrackedPoseDriver pre-wired.
3. On **Main Camera**, `Add Component ▸ AR Occlusion Manager`; set Environment Depth Mode =
   Best, Temporal Smoothing off.
4. **Canvas + RawImage**: `GameObject ▸ UI ▸ Canvas`, then right-click it ▸ `UI ▸ Raw Image`
   (rename to `DepthView`); anchor it to a corner. (An EventSystem auto-appears.)
5. **RGBDViewer**: `GameObject ▸ Create Empty` (rename `RGBDViewer`), `Add Component ▸ Depth
   Image Visualizer`. Drag **Main Camera** into *Occlusion Manager* and **DepthView** into
   *Target*.

RGB needs no wiring — `ARCameraBackground` shows it automatically. The depth overlay appears
once you run on a LiDAR device.

## Project Settings to set once (not in the Hierarchy)

- **XR Plug-in Management ▸ iOS ▸ Apple ARKit** = ON.
- **Player ▸ iOS ▸ Camera Usage Description** = e.g. "Used for AR hand tracking".
- **Player ▸ iOS**: Graphics API = Metal, Architecture = ARM64, Min iOS = 14+.

## Later: add the hand model (optional, same rig)

```
├── RGBDViewer
│     • DepthImageVisualizer
│     • DualStreamHandProvider   (ADD later)
│           Camera Manager    → Main Camera
│           Occlusion Manager → Main Camera
│           Model Asset       → ds_anchor_gate.onnx  (import the .onnx as a ModelAsset)
```
```
