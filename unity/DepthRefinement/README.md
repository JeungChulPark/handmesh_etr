# Depth-Aware Hand Pose Refinement (Unity AR Foundation)

Refines RGB **Recon (MobRecon)** 3D hand joints using **device depth** — iOS LiDAR
`sceneDepth` (via AR Foundation `environmentDepth`) or a metric monocular depth model —
to fix Z-axis / fingertip error. The Recon model is consumed as a black box (never
retrained); refinement is pure post-processing behind swappable provider interfaces.

Implements the pipeline from `spec-depth-refinement-unity.md` (PRD: *Depth-Aware Mobile
Hand Pose Estimation*). Mobile-optimized (GC-free steady state), 30 FPS target.

## Folder structure

```
unity/DepthRefinement/
├── package.json                       # UPM manifest + dependencies
├── README.md
├── Runtime/
│   ├── DepthRefinement.Runtime.asmdef
│   ├── Core/
│   │   ├── HandJoint.cs               # HandJointId(21), bone edges, reference lengths, HandJointData
│   │   ├── HandPose.cs                # per-frame container (joints, handedness, timestamp, frame size)
│   │   ├── IHandJointProvider.cs      # RGB joint source contract
│   │   └── IDepthProvider.cs          # metric depth source contract (+ IDepthFrameUpdater)
│   ├── Depth/
│   │   └── ARFoundationDepthProvider.cs   # Step 1-3: environmentDepth/sceneDepth + confidence + intrinsics
│   ├── Refinement/
│   │   ├── JointDepthSampler.cs       # Step 4: robust neighborhood sampling + rejection
│   │   ├── JointReconstructor.cs      # Step 5: (u,v,depth)+intrinsics → metric (x,y,z)
│   │   ├── ConfidenceFusion.cs        # Step 6: bone-scale + shift-only RANSAC root-depth + fusion
│   │   ├── BoneLengthConstraint.cs    # Step 7: enforce bone lengths
│   │   ├── OneEuroFilter.cs           # Step 8: per-joint, Z-focused temporal smoothing
│   │   └── DepthRefinementPipeline.cs # orchestrator + FR-8 accept/reject gate
│   ├── Recon/
│   │   └── SentisHandJointProvider.cs # Step 9: Recon ONNX inference (Unity Sentis)
│   ├── Mock/
│   │   └── MockHandJointProvider.cs   # synthetic joints for tests/editor
│   └── DepthRefinementManager.cs      # Step 1: entry point, wiring, Inspector config, OnRefined event
├── Samples~/
│   └── DebugVisualization/
│       └── HandJointVisualizer.cs     # Step 10: gizmo skeleton, RGB-vs-refined, wasRefined coloring
└── Tests/
    ├── DepthRefinement.Tests.asmdef
    └── Editor/
        ├── TestSupport.cs             # FakeDepthProvider
        ├── JointReconstructorTests.cs
        ├── OneEuroFilterTests.cs
        ├── BoneLengthConstraintTests.cs
        ├── JointDepthSamplerTests.cs
        └── PipelineGateTests.cs       # fusion alignment + occlusion gate + RGB-only fallback
```

## Package dependencies

Declared in `package.json` (auto-installed by UPM):

| Package | Version | Why |
|---|---|---|
| `com.unity.xr.arfoundation` | 5.1.0 | environmentDepth/occlusion, camera intrinsics |
| `com.unity.xr.arkit` | 5.1.0 | iOS LiDAR `sceneDepth` backend |
| `com.unity.sentis` | 2.1.0 | on-device Recon ONNX inference |

Unity **2022.3 LTS+**. For Android add `com.unity.xr.arcore` and replace the UV→pixel
mapping in `ARFoundationDepthProvider` with the ARCore FOV-crop mapping
(`XRCpuImage`/displayMatrix) — depth there is a crop, not a uniform scale.

### Install
Package Manager → **Add package from disk…** → select `unity/DepthRefinement/package.json`,
or add a `file:` dependency to your project `manifest.json`.

## Setup (scene wiring)

1. AR rig: `ARSession`, `XR Origin` with `ARCameraManager` and `AROcclusionManager`.
2. On the `AROcclusionManager` set **Environment Depth Mode = Medium/Best** (and
   **Environment Depth Temporal Smoothing** off for fast hands).
3. Add `ARFoundationDepthProvider` next to the `AROcclusionManager`; assign the
   `ARCameraManager`.
4. Add `SentisHandJointProvider`; assign the Recon `.onnx` **ModelAsset** and feed
   `InputTexture` each frame (cropped hand region from the AR camera background).
   For a no-device smoke test use `MockHandJointProvider` instead.
5. Add `DepthRefinementManager`; assign the joint provider and depth provider components.

## Inspector settings (DepthRefinementManager)

| Group | Field | Default | Meaning |
|---|---|---|---|
| Depth Sampling (FR-4) | Window Radius | 2 | neighborhood = (2r+1)² |
| | Min Pixel Confidence | 0.5 | drop pixels below |
| | Min Valid Ratio | 0.3 | require this fraction valid |
| | Max Local Spread (m) | 0.02 | flying-pixel rejection band (~finger thickness) |
| Fusion (FR-5/6) | Occlusion Threshold (m) | 0.03 | reject |sampled−predicted| above |
| | Min Inliers | 3 | min joints to trust alignment |
| | Flip Depth Axis | false | enable if model +z faces camera |
| Bone (FR-7) | Bone Strength | 0.5 | 0=off, 1=hard snap |
| | Pin Wrist | true | keep root fixed |
| | Reference Bone Lengths | (empty) | optional 20-length override (m) |
| Temporal (FR-6) | Enable Temporal Filter | true | 1€ smoothing |
| | Filter Min Cutoff / Beta / DCutoff | 1.0 / 0.02 / 1.0 | XY 1€ params |
| | Filter Z Min Cutoff / Z Beta | 0.6 / 0.01 | stronger on noisy depth axis |
| Performance | Depth Stride | 1 | run depth every Nth frame (asymmetric rate) |

`SentisHandJointProvider`: **Model Asset**, **Backend** (GPUCompute), **Crop Rect**
(normalized region the model runs on), **Handedness**, frame size.

Subscribe to `DepthRefinementManager.OnRefined(HandPose, Outcome)` for results;
each joint carries `RefinedPosition` (metric, CV camera space), `WasRefined`, `DepthConfidence`.

## Testing procedure

**EditMode unit tests** (no device needed):
1. Open **Window → General → Test Runner → EditMode** and **Run All**, or CLI:
   ```
   Unity -batchmode -runTests -projectPath <YourProject> -testPlatform EditMode \
         -testResults results.xml -quit
   ```
2. Expected: all tests pass — pinhole reconstruction (1e-4), 1€ variance reduction +
   step convergence, bone-length snap, sampler validity/flying-pixel rejection, fusion
   root-depth recovery, occlusion rejection, and RGB-only fallback.

**Editor smoke test (no device):** scene with `MockHandJointProvider` +
`DepthRefinementManager` + `HandJointVisualizer` (Debug Visualization sample). With no
depth provider it runs RGB-only (joints red); add a depth provider to see refined joints (green).

**On-device (LiDAR iPhone):** build to an iPhone Pro, point at a hand ~30–50 cm away,
confirm refined joints track in depth and the Profiler holds ≥30 FPS. Per PRD, run the
**Phase-0 feasibility spike** (offline camera-space, non-PA benchmark) before trusting
absolute-Z accuracy targets.

## Notes / limitations

- Output is **CV camera space** (+X right, +Y down, +Z forward). Convert to Unity world
  via the AR camera transform (the visualizer flips Y).
- Fingertips are sub-pixel under 256×192 LiDAR — corrected **indirectly** via root/palm
  anchoring + bone constraints, not by direct sampling (PRD SM-6).
- Absolute scale comes only from depth (LiDAR) + bone-length-fixed scale, never from a
  circular hand-driven rescale of relative depth.
- Sentis API targets 2.x; adjust `Worker`/`Tensor` calls if you pin another version.
