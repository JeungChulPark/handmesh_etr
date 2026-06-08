# Deferred Work

Surfaced during review of `spec-depth-refinement-unity.md` (2026-06-08). Not this story's problem — collected for later focused attention.

## Android / ARCore path (PRD cross-platform, Phase-0 gated)
- `ARFoundationDepthProvider.TryConvertUvToDepthPixel` implements only the ARKit uniform-scale mapping. ARCore depth is a FOV **crop** — implement `XRCpuImage.transformCoordinates2d`/displayMatrix mapping and ARCore stale-depth detection.
- No metric **monocular** `IDepthProvider` exists (PRD Android primary depth source). Add one (e.g. a Sentis Depth-Anything-V2-metric provider) behind `IDepthProvider`.
- Rationale: PRD scopes Android behind the Phase-0 feasibility spike; v1 implementation is iOS/LiDAR-first by design.

## Sentis inference hardening (perf / PRD FR-1)
- `SentisHandJointProvider` runs synchronous `Schedule` + immediate `ReadbackAndClone` each frame (blocking GPU readback, `DownloadToArray` allocates). Move to async readback + latest-only buffering + stale-frame drop to protect the 30 FPS / GC-free budget.
- Crop-bbox ↔ depth-space intrinsics co-registration (PRD FR-1 "systematic Z 오차 방지"): track intrinsics through the Recon crop/resize, not just the depth provider.

## Handedness-aware bones / calibration
- `HandPose.Handedness` is carried but never consumed. Add handedness-specific reference bone lengths and/or first-frame bone-length calibration (spec "Ask First").

## Tests / validation
- Integration test for CV camera-space → Unity world transform (currently only in the visualizer, untested).
- Device validation of model z-axis sign (`FlipDepthAxis`) and Sentis tensor y-orientation.
- Multi-joint simultaneous occlusion (MAD inlier-band inflation) case.
