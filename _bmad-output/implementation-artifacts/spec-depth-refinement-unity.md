---
title: 'Depth-Aware Hand Pose Refinement (Unity AR Foundation)'
type: 'feature'
created: '2026-06-08'
status: 'done'
baseline_commit: 'afdc3e40b9a760a44b4b31124bec1ba018108e18'
context:
  - '{project-root}/_bmad-output/planning-artifacts/prds/prd-handmesh_etr-2026-06-08/prd.md'
  - '{project-root}/_bmad-output/planning-artifacts/prds/prd-handmesh_etr-2026-06-08/addendum.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** 기존 RGB Recon(MobRecon, 출력 21×3 sigmoid root-relative)의 3D 관절은 절대 metric 깊이가 없어 Z축·손가락 끝 오차가 크다. Unity AR Foundation 앱에서 이를 device depth(iOS LiDAR / cross-platform environmentDepth)로 보정할 수단이 없다.

**Approach:** `unity/DepthRefinement/` UPM 패키지를 신규 생성한다. AR Foundation depth를 관절 2D 위치에서 robust 샘플링하고, intrinsics로 metric unproject한 뒤, shift-only root-depth 정렬 + confidence fusion + bone-length 제약 + One-Euro 필터를 거쳐 보정 관절을 산출한다. Recon은 Unity Sentis(ONNX)로 온디바이스 추론하며, 보정 모듈은 `IHandJointProvider`/`IDepthProvider` 인터페이스 뒤에서 제공자 교체가 가능하다. 30 FPS·모바일 최적(GC-free, CPU 샘플링) 목표.

## Boundaries & Constraints

**Always:**
- 21-joint 규약 고정: index 0=wrist(root), bone edges = MediaPipe 위상(0-1-2-3-4 thumb, 0-5-6-7-8 index, 0-9-10-11-12 middle, 0-13-14-15-16 ring, 0-17-18-19-20 pinky).
- 모든 metric unproject는 카메라 intrinsics를 요구하며, depth 해상도로 스케일된 fx,fy,cx,cy 사용.
- Depth 샘플은 단일 픽셀이 아닌 이웃 윈도 median + confidence 마스킹; 저신뢰/가림 샘플 거부.
- 절대 scale은 외부 신호(depth)에서만; relative depth를 손으로 스케일해 metric화하는 순환 금지(PRD §FR-3).
- v1 정렬 백본은 **shift-only**(bone-length로 scale 고정), RANSAC robust.
- FR-8 회귀 게이트: 유효 depth 부재/정렬 실패/제약 위반 시 보정 기각 → RGB-only 폴백, 관절별 `wasRefined` 플래그.
- 모바일 최적: 프레임 루프 GC 할당 회피(NativeArray/사전할당 버퍼), depth는 CPU(XRCpuImage) 샘플링.

**Ask First:**
- ONNX 모델 에셋 파일(.onnx)의 실제 경로/입출력 이름이 추정과 다를 때.
- bone-length 기준값 출처(MANO 상수 vs 첫 프레임 캘리브) 변경.
- depth를 매 프레임이 아닌 N프레임마다 실행하는 비대칭 레이트 도입 여부.

**Never:**
- Recon 모델 가중치 재학습·수정.
- Unity 시각화/XR 인터랙션의 제품화(디버그 viz 샘플까지만; PRD v2).
- 양손 동시 상호작용·손-객체 접촉(단일 손 보정만).
- relative-only depth 모델을 Depth Source로 사용.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Happy path | HandPose(21 joints, 2D+rootRel 3D, conf, handed) + env depth + confidence + intrinsics | metric 보정 21 joints, 관절별 wasRefined=true | N/A |
| Depth 없음/미준비 | depth texture/ CPU image 획득 실패 | RGB-only joints 그대로 출력, wasRefined=false | 경고 1회, 폴백 |
| 가림/저신뢰 관절 | 샘플 conf<τ 또는 |z_depth−z_rgb|>τ_occ | 해당 관절 depth 거부, 전역 정렬·RGB로 폴백 | 관절별 wasRefined=false |
| Intrinsics 미가용 | TryGetIntrinsics 실패 | 보정 스킵, RGB-only 출력 | 경고 1회 |
| 손 미검출 | provider가 빈 HandPose | 파이프라인 no-op, 직전 유효 유지 옵션 | N/A |
| 손가락 끝(sub-pixel) | fingertip 직접 샘플 불가 | root/palm 앵커+bone 제약 간접 보정, 직접 depth 미적용 | N/A |

</frozen-after-approval>

## Code Map

- `models/mobrecon_ds.py`, `models/manolayer.py` -- Recon 출력 규약 근거(21 joints, parent tree, sigmoid root-relative).
- `convert_to_onnx.py`, `Extra_model_input256_onnx_convert.ipynb` -- ONNX I/O: input `input0`(1,3,256,256, [0,1] ToTensor), output `(1,21,3)`.
- (신규) `unity/DepthRefinement/` -- 본 작업 산출물 전체. 기존 Python 코드 변경 없음.

## Tasks & Acceptance

**Execution:**
- [x] `unity/DepthRefinement/package.json` -- UPM 매니페스트(name, deps: com.unity.xr.arfoundation, com.unity.xr.arkit, com.unity.sentis), Samples~ 등록.
- [x] `unity/DepthRefinement/Runtime/DepthRefinement.Runtime.asmdef` -- 런타임 어셈블리(AR Foundation/Sentis 참조).
- [x] `unity/DepthRefinement/Runtime/Core/HandJoint.cs` -- `HandJointId` enum(21), bone-edge 정적 테이블, `HandJointData` struct(2D uv, rootRel 3D, confidence, wasRefined).
- [x] `unity/DepthRefinement/Runtime/Core/HandPose.cs` -- 프레임 컨테이너(joints[21], Handedness, timestamp, frame size).
- [x] `unity/DepthRefinement/Runtime/Core/IHandJointProvider.cs` -- `bool TryGetLatest(out HandPose)` + 이벤트.
- [x] `unity/DepthRefinement/Runtime/Core/IDepthProvider.cs` -- depth/confidence CPU 접근 + `bool TrySampleMetricDepth(Vector2 uvNorm, out float meters, out float conf)` + intrinsics 제공.
- [x] `unity/DepthRefinement/Runtime/Depth/ARFoundationDepthProvider.cs` -- Step1-3: `AROcclusionManager` environmentDepth(+confidence) CPU 획득, `ARCameraManager.TryGetIntrinsics` depth 해상도 스케일, displayMatrix 좌표 변환. iOS sceneDepth=environmentDepth 매핑.
- [x] `unity/DepthRefinement/Runtime/Refinement/JointDepthSampler.cs` -- Step4: uv→이웃 median 샘플 + confidence 마스킹 + occlusion/flying-pixel 거부.
- [x] `unity/DepthRefinement/Runtime/Refinement/JointReconstructor.cs` -- Step5: (u,v,depth)+intrinsics→metric (x,y,z) unproject.
- [x] `unity/DepthRefinement/Runtime/Refinement/ConfidenceFusion.cs` -- Step6: shift-only root-depth 정렬(RANSAC) + 유효 관절 `w=c_vis*c_depth` 블렌드.
- [x] `unity/DepthRefinement/Runtime/Refinement/BoneLengthConstraint.cs` -- Step7: 20 bone 길이 기준으로 위반 관절 보정/거부.
- [x] `unity/DepthRefinement/Runtime/Refinement/OneEuroFilter.cs` -- Step8: 관절·축별 1-Euro(속도 적응), Z 채널 집중.
- [x] `unity/DepthRefinement/Runtime/Refinement/DepthRefinementPipeline.cs` -- FR-4~8 오케스트레이션 + 회귀 게이트(wasRefined), 순수 C#(테스트 가능).
- [x] `unity/DepthRefinement/Runtime/Recon/SentisHandJointProvider.cs` -- Step9: Sentis로 `input0`(256 crop, [0,1]) 추론→`(21,3)`→HandPose(uv 정규화·rootRel z). crop bbox는 Inspector/외부 입력.
- [x] `unity/DepthRefinement/Runtime/Mock/MockHandJointProvider.cs` -- 합성 HandPose(테스트/에디터 검증용).
- [x] `unity/DepthRefinement/Runtime/DepthRefinementManager.cs` -- Step1: MonoBehaviour, provider+pipeline 배선, `[SerializeField]` Inspector 설정(window size, τ 임계, One-Euro β/minCutoff, depth stride), 보정 결과 이벤트.
- [x] `unity/DepthRefinement/Samples~/DebugVisualization/HandJointVisualizer.cs` -- Step10: LineRenderer/Gizmo로 RGB vs Refined 관절·bone, wasRefined 색상.
- [x] `unity/DepthRefinement/Tests/DepthRefinement.Tests.asmdef` + `Tests/Editor/*Tests.cs` -- I/O Matrix edge-case EditMode 단위 테스트(Reconstructor 수치, OneEuro 평활, BoneLength, Sampler 거부 로직).
- [x] `unity/DepthRefinement/README.md` -- 폴더 구조, 패키지 의존성, Inspector 설정, 테스트 절차(설치/Test Runner/디바이스).

**Acceptance Criteria:**
- Given 알려진 intrinsics와 합성 depth, when JointReconstructor가 (u,v,depth)를 unproject, then 해석적 기대 (x,y,z)와 1e-4 이내 일치.
- Given 정지 입력에 노이즈, when One-Euro 적용, then 출력 분산이 입력 대비 감소하고 step 입력에 과도 lag 없음(테스트로 검증).
- Given 한 관절의 depth가 RGB와 임계 이상 불일치, when 파이프라인 실행, then 그 관절 wasRefined=false이고 RGB 값 유지.
- Given depth/intrinsics 미가용, when 파이프라인 실행, then 전체 RGB-only 폴백 + 경고 1회, 예외 없음.
- Given bone 길이 위반 보정, when BoneLengthConstraint 적용, then 각 bone 길이가 기준의 허용오차 내로 수렴.
- Given 패키지 임포트, when Unity가 컴파일, then Runtime/Editor/Tests asmdef가 오류 없이 빌드되고 EditMode 테스트가 통과.

## Design Notes

- **좌표 규약:** Recon 출력 x,y는 256 crop 내 정규화[0,1] → full-frame uv = crop bbox로 매핑. z는 정규화 rootRel(절대 아님) → 절대 scale은 depth에서만. 샘플링은 full-frame uv를 depth-image 좌표로(displayMatrix) 변환 후 수행.
- **shift-only 정렬:** scale은 bone-length 기준으로 고정, root-depth(shift) t만 RANSAC 최소제곱으로 추정 → 정면 손 s·t 공선성 회피(PRD addendum B).
- **Sentis I/O:** input `input0`=(1,3,256,256) NCHW [0,1](ToTensor, mean/std 없음), output=(1,21,3) sigmoid. 출력 이름은 export별 `output0`/`identity` 가변 → Inspector로 지정.
- **모바일:** depth CPU 읽기는 `XRCpuImage`+`NativeArray<short/float>`; 파이프라인은 struct/사전할당 배열로 GC 0; depth는 필요 시 stride>1.

## Verification

**Commands:**
- Unity EditMode 테스트(Test Runner 또는 CLI): `Unity -batchmode -runTests -testPlatform EditMode -projectPath <proj>` -- expected: 전 테스트 통과.

**Manual checks (if no CLI):**
- 패키지를 빈 Unity(2022 LTS+, AR Foundation 5/6) 프로젝트에 임포트 → 컴파일 오류 0, asmdef 해소.
- MockHandJointProvider + DebugVisualization 샘플 씬에서 RGB vs Refined 관절이 그려지고 wasRefined 색상이 토글됨.
- LiDAR iPhone 디바이스 빌드에서 environmentDepth 획득·30 FPS 유지(프로파일러).

## Suggested Review Order

**Design intent (start here)**

- 전체 흐름 + FR-8 회귀 게이트: 샘플→정렬→재구성→융합→제약→필터, 실패 시 RGB-only 폴백
  [`DepthRefinementPipeline.cs:102`](../../unity/DepthRefinement/Runtime/Refinement/DepthRefinementPipeline.cs#L102)

**Core metric math (highest risk)**

- shift-only root-depth 정렬(RANSAC/MAD) + confidence 융합 + fingertip/가림 제외
  [`ConfidenceFusion.cs:75`](../../unity/DepthRefinement/Runtime/Refinement/ConfidenceFusion.cs#L75)
- bone-length로 metric scale 고정 — 순환 스케일링 방지의 핵심
  [`ConfidenceFusion.cs:53`](../../unity/DepthRefinement/Runtime/Refinement/ConfidenceFusion.cs#L53)
- 핀홀 unproject (intrinsics 필수)
  [`JointReconstructor.cs:19`](../../unity/DepthRefinement/Runtime/Refinement/JointReconstructor.cs#L19)
- 이웃 median 샘플 + 경계내 분모 + flying-pixel 거부
  [`JointDepthSampler.cs:55`](../../unity/DepthRefinement/Runtime/Refinement/JointDepthSampler.cs#L55)
- bone 길이 제약(forward 순서, strength 보간)
  [`BoneLengthConstraint.cs:30`](../../unity/DepthRefinement/Runtime/Refinement/BoneLengthConstraint.cs#L30)
- Z 집중 1€ 시간 평활(dt 클램프)
  [`OneEuroFilter.cs:26`](../../unity/DepthRefinement/Runtime/Refinement/OneEuroFilter.cs#L26)

**Device & model integration (boundary-crossing)**

- AR Foundation environmentDepth/sceneDepth + confidence 유효성 + intrinsics 스케일
  [`ARFoundationDepthProvider.cs:53`](../../unity/DepthRefinement/Runtime/Depth/ARFoundationDepthProvider.cs#L53)
- Sentis ONNX 추론 + 출력 shape 검증
  [`SentisHandJointProvider.cs:78`](../../unity/DepthRefinement/Runtime/Recon/SentisHandJointProvider.cs#L78)
- provider 배선 + HandPose 소유 복사(aliasing 방지) + depth stride
  [`DepthRefinementManager.cs:93`](../../unity/DepthRefinement/Runtime/DepthRefinementManager.cs#L93)

**Contracts & data (supporting)**

- 깊이 소스 계약(픽셀 접근·UV 매핑·intrinsics)
  [`IDepthProvider.cs:23`](../../unity/DepthRefinement/Runtime/Core/IDepthProvider.cs#L23)
- 21-joint 위상·bone edges·fingertip mask·reference 길이
  [`HandJoint.cs:1`](../../unity/DepthRefinement/Runtime/Core/HandJoint.cs#L1)

**Tests (peripherals)**

- 정렬/가림 거부/폴백 1회 경고/fingertip 제외
  [`PipelineGateTests.cs:1`](../../unity/DepthRefinement/Tests/Editor/PipelineGateTests.cs#L1)
- unproject 수치·1€ 분산감소·bone 수렴·sampler 거부
  [`JointReconstructorTests.cs:1`](../../unity/DepthRefinement/Tests/Editor/JointReconstructorTests.cs#L1)
