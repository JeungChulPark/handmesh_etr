---
stepsCompleted: [1, 2]
inputDocuments: []
workflowType: 'research'
lastStep: 1
research_type: 'technical'
research_topic: 'Mobile Depth-Aware Hand Pose Estimation System'
research_goals: 'RGB 기반 Hand Pose 추정 결과를 Depth 정보로 보정하여 3D Joint 정확도(특히 Z축/손가락 끝)를 개선하는 모바일 실시간 시스템 설계'
user_name: 'Jucpark'
date: '2026-06-08'
web_research_enabled: true
source_verification: true
---

# Research Report: technical

**Date:** 2026-06-08
**Author:** Jucpark
**Research Type:** technical

---

## Research Overview

본 연구는 RGB 기반 경량 Hand Pose Estimation(Recon Hand Tracking) 모델의 3D Joint 정확도(특히 Z축 방향 및 손가락 끝 위치)를 Depth 정보로 보정하는 모바일 실시간 시스템을 설계하기 위한 기술 조사이다. 최신 공개 출처(논문/오픈소스)에 대한 검증을 기반으로 하며, Android(RGB-only depth 중심)와 iPhone(LiDAR 중심)을 동등 비중으로 다룬다. 보정 전략은 기존 RGB 파이프라인 최소 변경과 구현 용이성을 우선한다.

---

## Technical Research Scope Confirmation

**Research Topic:** Mobile Depth-Aware Hand Pose Estimation System
**Research Goals:** RGB 기반 Hand Pose 추정 결과를 Depth 정보로 보정하여 3D Joint 정확도(특히 Z축/손가락 끝)를 개선하는 모바일 실시간 시스템 설계

**확정된 방향성 (사용자 확인):**

- **타깃 플랫폼:** Android + iPhone 동등 비중 → Android는 RGB-only depth 중심, iPhone은 LiDAR 중심으로 각각 별도 추천 아키텍처 제시
- **보정 전략:** 실용성/구현 용이성 우선 → 기존 RGB Recon 파이프라인 최소 변경, 검증된 경량 기법 우선 (정확도 트레이드오프 명시)

**Technical Research Scope:**

- Architecture Analysis — RGB+Depth fusion 설계 패턴, Android/iPhone 별 파이프라인 구조
- Implementation Approaches — Depth 추정 → Joint Depth Refinement → 기존 Recon 통합 방법론
- Technology Stack — Depth Anything V2 / FastDepth / MiDaS / MediaPipe / ARKit LiDAR 등
- Integration Patterns — RGB Hand Pose ↔ Depth 보정 fusion (Confidence Fusion, Bone Constraint, Temporal Smoothing)
- Performance Considerations — 30 FPS+, Snapdragon 8 Gen / A17·A18, 메모리 1GB 이하 제약

**Research Methodology:**

- Current web data with rigorous source verification
- Multi-source validation for critical technical claims
- Confidence level framework for uncertain information
- Comprehensive technical coverage with architecture-specific insights

**Scope Confirmed:** 2026-06-08

---

## Technology Stack Analysis

> 본 도메인에서 "기술 스택"은 ① RGB-only 모바일 Depth 추정 모델, ② Hand-specific Depth / 3D Hand Pose 프레임워크, ③ Apple LiDAR / Android Depth API, ④ 모바일 추론 런타임 & 하드웨어로 구성된다. 모든 수치는 2026년 6월 기준 공개 출처로 검증하였으며, Procrustes 정렬(PA) 지표는 **형상** 정확도이지 **절대 Z** 정확도가 아님에 유의한다.

### 1. RGB-only 모바일 Depth Estimation 모델

**핵심 트레이드오프 — Metric vs Relative depth:** 손 관절의 **절대 3D 보정**에는 **metric(절대 스케일) depth**가 필요하다. Relative/disparity 모델(MiDaS, Depth Anything V2 기본 체크포인트)은 `d_true ≈ α·d_pred + β` 형태의 scale-and-shift 모호성을 가지므로, 절대 보정에 쓰려면 매 프레임 α, β를 별도 기준(손 크기/MANO prior, sparse metric anchor)으로 풀어야 하며 이는 오차와 의존성을 추가한다. ([scale-shift](https://arxiv.org/html/2501.07742v1), [affine corrections, CVPR2025](https://arxiv.org/html/2501.05446))

| 모델 | 크기 | 출력 | 정확도(보고) | 모바일 지연/FPS (HW) | Export | License |
|---|---|---|---|---|---|---|
| **Depth Anything V2 – Small** | 24.7M / ~94MB FP32, 49.8MB CoreML FP16 | Relative + **metric 파인튜닝** 별도 | δ1~95.3% (metric 변형은 NYU/KITTI 파인튜닝) | **~34ms** iPhone15Pro Max NE / 24.6ms M3 Max; SD 8-series TFLite | PyTorch·ONNX·CoreML·TFLite (Qualcomm AI Hub) | **Apache-2.0** ✅ |
| FastDepth (MIT) | 0.37 GMACs, MobileNet enc | Metric(NYU 실내 스케일) | δ1 0.771, RMSE 0.604m | 178 FPS TX2 GPU / 27 FPS TX2 CPU | PyTorch/Caffe | **MIT** ✅ |
| MiDaS v2.1 Small | EfficientNet-Lite3 ~16M | **Relative**(역깊이) | zero-shot relative | 30 FPS iPhone11 NPU / 22 FPS SD865 GPU | **ONNX + TFLite**(공식) | **MIT** ✅ |
| Metric3D v2 | ViT-S/L/Giant2 | **Metric** | 7개 zero-shot metric 벤치 1위 | 서버급, 공식 모바일 빌드 없음 | PyTorch | 비상업 경향 ⚠️ |
| UniDepth V2 | ViT-S/B/L14 | **Metric**(intrinsic 자가추정) | d1 98.8% NYUv2 | 서버급, ONNX 추가 | PyTorch·ONNX | **CC-BY-NC-4.0** ⚠️ |
| Apple Depth Pro | ViT-L Dinov2 ×2 (~950M) | **Metric, intrinsic-free** | SOTA zero-shot metric, 경계 선명 | 2.25MP 0.3s (서버 GPU), 폰 실시간 ✗ | PyTorch·CoreML 포팅 | Apple sample-code |
| **PromptDA** (CVPR2025) | DA-V2 + LiDAR prompt | **Metric, 4K** | ARKitScenes SOTA | LiDAR 입력 필요(폰 LiDAR 파이프라인) | PyTorch·HF | 연구용 |

- **상업 친화 + 모바일 + metric 최적해: Depth Anything V2 Small의 metric 파인튜닝** (Apache-2.0, 24.7M, ~34ms NE). Relative scale-shift 해결 불필요. ([DA-V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2), [Qualcomm AI Hub](https://aihub.qualcomm.com/mobile/models/depth_anything_v2), [Apple CoreML](https://huggingface.co/apple/coreml-depth-anything-v2-small))
- 2025년 Mamba 기반 **LMDepth INT8은 2.63MB / ~122 FPS급** — 모바일 depth net이 매우 저렴함을 시사. ([LMDepth](https://arxiv.org/html/2505.00980v1))
- **"MobileMonodepth"**는 단일 정식 모델이 아니라 FastDepth/MiDaS-small/DA-V2-Small 양자화로 채워지는 범주. 벤더 툴킷(Qualcomm AI Hub, CoreML, OpenVINO)이 실제 배포 경로.

### 2. Hand-specific Depth / 3D Hand Pose 프레임워크

**MediaPipe Hand Landmarker (핵심):** 메인스트림 중 유일하게 모바일 실시간 + 상업 허용(Apache-2.0)으로 **world landmarks(미터 단위 3D)** 를 직접 출력. 단 z(깊이)는 단안 학습 추정치로 **약함/근사** — 원점은 손 기하 중심, "z 스케일은 x와 대략 동일", **Google이 깊이 정확도 지표를 공개하지 않음**. → **신뢰 가능한 metric depth 소스가 아니라 prior/sanity-bound로만 사용**. 지연 17.12ms CPU / 12.27ms GPU(Pixel6). ([Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker))

| 방법(연도) | 접근 | FreiHAND PA-MPJPE/MPVPE | 실시간/모바일 | Code & License |
|---|---|---|---|---|
| **MobRecon** (CVPR2022, *현 repo 계열*) | 경량 2D + depth-separable spiral conv 3D | 경쟁력(속도/시간일관성 중심) | **83 FPS @ Apple A14 CPU** ✅ | open (MANO 종속) |
| ReJSHand (2025.03) | refined joint+skeleton, attention | 6.3 / 6.4 mm | **72 FPS** ✅ | open |
| M3DHMR (2025.05) | 실시간 **camera-space** 메시 | 6.5 / 6.6 mm | 실시간, camera-space(metric-ish) ✅ | open |
| Fast-HaMeR (2026) | HaMeR 지식증류 → MobileNet/MobileViT 학생 | HaMeR 대비 ~0.4mm 이내 | ~1.5× 빠름, 헤드셋/폰 타깃 | CC-BY-4.0 |
| HaMeR (CVPR2024) | 순수 트랜스포머, ViT-H | 6.0 / 5.7 mm | 서버급 ✗ | open(MANO) |
| Hamba (NeurIPS2024) | graph-guided Mamba(SSM) | — / 5.3 mm | 연구급 | 연구 license |
| WiLoR (2024) | detection+ViT+mesh refine, **최고 정확도** | **5.5 / 5.1 mm** | 모바일 실시간 ✗ | **CC-BY-NC-ND ⚠️상업 불가** |

- **정확도 사다리:** WiLoR≈Hamba < HaMeR < 실시간군(ReJSHand/M3DHMR). SOTA-heavy와 실시간 모델 간 격차는 **~1mm PA**에 불과 → 모바일 모델로 충분한 경우 多. ([HaMeR](https://geopavlakos.github.io/hamer/), [WiLoR](https://github.com/rolpotamias/WiLoR), [Hamba](https://github.com/humansensinglab/Hamba), [MobRecon](https://arxiv.org/abs/2112.02753), [ReJSHand](https://arxiv.org/abs/2503.05995), [M3DHMR](https://arxiv.org/abs/2505.20058))
- **라이선스 주의:** WiLoR(NC-ND)은 제품 사용 불가, MANO 종속 모델은 MANO 학술 라이선스 확인 필요. **MobRecon/ReJSHand/M3DHMR/MediaPipe가 상업적으로 안전.**
- **HandNeRF**(ICCV2023 등)는 멀티뷰/오프라인, 레이당 다수 포인트 쿼리로 **모바일 실시간 불가** → 보정 모듈로 부적합. Gaussian/point splatting 대안도 경량 온디바이스 모듈로는 부적합. ([HandNeRF](https://arxiv.org/abs/2303.13825))
- **"MonoHand3D"/"HandDepth"**는 단일 정식 논문으로 확정되지 않는 일반 용어. 실질 내용은 2.5D/root-depth 및 깊이 refinement 기법(§아래)에 해당.

**Z축 모호성 해소 + Joint Depth Refinement 기법** (기존 RGB 예측에 bolt-on, 모두 경량/모바일 가능):
1. **2.5D 표현**(지배적): 2D 키포인트 + scale-normalized root-relative depth, 기준 본 길이로 정규화 → scale/translation 불변. ([Iqbal 2.5D](https://arxiv.org/pdf/1804.09534))
2. **Bone-length 정규화 / 제약**: MANO 본 길이 비율로 잘못된 z를 거부·수정. ([ACCV2024](https://dl.acm.org/doi/10.1007/978-981-96-0885-0_14), [MS-MANO](https://arxiv.org/html/2404.10227v1))
3. **절대 root-depth 복원(RootNet-style)** + 카메라 intrinsics → metric XYZ.
4. **Joint-wise depth sampling + confidence-weighted fusion**: metric depth map을 각 투영 관절 위치에서 샘플링, 국소 이웃·신뢰도로 가중 융합(가림/노이즈 대응).
5. **Temporal smoothing**: z 채널 jitter 대폭 감소(정확도 비용 작음, 체감 개선 큼).

### 3. Apple LiDAR / Android Depth API 스택

> **공통 핵심 한계:** 모든 소비자용 모바일 depth API는 환경/룸스케일(~0.5–5m) 대상이며, **<30cm 근거리의 cm 단위 손가락을 분리 해상하도록 설계되지 않았다.** 샘플 depth는 관절별 **Z prior**로 쓰되 손가락 정밀 GT로 신뢰하면 안 된다.

| | ARKit sceneDepth (LiDAR) | smoothedSceneDepth | ARCore Depth API | ARCore Raw Depth |
|---|---|---|---|---|
| 센싱 | 능동 LiDAR + ML | LiDAR + 시간평균 | depth-from-motion(+ToF) | 동상, 고신뢰만 |
| 디바이스 | iPhone/iPad **Pro**(12 Pro→17 Pro) | 동상 | ARCore 87%+, **센서 불필요** | 동상 |
| 해상도 | **256×192** | 256×192 | ~**160×120** | sparse |
| 단위 | meters(Float32) | meters | mm(uint16) | mm + confidence |
| FPS | **최대 60Hz** | 최대 60Hz | 프레임당(모션 게이트) | 프레임당 |
| 권장 범위 | ~0.5–5m, 최소 ~30cm | 동상 | ~0.5–5m | ~0.5–5m |
| 정확도 | 근거리 소형 ~±1cm | 더 부드럽지만 지연 | 거칠고 모션 의존 | 높지만 sparse |

- **iOS:** `ARFrame.sceneDepth`→`ARDepthData`(depthMap + confidenceMap, 둘 다 256×192). depthMap은 광축 Z(미터). `ARConfidenceLevel`(low/medium/high)로 **저신뢰 픽셀 마스킹 필수**. `ARCamera.intrinsics`를 256×192로 스케일 후 `X=(u−cx)Z/fx, Y=(v−cy)Z/fy`로 언프로젝트. **단일 픽셀이 아니라 신뢰도 마스킹된 이웃의 median 샘플링** 권장(손가락이 sub-pixel). ([sceneDepth](https://developer.apple.com/documentation/arkit/arframe/3566299-scenedepth), [intrinsics](https://developer.apple.com/documentation/arkit/arcamera/intrinsics))
- **LiDAR 디바이스:** **Pro 전용**(iPhone 12 Pro~17 Pro, 2020+ iPad Pro). **비-Pro(16, 16e 포함) LiDAR 없음**. 칩 세대(A17 Pro=15 Pro, A18 Pro=16 Pro)와 LiDAR 탑재는 무관. ([LiDAR 목록](https://help.roomsketcher.com/hc/en-us/articles/29949063142045-Does-My-Phone-or-Tablet-Have-LiDAR))
- **"EnvironmentDepth"는 ARKit 용어가 아니라 Unity AR Foundation 추상화** → iOS에서 `AROcclusionManager.environmentDepthTexture`가 `sceneDepth.depthMap`에 매핑, `...TemporalSmoothingRequested=true`면 smoothedSceneDepth. human depth/stencil(사람 분할)은 비-LiDAR A12+에서도 동작하나 손가락 정밀 불가. ([ARKit Occlusion](https://docs.unity3d.com/Packages/com.unity.xr.arkit@6.0/manual/arkit-occlusion.html))
- **LiDAR 손 한계:** 256×192 저해상(손가락 sub-pixel), **<~30cm 최악 영역**, 얇고 빠른 표면에서 노이즈·저신뢰, 직사광 IR 산란. 정면 근거리 손은 **TrueDepth(AVFoundation depth)** 가 더 고해상이나 세대별 캘리브 불일치 존재. ([MDPI 특성](https://www.mdpi.com/1424-8220/23/18/7832))
- **Android:** ARCore Depth API는 **depth-from-motion**(ToF 있으면 자동 융합). 문제는 **정지 카메라 + 움직이는 손 → 손 영역 시차 부족 → invalid/garbage depth**(LiDAR보다 불리). **Raw Depth + confidence 마스킹** 권장. Android ToF는 2019–2020 정점 후 쇠퇴(삼성 S22부터 미탑재) → **현대 Android에 ToF 가정 불가**. ([Depth API](https://developers.google.com/ar/develop/depth), [Raw Depth](https://developers.google.com/ar/develop/java/depth/raw-depth), [S22 ToF 미탑재](https://www.gsmarena.com/report_samsung_considered_bringing_back_the_3d_tof_sensor_for_the_galaxy_s22_decided_against_it-news-48719.php))

### 4. 모바일 추론 런타임 & 하드웨어

**런타임 (2025–2026 변동 사항):**
- **TFLite → LiteRT 개명**(2024말). GPU는 신규 **MLDrift** 가속. **NNAPI는 Android 15에서 deprecated** → LiteRT + 벤더 NPU delegate 권장. **2025.11 LiteRT-QNN NPU Accelerator** 출시(구 Hexagon delegate 대체): CPU 대비 최대 100×, GPU 대비 ~10×, SD 8 Elite Gen5에서 72개 중 64개 모델 NPU 완전 위임. ([LiteRT-QNN](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/), [NNAPI 마이그레이션](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide))
- **ONNX Runtime Mobile:** EP 기반 — Android(QNN EP=Hexagon NPU, XNNPACK), iOS(Core ML EP). 2025 QNN EP **Adreno GPU 백엔드** 프리뷰. ([QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html))
- **Core ML:** CPU/GPU/ANE 자동 분할이나 **배치 결정 불투명·스케줄 제어 불가** → 2모델 동시 시 ANE에서 직렬화 가능성(설계 시 고려). ([ANE 한계](https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in))
- **Qualcomm:** SNPE+QNN → **QAIRT** 통합, **AI Hub**로 실측 벤치/최적화 모델 제공.
- **경량 런타임:** MNN(CPU 20–40% 빠름 경향), NCNN — 벤더 NPU SDK 없이 의존성 가벼운 경로. **PyTorch ExecuTorch** v1.2 Beta(Meta 프로덕션), Core ML/Qualcomm 백엔드 — 단 타이트한 30FPS 듀얼모델 NPU 경로는 LiteRT-QNN/ORT-QNN이 더 성숙. ([ExecuTorch](https://pytorch.org/blog/executorch-beta/))

**하드웨어 (NPU/ANE):**
| SoC/Chip | NPU 성능 | 비고 |
|---|---|---|
| SD 8 Gen 3 (2023) | 8 Gen 2 대비 ~98%↑ | 멀티모달 genAI |
| SD 8 Elite Gen4 (2024) | 8 Gen 3 대비 +45% | |
| SD 8 Elite Gen5 (2025.09) | **~80 TOPS** | |
| Apple A17 Pro/A18/A18 Pro | 16-core **35 TOPS** | A18=동일 raw, 효율↑ |
| Apple A19 Pro (2025) | **48 TOPS** | A18 Pro 대비 최대 4× |

**실측 지연 (33ms 예산 대비 충분):** MediaPipe Hand Detector ~1.01ms (QNN), Landmark ~1.30ms, MiDaS-V2 depth ~1.1ms(S23 양자화). **메모리:** MiDaS INT8 ~16.9MB, MediaPipe hand 합산 ~15MB → **앱+2모델 1GB 미만 용이**. INT8은 ~4× 축소 + NPU 네이티브 가속(1–2% 정확도 하락). ([Midas AI Hub](https://aihub.qualcomm.com/compute/models/midas), [MediaPipe-Hand AI Hub](https://aihub.qualcomm.com/models/mediapipe_hand))

**동시성 결론:** SD 8 Gen3/Elite, A18/A19에서 **hand 모델 + MiDaS급 depth 모델 INT8 동시 30FPS 실현 가능**. 단일 NPU 멀티플렉싱 시 각 모델 지연 상승 → **depth는 NPU, hand는 GPU로 유닛 분리** 또는 시간 stagger 권장(NPU·GPU 메모리 공유). 병목은 FLOPs가 아니라 **카메라/ISP·전처리·2모델 스케줄 jitter**. ([동시실행 연구](https://arxiv.org/pdf/2308.05869), [이종 SoC 동시성](https://arxiv.org/html/2501.14794v2))

**Depth 모델 vs LiDAR 컴퓨트 트레이드오프:**
- **LiDAR는 NPU/GPU 추론 비용 ~0**(전용 센서 HW)·어두운 곳/무텍스처에서도 동작 → **Pro iPhone만 타깃이면 depth 모델 생략, NPU 전체를 hand에 할애**가 최저전력·최고신뢰 경로.
- Android/비-Pro iPhone/임의 RGB 카메라 지원 필요 시 → **양자화 경량 단안 depth net**(MiDaS-small/LMDepth급) 실행, 30FPS 듀얼 예산 내 수용 가능.
- **하이브리드:** LiDAR 가용 시 LiDAR, 그 외 depth net 폴백. ([Prompt Depth Anything](https://arxiv.org/pdf/2412.14015))

**기술 스택 출처 신뢰도:** 모델 카드/공식 문서/논문 기반 high confidence. 일부 항목(Fast-HaMeR 정확 FPS, WiLoR recon FPS)은 정성적. FreiHAND PA 지표는 형상 정확도이므로 **절대 Z 정확도를 직접 보증하지 않음** → 절대 깊이는 intrinsics/센서/temporal cue에서 와야 한다는 점을 재확인.

## Integration Patterns Analysis

> 본 도메인의 "통합 패턴"은 ① RGB Hand Pose ↔ Depth 보정 **fusion 알고리즘 정식화**, ② RGB/Depth **시간·좌표 동기화 및 캘리브레이션**, ③ **온디바이스 파이프라인/스레딩**, ④ **기존 MobRecon 모델과의 결합 방식**, ⑤ **ARKit/ARCore 세션 공존**으로 구성된다.

### 1. Fusion 알고리즘 패턴 (RGB 3D Joint ← Depth 보정)

**공통 핵심 연산 — Back-projection** (image+depth → camera-space 3D): `Z=D(u,v); X=(u−cx)Z/fx; Y=(v−cy)Z/fy`. RootNet의 절대 포즈 복원과 동일 연산. ([RootNet](https://arxiv.org/abs/1907.11346))

**① Joint-wise depth sampling + back-projection**
- 예측 2D 관절 `u_j`에서 depth 샘플링 → back-project. 해상도 불일치 시 **bilinear 금지(불연속에서 가짜 깊이 생성), nearest 또는 연결된 segment 내 보간**.
- **이웃 median**: `Z_j = median{ D(u,v) : (u,v)∈N(u_j), valid }` — 실루엣 경계의 전경/배경 혼합·ToF flying pixel 제거. window median 대비 ~15–20mm(손가락 두께) 이상 차이 픽셀 제거 후 평균. ([IJCV'15](https://link.springer.com/article/10.1007/s11263-015-0826-9), [KeypointFusion AAAI'24](https://github.com/ru1ven/KeypointFusion))

**② Confidence-weighted fusion (per-joint Z blend)**
- `z_j^fused = w_j·z_j^depth + (1−w_j)·z_j^rgb`, `w_j = c_j^vis · c_j^depth` (히트맵 신뢰도 × depth 신뢰도).
- 원리적(역분산/Bayesian) 형태: `w_j = σ_rgb²/(σ_rgb²+σ_depth²)` → 최소분산 융합. depth가 저진폭/경계 근접이면 σ_depth↑, 히트맵 평탄하면 σ_rgb↑. ([CrossFuNet](https://www.mdpi.com/1424-8220/21/18/6095), [RGB-D body fusion](https://www.mdpi.com/2076-3417/15/15/8746))

**③ Global scale + root-depth alignment (RootNet-style) — 최고 가성비·최저 위험**
- 단안 RGB net은 root-relative·scale-ambiguous → **모든 관절이 아니라 전역 scale·root translation만** 견고한 depth 샘플(wrist/palm)로 고정, 잘 학습된 상대 articulation 보존.
- **가중 최소제곱 scale-and-shift fit**: `(s*,t*) = argmin Σ_{j∈V} w_j(s·ẑ_j + t − z_j^depth)²` (closed-form) → `z_j^aligned = s*ẑ_j + t*`를 depth 없는 관절 포함 전체에 적용.
- **RANSAC + MAD**: 손바닥 뒤 손가락의 gross outlier가 LSQ를 망치므로 robust fit 필수. wrist + palm-MCP로 seed 후 inlier 확장. ([RSA NeurIPS'24](https://proceedings.neurips.cc/paper_files/paper/2024/file/cc92809cd8dfbd035801966ab4896741-Paper-Conference.pdf), [RANSAC scale recovery](https://www.researchgate.net/publication/349199673_Self-Supervised_Monocular_Depth_Estimation_Scale_Recovery_using_RANSAC_Outlier_Removal))

**④ Energy minimization / optimization fusion**
- `E(J)=λ_data·E_rgb + λ_depth·E_depth + λ_bone·E_bone + λ_temp·E_temp + λ_reg·E_reg`
  - `E_rgb = Σ_j c_j^vis‖Π_K(J_j)−u_j‖²` (재투영), `E_depth = Σ_j w_j(J_{j,z}−z_j^depth)²`, `E_bone = Σ(‖J_a−J_b‖−ℓ_ab)²`, `E_temp = Σ‖J_j^t−J_j^{t−1}‖²`, MANO prior `‖θ‖²,‖β‖²`.
- 솔버: 비선형 MANO fit엔 **Levenberg–Marquardt/Gauss–Newton**. RGB term이 depth 없는 관절을 정규화, depth term이 유효 관절을 metric anchor. ([ShaRPy 불확실성 가중](https://arxiv.org/abs/2303.10042), [Kulon CVPR'20](https://arxiv.org/pdf/2004.01946))

**⑤ Kalman / temporal filtering (Z 채널, cross-frame)**
- **1-Euro filter**(XR 손추적 사실상 표준): 속도 적응 저역통과 `fc = fcmin + β·|ẋ|`. 느릴 때 jitter 억제, 빠를 때 lag 억제. Z 채널에 특히 유효. ([1€ filter](https://gery.casiez.net/1euro/))
- **Kalman**: 측정 공분산 `R_t`를 관절·프레임별 depth 신뢰도로 설정 → 가려진 depth는 무시(예측 coast), 좋은 depth는 신뢰. §2의 시간 확장. ([Kalman in depth space](https://link.springer.com/article/10.1186/1687-6180-2012-36))

**⑥ Learned fusion**: keypoint 위치에서 RGB feature + back-projected point feature 융합 → 보정 3D joint 회귀. KeypointFusion(DexYCB PA 4.79mm SOTA), Pyramid Deep Fusion. *frozen RGB net 위 최소 모듈*: `[Ĵ^rgb, z_j^depth, c_j^vis, c_j^depth]` → 작은 MLP/GCN로 residual 출력. ([KeypointFusion AAAI'24](https://ojs.aaai.org/index.php/AAAI/article/view/28166), [Pyramid Deep Fusion](https://arxiv.org/html/2307.06038))

**함정 & bad-depth 거부:**
- **가려진 관절**(손바닥 뒤 손가락): 샘플 depth는 가림 표면 → `|z_j^depth − z_j^rgb| > τ_occ`면 거부, RGB/전역 fit으로 폴백.
- **Flying pixel(실루엣 경계)**: 얇은 손가락은 거의 전부 경계 → "confident but wrong" depth. ToF amplitude threshold + 이웃 median/variance + **손 마스크 몇 px erode 후 샘플링**.
- **권장 계층 설계:** (1)robust 샘플링+validity mask → (2)occlusion/flying-pixel 거부 → (3)RANSAC scale+root-depth alignment(백본 metric anchor) → (4)신뢰 가능 관절만 confidence blend/LM/Kalman refine → (5)Z 채널 1-Euro 최종 jitter 제거. **learned fusion(⑥)은 RGB-D로 end-to-end 재학습 가능할 때만.**

### 2. 시간·좌표 동기화 및 캘리브레이션

**RGB↔Depth 시간 동기화:**
- **ARKit:** `capturedImage`(RGB)와 `sceneDepth`(ARDepthData)가 **동일 ARFrame·동일 timestamp·동일 시점** → 별도 클럭 reconcile 불필요. depth 최대 60fps, RGB와 aspect ratio 동일(256×192 vs 1920×1440). 주의: `capturedDepthData`는 **전면 TrueDepth**(별도 timestamp)이니 후면 LiDAR엔 `sceneDepth` 사용. ([ARFrame](https://developer.apple.com/documentation/arkit/arframe))
- **ARCore:** 동일 `Frame`에서 `acquireCameraImage()`+`acquireDepthImage16Bits()`. 단 **depth가 준비 안 되면 이전 프레임의 stale depth 반환** → 정밀 시 stale 여부 확인, `NotYetAvailableException` 처리. ([Depth guide](https://developers.google.com/ar/develop/java/depth/developer-guide))
- **비동기 추론 패턴:** AR 콜백에서 **timestamp+depth map+RGB를 한 단위로 스냅샷**해 추론 잡과 함께 운반 → 완료 시 "잡과 함께 온 depth" 사용(움직이는 세션 재조회 X).

**좌표 정렬/캘리브레이션:**
- **ARKit:** depth가 컬러 카메라 시점에 **사전 정합** → depth intrinsics = RGB intrinsics를 해상도비(예 7.5×)로 스케일. `displayTransform`은 **화면 오버레이용**(3D 수학용 intrinsics와 분리). 일부 iPad는 정합 오프셋·고해상 캡처 시 FOV 변화 주의. ([intrinsics](https://developer.apple.com/documentation/arkit/arcamera/intrinsics))
- **ARCore:** depth가 카메라 FOV의 **crop**(aspect ratio 다를 수 있음) → 단순 스케일 불가, `transformCoordinates2d()` 사용.
- **Standalone 단안 depth net의 결정적 장점:** depth를 **동일 RGB 이미지**에서 추론 → **픽셀 정렬 자동**(extrinsic 캘리브·롤링셔터·클럭 오프셋 없음). 트레이드오프: 단안 metric 오차 >12% vs HW 센서 ~1%(실내). **하이브리드**(단안 depth를 sparse LiDAR로 rescale)가 best-of-both. ([One-Shot Metric Alignment](https://arxiv.org/pdf/2506.17110), [DELTAR](https://arxiv.org/pdf/2209.13362))

### 3. 온디바이스 파이프라인 & 스레딩 (30 FPS)

- **Capture와 inference 디커플 + stale frame 드롭** 필수(NN은 33ms 예산 초과 빈번).
- **Producer–consumer 1-slot(latest-only) 버퍼:** 새 프레임이 버퍼 덮어씀 → 항상 최신 처리. **Android CameraX `STRATEGY_KEEP_ONLY_LATEST`** 기본 제공, **iOS `alwaysDiscardsLateVideoFrames=true`**. ([CameraX](https://developer.android.com/media/camera/camerax/analyze))
- **스테이지 병렬화:** LiDAR depth는 거의 무비용 버퍼 복사(ARFrame 동승). depth **net**은 별도 가속 큐(hand=NPU, depth=GPU) 또는 NPU 인터리브. 픽셀 정렬이면 전처리 텐서 공유 가능. **double-buffer** 출력.
- **드롭 정책:** 백로그 금지. depth net이 못 따라오면 **저빈도(격프레임) 실행 + 직전 depth 재사용**(관절 depth는 2D pose보다 느리게 변함). MediaPipe가 참조 구현. ([MediaPipe](https://arxiv.org/pdf/1906.08172))

### 4. 기존 MobRecon 모델 결합 (현 repo 계열)

- MobRecon은 단안 RGB → 경량 2D stack + depth-separable spiral-conv 3D decoder → joint/mesh, **root-relative/weak-perspective**(절대 metric 깊이 없음) = depth가 채울 정확히 그 gap. ([MobRecon](https://arxiv.org/abs/2112.02753))
- **Depth-correction 모듈은 엄격히 post-processing, 모델 불변:** ①hand model(RGB)→2D joint+relative mesh → ②depth-correction glue(관절별 depth 샘플 → scale/root-depth alignment → 충돌 관절 거부/refine) → ③metric world-anchored joint/mesh. **exported 모델(TFLite/CoreML/ONNX) 재학습·수정 불필요** → depth 소스 교체·튜닝 자유.
- **Export 현실:** PyTorch→ONNX→TF→TFLite(직접 PyTorch→TFLite 경로 없음) 또는 PyTorch→CoreML(ANE). **spiral-conv 커스텀 op 변환 커버리지 조기 검증** 필요(현 repo가 이미 onnx/tflite 변환 작업 중).
- **Native vs Cross-platform:** Native(Swift/Kotlin) 최저 지연·zero-copy·ANE/NNAPI 직접 접근(권장). Unity AR Foundation은 `AROcclusionManager`로 ARKit/ARCore depth 단일 API 추상화(크로스플랫폼, marshaling 비용). Flutter는 30fps depth-fusion 루프엔 부적합(버퍼 복사 多).

### 5. ARKit/ARCore 세션 공존

- **두 카메라 동시 소유 불가 → AR이 카메라 소유, 프레임만 차용.**
- **iOS:** `ARWorldTrackingConfiguration`(`.sceneDepth`) → `session(_:didUpdate:)`에서 `frame.capturedImage`를 hand model에 투입 + 동일 프레임 `frame.sceneDepth` 융합. (ARKit+Vision hand pose 앱의 확립된 패턴) ([WWDC20 Vision](https://developer.apple.com/videos/play/wwdc2020/10653/))
- **Android:** ARCore Session → `acquireCameraImage()`를 모델에 투입 + `acquireDepthImage16Bits()` 융합. 토글 필요 시만 `SharedCamera`. **acquire한 이미지 반드시 close**(maxImages=16 초과 시 IllegalStateException). ([ARCore ML input](https://developers.google.cn/ar/develop/java/machine-learning?hl=en))
- **"depth만을 위한 AR" 비용:** ARCore는 기본 VGA CPU 스트림이면 `acquireCameraImage()` **실질 무비용**(다른 해상도 요청 시 추가 스트림 비쌈). ARKit은 LiDAR depth만 원해도 **VIO 월드트래킹 등 전체 비용** 지불 → 매 프레임 소비 시 발열·배터리, 소비를 throttle. 월드 앵커링 불필요하면 **standalone 단안 depth net이 AR 세션 오버헤드 회피** → 비-LiDAR/배터리 민감 기기에 유리.

**통합 패턴 출처 신뢰도:** Apple/Google 공식 문서 + peer-reviewed fusion 논문 기반 high confidence. fusion 정식화(§1)는 hand/body RGB-D 문헌에서 검증.

<!-- Content will be appended sequentially through research workflow steps -->
