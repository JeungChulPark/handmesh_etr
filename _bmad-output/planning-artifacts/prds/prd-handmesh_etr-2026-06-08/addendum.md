# Addendum — Depth-Aware Mobile Hand Pose Estimation

PRD 본문이 capability에 집중하기 위해 분리한 기술적 깊이. 다운스트림 아키텍처/솔루션 설계의 입력. 출처는 동일 세션 기술 연구 문서(`technical-mobile-depth-aware-hand-pose-estimation-system-research-2026-06-08.md`).

## A. Depth Source 모델 후보 (FR-3)

**Android / 비-Pro iPhone — Monocular Depth (metric 모델 필수):**
- **Depth Anything V2 Small + metric 파인튜닝** — Apache-2.0, 24.7M, ~34ms iPhone15Pro NE, TFLite/CoreML/ONNX export. **상업·모바일·metric 1순위 후보.** ⚠️ **metric 모델을 사용**할 것 — relative 모델을 손 관절로 스케일하는 방식은 순환 스케일링이므로 Depth Source로 금지(§B-0).
- 대안: FastDepth(MIT, NYU 실내 스케일), LMDepth(INT8 2.63MB). MiDaS-small은 relative-only라 **Depth Source 부적합**(prior 용도만).
- ⚠️ UniDepth/Metric3D는 정확도 높으나 서버급/비상업 라이선스 — 모바일 부적합.

**iPhone Pro — LiDAR Depth:**
- ARKit `ARFrame.sceneDepth`(ARDepthData: depthMap 256×192 Float32 meters + confidenceMap). 최대 60Hz, 컴퓨트 ~0.
- `smoothedSceneDepth`는 정지/저속에만(고속 lag). intrinsics를 256×192로 스케일 후 언프로젝트.

**라이선스 게이트:** 상용 배포 → Apache-2.0/MIT 후보만. WiLoR(NC-ND), UniDepth(NC) 등 배제.

## B-0. 순환 스케일링 방지 (핵심 설계 원칙)

Monocular relative depth를 손 관절 위치로 scale-align해 metric화하면, 그 depth는 RGB가 이미 아는 손 스케일을 재사용할 뿐 **새 절대 Z 정보를 더하지 못한다**(순환). 따라서 절대 깊이는 반드시 *외부 신호*(LiDAR 또는 **metric 단안 모델**)에서 와야 한다. §B의 scale/shift 정렬은 그 metric 신호의 잔여 bias 제거일 뿐이다.

## B. Fusion 알고리즘 (FR-4, FR-5, FR-8) — 권장 계층 설계

1. **Robust 샘플링:** 이웃 윈도 median + 신뢰도 마스킹, bilinear 금지(불연속 가짜 깊이).
2. **불량 거부:** occlusion test `|z_depth − z_rgb| > τ_occ` → 거부; flying-pixel은 ToF amplitude threshold + 마스크 erode.
3. **전역 Root-Depth Alignment (v1 백본):** 정면 손에서 scale·shift 동시추정은 공선성으로 불안정 → **v1은 본 길이로 scale 고정, shift-only**: `t* = argmin_t Σ w_j(ẑ_j·s_bone + t − z_j^depth)²`(closed-form), **RANSAC+MAD**로 outlier(가린 손가락) 방어. RootNet-style. (full scale-and-shift는 v2 옵션)
4. **관절별 fusion(선택):** `z_fused = w·z_depth+(1−w)·z_rgb`, `w=c_vis·c_depth` 또는 inverse-variance `σ_rgb²/(σ_rgb²+σ_depth²)`. 가림 관절 제외.
5. **본 길이 제약:** `E_bone=Σ(‖J_a−J_b‖−ℓ_ab)²` 위반 보정 거부. 필요 시 LM으로 MANO fit.
6. **회귀 게이트(FR-8):** 정렬 실패/유효 depth 부재/제약 위반 시 RGB-only 폴백, 관절별 "보정 적용" 플래그.

**v1 권장:** 3번(shift-only 전역 정렬)을 백본으로 우선 구현 → 효과/난이도 최적. 4·5는 신뢰 관절 한정 추가(§Open Q6).
**unproject 전제:** 모든 단계는 카메라 intrinsics를 요구하며, Recon crop/resize 시 intrinsics를 함께 변환 추적해야 systematic Z 오차를 막는다.

## C. Temporal Filtering (FR-6)

- **1-Euro filter**(XR 손추적 사실상 표준): 속도 적응 `fc=fcmin+β·|ẋ|`. 관절·축별, Z 채널 집중. β↑(고속 lag 해소), fcmin↓(저속 jitter 해소).
- 대안: per-joint Kalman, 측정 공분산 R_t를 depth 신뢰도로 설정(가림 시 예측 coast).

## D. 동기화·정렬 (FR-3, 통합)

- **ARKit:** RGB+depth 동일 ARFrame → timestamp·시점 자동 정렬. depth intrinsics = RGB intrinsics × 해상도비.
- **ARCore:** depth가 FOV crop + stale fallback 가능 → `transformCoordinates2d()`, `NotYetAvailableException` 처리.
- **Monocular net 장점:** 동일 RGB 추론 → 픽셀 정렬 자동, extrinsic 캘리브 불필요.
- **비동기 추론:** AR 콜백에서 timestamp+depth+RGB 한 단위 스냅샷, 추론 잡과 함께 운반.

## E. 온디바이스 파이프라인 (NFR-1)

- Producer-consumer latest-only 버퍼(Android CameraX KEEP_ONLY_LATEST / iOS alwaysDiscardsLateVideoFrames). 백로그 금지.
- 스테이지 병렬: hand=NPU, monocular depth=GPU 유닛 분리(또는 시간 stagger), double-buffer 출력. depth net이 못 따라오면 격프레임 실행+직전 depth 재사용.
- 런타임: Android LiteRT-QNN/ORT-QNN(NNAPI deprecated), iOS CoreML+ANE.
- **지연 주의(모순 해소):** depth 모델 지연은 모델·정밀도·HW에 크게 의존 — MiDaS급 INT8은 ~1ms대(저정확), DAv2-Small metric은 ~30ms대(고정확, §A). **metric depth를 매 프레임 돌리면 30FPS 예산을 거의 소진**하므로 **비대칭 레이트**(hand 매 프레임, depth N프레임마다 + 직전 재사용)가 사실상 필수. hand ~1–3ms.
- LiDAR 경로는 depth 모델 불필요 → NPU 전체를 hand에 할애(최저전력·최고 FPS).

## F. 세션 공존 비용 (통합)

- AR이 카메라 소유, 프레임 차용(iOS `frame.capturedImage`→Recon / Android `acquireCameraImage()`). Android 이미지 반드시 close(maxImages=16).
- ARKit은 depth만 원해도 VIO 전체 비용 → 소비 throttle. 월드앵커 불필요하면 standalone monocular depth가 AR 오버헤드 회피.

## G. 평가 방법 (SM) — 필수 조건

- **데이터셋:** **camera-space, non-PA 절대 3D GT**가 있는 셋(DexYCB/HO3D 등). ⚠️ FreiHAND류 PA·root-relative GT만으로는 절대 Z를 측정할 수 없으므로 1차 지표로 부적합(보조 형상 지표로만).
- **GT depth 입력 금지:** 평가 루프는 **런타임 depth 소스(LiDAR/단안 모델 추론 결과)를 in-loop**로 사용한다. GT depth를 보정 입력으로 쓰면 단안 오차가 빠져 SM-1이 허위로 좋아진다.
- **플랫폼 분리:** LiDAR 경로와 metric 단안 경로의 지표를 **분리 보고**(평균 합산 금지 — 정확도 절벽 은폐 방지).
- **Phase-0 연계:** §6.0 G-1(베이스라인)·G-2(LiDAR 이득)·G-3(Android 이득)이 이 프로토콜로 측정되어 SM 절대 목표·Android 등급을 확정.
- **온디바이스:** FPS·메모리·발열 실측(정확도 GT는 모바일에서 확보 곤란).

## H. v2 후보 (이연)

- Unity 시각화 렌더링(원 FR-7), XR 앱 인터랙션(원 FR-8).
- 하이브리드 고정밀: PromptDA류(monocular + LiDAR prompt) metric depth.
- 양손/손-객체, learned fusion(RGB-D end-to-end 재학습 시).
