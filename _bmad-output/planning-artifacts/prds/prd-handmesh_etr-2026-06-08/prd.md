---
title: Depth-Aware Mobile Hand Pose Estimation
status: final
created: 2026-06-08
updated: 2026-06-08
---

# PRD: Depth-Aware Mobile Hand Pose Estimation
*Working title — confirm.*

## 0. Document Purpose

이 PRD는 PM·아키텍트·다운스트림 워크플로(아키텍처 설계, 에픽/스토리 분해)를 위한 것이다. 기존 RGB 기반 Recon Hand Tracking의 3D 관절 정확도(특히 Z축·손가락 끝)를 Depth 정보로 보정하는 **Depth Refinement Module**의 요구사항을 정의한다. 용어는 §3 Glossary에 고정하고, 기능은 Feature로 묶어 FR을 전역 번호로 중첩하며, 추론한 부분은 `[ASSUMPTION]` 태그로 인라인 표기하고 §9에 색인한다.

본 PRD는 동일 세션에서 작성된 기술 연구 문서를 1차 입력으로 삼는다: `_bmad-output/planning-artifacts/research/technical-mobile-depth-aware-hand-pose-estimation-system-research-2026-06-08.md` (RGB-only depth 모델, hand pose 프레임워크, LiDAR/ARCore depth API, 모바일 런타임, RGB-D fusion 알고리즘, 온디바이스 통합 패턴). 구체 기술 메커니즘 결정·모델 후보는 PRD 본문이 아니라 `addendum.md`에 보존한다.

원 입력의 FR-1~8과 본 PRD의 전역 FR 번호 매핑은 §부록 A에 표로 정리한다.

## 1. Vision

스마트폰에서 RGB 영상만으로 손을 추적하면 3D 관절의 **절대 깊이(Z)** 가 본질적으로 모호하다 — 작은 손이 가까이 있는 경우와 큰 손이 멀리 있는 경우가 동일하게 투영된다. 그 결과 현재 시스템은 실제 손 위치와 추정 위치의 차이, 손가락 끝 오차, 특히 Z축 방향 오차를 보인다. XR에서 이는 가상 객체가 손과 어긋나 보이는 정렬 실패로 직결된다.

Depth-Aware Mobile Hand Pose Estimation은 **기존 Recon Hand Tracking을 교체하지 않고**, 그 출력에 **플랫폼 적응형 Depth 보정 단계**를 후처리로 덧붙인다. Android에서는 단안(monocular) Depth 추정으로, iPhone에서는 LiDAR Depth로 metric 깊이를 얻어, 각 관절의 Z를 견고하게 앵커링하고 시간적으로 안정화한다. 모델 자체는 재학습하지 않으므로 기존 자산을 보존하면서 정확도만 끌어올린다.

이 모듈은 **Unity AR Foundation 패키지**로 전달되어 Android·iOS에 단일 통합 표면을 제공하며, 30 FPS 이상·메모리 1GB 이하·엔드투엔드 지연 50ms 미만이라는 모바일 실시간 제약 안에서 동작한다. 성공하면 RGB-only 대비 MPJPE·Z-error·jitter를 측정 가능하게 줄여, 연구 검증과 제품 XR 정렬 양쪽을 만족시킨다.

## 2. Target User

### 2.1 Jobs To Be Done

- **XR 사용자로서**, 가상 객체가 내 손과 정확히 정렬되기를 원한다 — 손가락 끝과 Z 깊이가 어긋나면 몰입이 깨진다.
- **모바일 앱/콘텐츠 개발자로서**, 플랫폼별 depth 차이(Android 단안 vs iPhone LiDAR)를 직접 다루지 않고 **단일 Unity AR Foundation API**로 보정된 3D 관절을 받고 싶다.
- **연구자로서**, 동일 파이프라인이 Android·iOS 양쪽에서 동작하고, RGB-only 대비 정확도 개선을 **오프라인 벤치마크로 정량 입증**할 수 있기를 원한다.
- **제품 통합자로서**, 30 FPS·1GB·50ms 예산 안에서 기존 Recon 모델을 건드리지 않고 보정 모듈만 끼워 넣고 싶다.

### 2.2 Non-Users (v1)

- 데스크톱/서버 GPU 환경의 고정밀 오프라인 손 재구성 사용자 — 본 제품은 모바일 실시간 전용이다.
- 전면 카메라(TrueDepth) 기반 셀피 거리 손 추적 — v1은 후면 카메라·환경 depth 경로를 가정한다. `[ASSUMPTION]`
- LiDAR 미탑재 비-Pro iPhone에서 **하드웨어 depth** 정확도를 기대하는 사용자 — 해당 기기는 Android와 동일한 단안 depth 경로로 폴백한다(§Platform).
- 양손 상호작용/손-객체 상호작용의 정밀 접촉 추정 사용자 — v1은 단일 손 관절 보정에 집중한다. `[ASSUMPTION]`

### 2.3 Key User Journeys

- **UJ-1. Jin이 AR 앱에서 가상 반지를 손가락에 끼운다.**
  - **Persona + context:** Jin은 LiDAR 탑재 iPhone 16 Pro로 AR 쇼핑 앱을 쓴다. RGB-only 시절엔 반지가 손가락 끝에서 몇 cm 떠 보였다.
  - **Entry state:** 앱 실행, AR 세션 활성, 손을 카메라 앞 ~30–50cm에 둠.
  - **Path:** 손을 들어올림 → Recon이 2D/3D 관절 검출 → LiDAR sceneDepth에서 관절별 깊이 샘플 → 보정 모듈이 metric Z로 재앵커링·시간 안정화.
  - **Climax:** 가상 반지가 실제 손가락 마디에 밀착되어 손을 돌려도 따라온다.
  - **Resolution:** Jin은 깊이 어긋남 없이 자연스러운 정렬을 경험한다. **Edge case:** 손가락이 손바닥 뒤로 가려지면 해당 관절은 depth 샘플을 거부하고 RGB 예측+본 길이 제약으로 폴백한다.

- **UJ-2. Mina(개발자)가 Unity로 크로스플랫폼 손 인터랙션을 붙인다.**
  - Mina는 Unity AR Foundation 프로젝트에 본 패키지를 임포트하고, Android·iOS 빌드 모두에서 **동일한 보정 3D 관절 스트림 API**를 구독한다. 플랫폼별 depth 소스 분기는 패키지 내부가 처리하므로 그녀는 한 줄의 분기도 작성하지 않는다. *(Scope: 통합 표면 — 시각화/XR 인터랙션 구현은 그녀의 몫이며 v1 범위 밖)*

- **UJ-3. Dr. Park(연구자)이 개선을 정량 입증한다.**
  - GT가 있는 RGB-D 데이터셋에서 refinement 전/후 MPJPE·Z-error·jitter를 측정해 RGB-only 베이스라인 대비 상대 개선률을 보고하고, 온디바이스에서 FPS·메모리를 실측한다.

## 3. Glossary

- **Recon Hand Tracking (Recon)** — 기존 RGB 기반 경량 손 메시/포즈 모델(현 repo의 MobRecon 계열). 단안 RGB 한 장에서 2D 관절과 **root-relative(절대 스케일 없음) 3D 관절/메시**를 출력한다. 본 제품이 입력으로 사용하며 **재학습·수정하지 않는다.**
- **Hand Joint (Joint)** — 손의 21개 관절점. 각 관절은 2D 픽셀 위치, root-relative 3D 좌표, 검출 신뢰도(confidence)를 가진다.
- **Depth Map** — 픽셀별 깊이 값을 담은 2D 맵. Android는 단안 추정, iPhone은 LiDAR로 획득. **metric(미터 단위)** 을 목표로 한다.
- **Depth Source** — 플랫폼별 Depth Map 공급 경로. {Android·비-Pro iPhone: Monocular Depth, LiDAR iPhone: LiDAR Depth} 중 하나.
- **Monocular Depth** — RGB 한 장에서 신경망으로 추정한 Depth Map. 동일 RGB에서 추론되어 **픽셀 정렬**이 보장되나 metric 정확도가 낮다.
- **LiDAR Depth** — LiDAR 탑재 iPhone의 ARKit `sceneDepth` 기반 Depth Map. 256×192·최대 60Hz·metric. 컴퓨트 비용 거의 없음.
- **Depth Refinement** — Recon의 root-relative 3D 관절을 Depth Source로 보정해 metric 3D 관절을 산출하는 후처리 단계. 본 제품의 핵심.
- **Refined Joint** — Depth Refinement를 거친 metric 3D 관절. 본 제품의 주 출력.
- **Root-Depth Alignment** — 견고한 관절(손목/손바닥)의 샘플 깊이로 root-relative 포즈 전체의 metric scale·root translation을 정하는 정렬.
- **Joint Confidence** — 관절별 신뢰도. Recon 히트맵 신뢰도(c_vis)와 Depth 신뢰도(c_depth)의 결합으로 fusion 가중에 사용.
- **Temporal Filtering** — 프레임 간 관절(특히 Z)의 jitter를 줄이는 시간 평활화.
- **Refinement Module** — FR-1~8을 포괄하는 본 제품 단위. Unity AR Foundation 패키지로 전달.
- **End-to-End Latency** — 카메라 프레임 캡처부터 Refined Joint 출력까지의 지연(파이프라이닝으로 1프레임 초과 가능).
- **RGB-only Baseline** — Depth Refinement를 적용하지 않은 현행 Recon 출력. 모든 성공 지표의 비교 기준.

## 4. Features

> FR은 전역 번호(FR-1~FR-7)로 중첩한다. 각 FR의 "Consequences (testable)"가 §부록 B Acceptance Criteria의 근거가 된다. 구체 알고리즘·모델 선택은 `addendum.md` 참조.

### 4.1 RGB 입력 처리 및 손 관절 검출 (기존 Recon)

**Description:** 후면 카메라 RGB 프레임을 받아 전처리하고, 기존 Recon 모델로 손 관절을 검출한다. 이 단계는 대부분 기존 자산이며, 본 제품은 그 출력(2D 관절 + root-relative 3D 관절 + 신뢰도)을 **변경 없이** 소비한다. Realizes UJ-1, UJ-2.

**Functional Requirements:**

#### FR-1: RGB 프레임 수집 및 전처리

시스템은 AR 세션의 후면 카메라에서 RGB 프레임을 받아 Recon 입력 규격으로 전처리한다. Realizes UJ-1.

**Consequences (testable):**
- 카메라/AR 세션이 공급하는 프레임을 Recon 입력 해상도·정규화 규격으로 변환해 전달한다.
- 최신 프레임만 처리하고 백로그가 쌓이면 stale 프레임을 폐기한다(latest-only).
- 처리 프레임의 timestamp, **카메라 intrinsics**, (해당하면) Depth Map을 한 단위로 묶어 비동기 추론 잡과 함께 운반한다(완료 시 "잡과 함께 온" intrinsics/depth 사용).
- Recon이 입력을 crop/resize하면 **intrinsics를 그 변환에 맞게 갱신해 추적**한다(이후 unprojection의 systematic Z 오차 방지).

#### FR-2: 손 관절 검출 (Recon, 불변)

시스템은 전처리된 RGB로 기존 Recon 모델을 실행해 손 관절을 검출한다. Recon 모델은 재학습·수정하지 않는다. Realizes UJ-1, UJ-2.

**Consequences (testable):**
- 프레임당 21개 관절의 2D 위치, root-relative 3D 좌표, Joint Confidence를 출력한다.
- 검출된 손의 **좌/우(handedness)** 를 식별해 다운스트림에 전달한다(본 길이·MANO 제약과 정렬을 위해 필수).
- 손 미검출 시 보정 단계를 건너뛰고 빈 결과 또는 직전 유효 결과 정책을 따른다. `[ASSUMPTION: 미검출 시 빈 결과 반환]`
- Recon 모델 가중치/구조는 본 모듈에 의해 변경되지 않는다(블랙박스 호출).

**Notes:** `[NOTE FOR PM]` 본 모듈의 confidence-weighted fusion(FR-5)은 Recon이 **per-joint Joint Confidence를 실제로 노출**한다는 전제에 의존한다. 노출하지 않으면 fusion 가중을 2D 가시성/재투영 잔차로 대체해야 한다 → §Open Q8.

**Out of Scope:** Recon 모델 자체의 정확도 개선·재학습.

### 4.2 Depth 획득 (플랫폼 적응형)

**Description:** 플랫폼에 맞는 Depth Source에서 **metric** Depth Map을 획득한다. LiDAR 탑재 iPhone은 ARKit `sceneDepth`(LiDAR Depth, 하드웨어 metric), 그 외(Android·비-Pro iPhone)는 **metric을 직접 출력하는 단안 모델**(예: Depth Anything V2 metric finetune)을 사용한다. 두 경로는 Unity AR Foundation의 occlusion/depth 추상화 또는 네이티브 등가 경로로 노출되며, 상위 보정 로직은 동일 인터페이스로 Depth Map을 소비한다. Realizes UJ-1, UJ-2.

> **설계 원칙(순환 스케일링 방지):** Monocular 경로는 **metric 모델**을 사용해 절대 깊이를 *외부 신호*로 들여온다. relative depth를 손 관절로 스케일해 metric화하는 방식은 **금지** — 그것은 RGB가 이미 가진 정보로 RGB를 보정하는 순환이며 새 절대 Z를 더하지 못한다. §FR-5의 scale/shift 정렬은 metric 신호의 잔여 bias 보정일 뿐, 절대 깊이의 생성원이 아니다.

**Functional Requirements:**

#### FR-3: 플랫폼 적응형 metric Depth Map 획득

시스템은 실행 기기의 Depth Source를 자동 선택해 처리 중인 RGB 프레임과 정렬된 metric Depth Map을 제공한다. Realizes UJ-1.

**Consequences (testable):**
- LiDAR 탑재 기기에서는 LiDAR Depth(metric, 신뢰도 맵 포함)를 획득한다.
- LiDAR 미탑재 기기에서는 **metric 단안 모델**로 폴백한다(relative-only 모델은 Depth Source로 사용하지 않는다).
- 각 Depth Map은 RGB 관절 좌표계로 매핑 가능한 정렬 정보와 함께 제공된다. **iOS**: intrinsics를 depth 해상도(256×192)로 스케일. **Android(ARCore)**: depth가 카메라 FOV의 **crop**이라 단순 해상도 스케일이 불가하며 `transformCoordinates2d()` 등으로 좌표 변환한다.
- Depth Map은 처리 중인 RGB 프레임과 동일 timestamp(ARKit) 또는 정렬 가능한 최신 프레임을 사용한다(ARCore의 **stale depth 폴백을 명시적으로 감지·처리**).

**Feature-specific NFRs:** Depth 획득은 30 FPS 예산을 위협하지 않아야 한다. LiDAR는 사실상 무비용. metric 단안 모델은 단독으로도 프레임 예산의 상당부를 소비하므로 hand는 매 프레임, **depth는 N프레임마다 실행하고 직전 depth를 재사용**하는 비대칭 레이트를 허용한다(§NFR-1).

**Notes:** `[NOTE FOR PM]` 비-Pro iPhone을 LiDAR가 아닌 metric 단안으로 폴백 처리하는 정책 확정 필요(§Open Q3). Android 단안 metric depth가 실제로 절대 Z를 줄이는지는 **Phase-0 Feasibility Spike(§6.0)에서 게이팅**한다.

### 4.3 Depth 기반 3D 관절 보정

**Description:** Recon의 root-relative 3D 관절과 metric Depth Map을 결합해 metric Refined Joint를 산출한다. 핵심은 관절별 깊이 샘플링 → 불량 샘플 거부 → 전역 Root-Depth Alignment → (신뢰 가능한 관절에 한해) 보정·제약이다. 이 단계가 본 제품의 가치 핵심이며 Z축·손가락 끝 오차를 직접 겨냥한다. Realizes UJ-1, UJ-3.

**Functional Requirements:**

#### FR-4: 관절별 Depth 샘플링 및 불량 샘플 거부

시스템은 각 관절의 2D 위치에서 Depth Map을 견고하게 샘플링하고, 가림·경계 노이즈로 인한 불량 샘플을 거부한다. Realizes UJ-1.

**Consequences (testable):**
- 단일 픽셀이 아니라 이웃 윈도의 robust 통계(예: median)와 신뢰도 마스킹으로 관절 깊이를 추정한다.
- 가려진 관절(샘플 깊이가 RGB 예측과 임계 이상 불일치)은 무효 처리하고 폴백 경로로 넘긴다.
- 실루엣 경계의 flying-pixel/저신뢰 샘플을 거부한다.
- 유효 깊이가 없는 관절은 다운스트림이 식별 가능하도록 표시된다.

#### FR-5: Metric 3D 관절 재구성 (Root-Depth Alignment + 보정)

시스템은 유효한 관절 깊이와 **카메라 intrinsics**로 root-relative 포즈를 metric camera-space로 재구성하고, 신뢰 가능한 관절에 한해 깊이 보정·해부학적 제약을 적용해 Refined Joint를 산출한다. Realizes UJ-1, UJ-3.

**Consequences (testable):**
- 2D 관절 + 샘플 깊이 + intrinsics로 metric 3D를 unproject한다(intrinsics 부재 시 보정 불가로 폴백).
- **v1 백본은 shift-only root-depth 정렬**: 본 길이로 전역 scale을 고정하고, 견고한 관절(손목/손바닥-MCP) 기준 root-depth(shift)를 추정해 모든 관절을 metric으로 앵커링한다(유효 깊이 없는 관절 포함). 정면 손에서 scale·shift 동시 추정의 공선성 불안정을 회피한다.
- 전역 정렬은 outlier(가린 손가락 등)에 견고하다(RANSAC 등 robust fitting).
- 유효 깊이가 있는 관절에 한해 Joint Confidence 가중으로 Depth와 RGB 깊이를 융합한다(가림 관절은 융합에서 제외).
- 본 길이(해부학) 제약을 위반하는 보정은 거부·완화된다.

**Notes:** 구체 정렬·융합 알고리즘(RootNet-style shift-only, RANSAC, confidence/inverse-variance fusion, bone-length term)은 addendum.md.

#### FR-8: Refinement 채택/기각 런타임 게이트

시스템은 프레임·관절 단위로 보정 결과의 타당성을 검사해, RGB-only Baseline보다 나빠질 위험이 있으면 보정을 기각하고 RGB-only(또는 직전 유효) 값을 내보낸다. Realizes UJ-1.

**Consequences (testable):**
- 유효 깊이 샘플이 없거나 전역 정렬이 실패한 프레임은 보정을 적용하지 않고 RGB-only를 출력한다.
- 본 길이/시간 연속성 제약을 크게 위반하는 보정 관절은 기각된다.
- 각 출력 관절에 "보정 적용 여부" 플래그를 부착해 SM-C2(가림 구간 왜곡 금지)·NFR-6(회귀 금지)을 런타임에서 강제·측정 가능하게 한다.

### 4.4 시간적 필터링

**Description:** 프레임 간 Refined Joint(특히 Z 채널)의 jitter를 줄여 XR에서 떨림 없는 안정적 정렬을 제공한다. Realizes UJ-1.

**Functional Requirements:**

#### FR-6: 관절별 Temporal Filtering

시스템은 Refined Joint를 프레임 간 시간 평활화하되, 빠른 손 움직임의 지연(lag)을 최소화한다. Realizes UJ-1.

**Consequences (testable):**
- 정지/저속에서 jitter를 측정 가능하게 감소시킨다(§SM).
- 빠른 움직임에서 과도한 lag을 유발하지 않는다(속도 적응형 평활화).
- 관절·축별로 적용되며 Z 채널에 특히 작용한다.
- 필터 비활성/활성 출력을 모두 노출해 평가·튜닝이 가능하다. `[ASSUMPTION]`

### 4.5 Unity AR Foundation 패키지 및 출력 API (전달 표면)

**Description:** Refinement Module을 Unity AR Foundation 패키지로 전달하고, 보정된 metric 3D 관절을 단일 크로스플랫폼 API로 노출한다. 풍부한 시각화와 XR 인터랙션 구현은 v1 범위 밖(소비 앱의 몫)이며, v1은 데이터 출력 표면까지를 제공한다. Realizes UJ-2.

**Functional Requirements:**

#### FR-7: 보정 관절 출력 API (Unity AR Foundation)

시스템은 Android·iOS 모두에서 동일한 인터페이스로 프레임별 Refined Joint 스트림을 Unity 앱에 제공한다. Realizes UJ-2.

**Consequences (testable):**
- 단일 API 호출로 21개 Refined Joint(metric 3D, 관절별 유효성/신뢰도 포함)와 프레임 timestamp를 구독할 수 있다.
- 플랫폼별 Depth Source 분기는 패키지 내부에 캡슐화되어 소비 코드에 노출되지 않는다.
- Android·iOS 빌드에서 동일 API 시그니처로 동작한다(AR Foundation 추상화 준수).
- 패키지는 Unity AR Foundation 프로젝트에 표준 임포트 절차로 통합된다.

**Out of Scope (v1):** 메시/관절 렌더링·시각화(원 FR-7), XR 앱 레벨 인터랙션·이벤트(원 FR-8) → v2.

## 5. Non-Goals (Explicit)

- Recon 모델 자체의 재학습·구조 변경 — 본 제품은 후처리 보정만 추가한다.
- 풍부한 Unity 시각화 및 XR 앱 인터랙션 구현(원 FR-7/FR-8) — v1은 보정 데이터 출력 표면까지. v2로 이연.
- 양손·손/객체 상호작용의 접촉 수준 정밀 추정. `[ASSUMPTION]`
- 데스크톱/서버 GPU 고정밀 오프라인 재구성.
- 전면 TrueDepth 기반 근거리 손 추적 경로. `[ASSUMPTION]`
- 비-LiDAR 기기에서 하드웨어급 metric depth 정확도 보장 — 해당 기기는 Monocular 경로의 정확도 한계를 따른다.
- 손 검출/추적 자체의 신규 구현 — 기존 Recon에 의존.

## 6. MVP Scope

### 6.0 Phase-0 Feasibility Spike (HARD GATE)

본 구현 착수 **전에 반드시 통과해야 하는 검증 게이트.** 본 PRD의 정확도 목표(SM-1~3, §7)는 이 결과로 확정된다.

- **G-1 베이스라인 측정:** RGB-only Baseline의 MPJPE·Z-error·jitter를 오프라인 GT 데이터셋(camera-space, non-PA)에서 측정해 §Open Q2를 해소한다.
- **G-2 LiDAR 경로 이득:** iPhone LiDAR depth로 refinement 적용 시 Z-error/MPJPE 감소를 측정(런타임 depth 소스를 in-loop로 사용, GT depth를 입력으로 쓰지 않음).
- **G-3 Android 단안 이득(게이팅):** metric 단안 depth로 refinement가 RGB-only 대비 **유의미하게 Z-error를 줄이는지** 측정. 입증 시 Android를 iPhone과 **동등 등급**으로, 미달 시 **experimental 등급**으로 강등한다(§9).
- **게이트 판정:** G-1·G-2 통과 + G-3 결과로 Android 등급 결정 후에만 전체 FR 구현 착수. SM 절대 수치는 이 시점에 [ASSUMPTION]에서 확정값으로 대체한다.

### 6.1 In Scope

- 핵심 보정 파이프라인 FR-1~6 + **FR-8(채택/기각 게이트)**: RGB 입력 → Recon 관절 검출(handedness 포함) → 플랫폼 적응형 metric Depth 획득 → Depth 샘플링·거부 → Metric 재구성(shift-only Root-Depth Alignment) → Temporal Filtering → 런타임 회귀 게이트.
- 플랫폼 적응형 Depth Source: Android(+비-Pro iPhone) metric 단안, LiDAR iPhone LiDAR.
- 최소 출력 표면 FR-7: Unity AR Foundation 패키지의 Refined Joint 출력 API(보정 적용 플래그 포함).
- camera-space(non-PA) 오프라인 벤치마크 기반 정확도 평가 하니스(플랫폼별 분리) + 온디바이스 FPS/메모리 측정.

### 6.2 Out of Scope for MVP

- Unity 시각화 렌더링(원 FR-7) — v2. `[NOTE FOR PM]` 데모 임팩트가 큰 항목이라 일정 여유 시 우선 재검토.
- XR 앱 레벨 통합·인터랙션(원 FR-8) — v2.
- 양손/손-객체 상호작용 — v2+.
- 전면 카메라 경로, 하이브리드(LiDAR+Monocular 융합) 고정밀 모드 — v2+. (연구 문서가 PromptDA류 하이브리드를 후보로 식별)

## 7. Success Metrics

> 모든 정확도 지표는 **RGB-only Baseline** 대비 **상대 개선률**로 정의하고, **플랫폼별(LiDAR iPhone vs metric 단안)로 분리 보고**한다(평균 하나로 합치지 않는다 — 정확도 절벽 은폐 방지). 절대 수치 목표는 `[ASSUMPTION]`이며 Phase-0(§6.0) 결과로 확정한다.
>
> **평가 방법(필수 조건):** ① **camera-space, non-PA(절대) 3D GT**가 있는 데이터셋(DexYCB/HO3D 등) 사용 — FreiHAND류 PA/root-relative GT만으로는 절대 Z를 측정할 수 없다. ② 평가 루프에 **실제 런타임 depth 소스를 in-loop**로 사용하고 **GT depth를 입력으로 쓰지 않는다**(단안 오차가 빠져 지표가 허위로 좋아지는 것 방지). ③ 정확도는 오프라인, FPS/메모리/발열은 온디바이스 실측.

**Primary** *(플랫폼별 분리; iPhone-LiDAR이 1차 입증 경로, Android는 Phase-0 게이팅)*
- **SM-1: Z-error 감소** — 관절 깊이(Z) 평균 오차가 Baseline 대비 감소. 목표: **iPhone(LiDAR) ≥30%`[ASSUMPTION]`**, **Android(단안) ≥10%`[ASSUMPTION]`**(보수적, Phase-0 확정). Validates FR-4, FR-5.
- **SM-2: MPJPE 감소** — 절대(non-PA) MPJPE가 Baseline 대비 감소. 목표: iPhone **≥20%**, Android **≥8%**. `[ASSUMPTION]` Validates FR-5.
- **SM-3: Jitter 감소** — 정지/저속 시퀀스의 프레임 간 관절 변동(특히 Z)이 Baseline 대비 **≥40% 감소**(플랫폼 공통, Temporal Filtering 효과). `[ASSUMPTION: 40%]` Validates FR-6.

**Secondary**
- **SM-4: 처리량 유지** — 타깃 기기에서 **≥30 FPS** 유지. Validates NFR-1.
- **SM-5: 지연 예산** — End-to-End Latency **<50ms**. Validates NFR-1.
- **SM-6: 손가락 끝 오차 감소(간접)** — fingertip(5개)은 LiDAR 256×192에서 sub-pixel이라 **직접 depth 샘플로 보정하지 않는다.** root/palm 앵커 + 본 길이 제약의 **간접 효과**로 위치 오차가 Baseline 대비 악화되지 않고 가능한 경우 감소함을 측정. `[ASSUMPTION]` Validates FR-5, FR-8.

**Counter-metrics (do not optimize)**
- **SM-C1: 빠른 움직임 lag** — Temporal Filtering이 고속 구간 추적 lag을 악화시키지 않아야 한다(평활화 과적용 방지). Counterbalances SM-3.
- **SM-C2: 가림 구간 왜곡** — Depth 보정이 가려진 관절에서 RGB-only보다 큰 오차를 만들지 않아야 한다(FR-8 게이트로 강제). Counterbalances SM-1, SM-2.
- **SM-C3: 메모리/발열** — 정확도 향상을 위해 메모리 1GB·실시간·발열 예산을 초과하지 않아야 한다. Counterbalances SM-1~3.

## 8. Cross-Cutting NFRs

- **NFR-1: 실시간 성능** — 타깃 기기에서 **≥30 FPS** 처리량, **End-to-End Latency <50ms**. (30 FPS=33ms/frame 처리량과 50ms 파이프라인 지연을 구분.) metric 단안 경로는 hand=매 프레임, depth=N프레임마다 실행(직전 depth 재사용)하는 **비대칭 레이트**를 허용해 예산을 맞춘다. hand=NPU / depth=GPU 유닛 분리.
- **NFR-2: 메모리** — 앱+Recon 모델+Depth 경로+보정 모듈 합산 런타임 메모리 **<1GB**.
- **NFR-3: 모바일 디바이스 지원** — Snapdragon 8 Gen 시리즈 및 Apple A17/A18(이상) 급. 동시 추론(hand+monocular depth)은 NPU/GPU 유닛 분리로 예산 충족. `[ASSUMPTION: 구체 지원 기기 목록 미확정 — §Q1]`
- **NFR-4: Unity AR Foundation 지원** — AR Foundation의 카메라/occlusion·depth 추상화를 준수해 Android(ARCore)·iOS(ARKit) 단일 코드 경로로 동작.
- **NFR-5: 모델 불변성** — 기존 Recon 모델(TFLite/CoreML/ONNX export)을 재학습·수정하지 않고 블랙박스로 호출. (현 repo의 onnx/tflite 변환 작업과 정합)
- **NFR-6: 정확도 회귀 금지** — 어떤 보정 경로도 RGB-only Baseline 대비 주요 지표를 악화시키지 않는다(FR-8 런타임 게이트로 강제).
- **NFR-7: 전력·발열** — 지속 사용 시 thermal throttling으로 FPS가 무너지지 않아야 한다. ARKit를 depth 용도로 띄우면 VIO 월드트래킹 전체 비용이 들므로 프레임 소비를 throttle하고, 월드 앵커링 불필요 시 standalone 단안 경로로 AR 세션 오버헤드를 회피한다.

## 9. Platform & Hardware Constraints

- **타깃 폼팩터:** 후면 카메라 모바일(스마트폰), Unity AR Foundation 앱 내 동작.
- **Android:** ARCore + **metric 단안 depth**. ToF 센서는 가정하지 않음(현대 플래그십 대부분 미탑재). ARCore depth-from-motion은 "정지 카메라+움직이는 손"에서 시차 부족으로 취약하므로 **metric 단안 신경망 depth를 1차 경로**로 한다. `[ASSUMPTION]` **등급은 Phase-0(§6.0) G-3로 결정** — 이득 입증 시 iPhone과 동등, 미달 시 **experimental(best-effort) 등급**.
- **iOS:** LiDAR 탑재 Pro 기기 → ARKit `sceneDepth`(1차 입증 경로). 비-Pro iPhone → metric 단안 폴백.
- **공통 depth 한계:** 모바일 depth는 룸스케일(~0.5–5m·≤256×192)이라 <30cm 손가락을 분리 해상하지 못함 → 관절별 **Z prior**로 사용하고 단일 픽셀을 신뢰하지 않는다(FR-4).
- **메모리 footprint(참고, 연구 실측):** Monocular depth INT8 ~17MB, Recon hand 수십 MB 수준 → 1GB 예산은 여유.

## 10. Integration & Dependencies

- **상류 의존:** 기존 Recon Hand Tracking(현 repo MobRecon 계열) — 2D/3D 관절·신뢰도 출력. 모델 export(spiral-conv 등 커스텀 op) 변환 커버리지 조기 검증 필요.
- **플랫폼 SDK:** ARKit(sceneDepth/intrinsics), ARCore(Depth/Raw Depth), Unity AR Foundation(AROcclusionManager).
- **런타임:** TFLite/LiteRT(+QNN), CoreML(ANE), ONNX Runtime Mobile 중 플랫폼별 선택(아키텍처 단계 확정).
- **세션 공존:** AR 세션이 카메라를 소유하고 RGB 프레임을 차용해 Recon에 투입, 동일 프레임 depth를 융합(§연구 통합 패턴).
- **다운스트림 소비자:** Unity 앱(시각화/XR) — v1은 데이터 API까지 제공.

## 11. Open Questions

1. 구체 타깃 지원 기기 목록(Android 최소 사양, iOS 최소 버전)은? (NFR-3 확정용)
2. RGB-only Baseline의 현재 MPJPE/Z-error/jitter 수치는? → **Phase-0 G-1에서 측정·확정**(SM 절대 목표 고정).
3. 비-Pro iPhone 폴백 정책: metric 단안으로 일원화 vs LiDAR 기기 전용 기능 분기?
4. camera-space 절대 3D GT 데이터셋 확정(DexYCB/HO3D 등)과 평가 프로토콜 — **런타임 depth in-loop, GT depth 입력 금지** 확정.
5. metric 단안 모델 선택 시 라이선스 제약(상업 배포) 확인 — 상용 가능(Apache/MIT, 예: DAv2 Apache-2.0) 후보 우선.
6. v1 보정 범위: shift-only Root-Depth Alignment 백본만 vs 관절별 fusion까지? (난이도 대비 효과 — Phase-0 데이터로 판단)
7. 패키지 배포 채널(사내 UPM 레지스트리 / Git URL / 에셋)?
8. 기존 Recon이 **per-joint Joint Confidence를 실제로 노출**하는가? 미노출 시 fusion 가중을 2D 가시성/재투영 잔차로 대체(FR-2 전제 검증).

## 12. Assumptions Index

- §2.2 — v1은 후면 카메라·환경 depth 경로 가정(전면 TrueDepth 제외).
- §2.2 / §5 — 양손·손-객체 상호작용 v1 제외.
- §4.1 FR-2 — 손 미검출 시 빈 결과 반환.
- §4.2 FR-3 — 비-Pro iPhone은 metric 단안 폴백.
- §4.4 FR-6 — 필터 on/off 출력 동시 노출.
- §6.0 / §7 — SM 절대 목표(iPhone 30%/20%, Android 10%/8%, jitter 40%)는 잠정 → Phase-0로 확정.
- §8 NFR-3 — 구체 지원 기기 목록 미확정. NFR-7 — 발열 예산 미정량.
- §9 — Android 1차 경로를 metric 단안 신경망으로 가정, 등급은 Phase-0 게이팅.

---

## 13. Risk Register

| ID | 리스크 | 영향 | 완화 |
|---|---|---|---|
| **R-1** | Android metric 단안 depth(오차 >10%)가 RGB-only 절대 Z보다 정확하지 않아 보정 이득이 없거나 악화 | Android 가치 가설 붕괴 | **Phase-0 G-3 하드 게이트**로 사전 측정, 미달 시 experimental 강등(§9). FR-8 회귀 게이트로 런타임 악화 차단 |
| **R-2** | 평가가 절대 Z를 못 재거나(PA GT) GT depth를 입력으로 써 단안 오차를 은폐 → 허위 성공 | 잘못된 출시 판단 | §7 평가 조건: camera-space non-PA GT + 런타임 depth in-loop + 플랫폼별 분리 |
| **R-3** | LiDAR 256×192·<30cm에서 손가락 sub-pixel → fingertip 직접 보정 불가 | 손가락 끝 정확도 기대 미달 | SM-6을 간접 효과로 재정의, fingertip은 root/palm 앵커+본 길이 제약 의존 |
| **R-4** | metric 단안 depth 단독이 프레임 예산 잠식 → 30 FPS 실패 | 실시간 NFR 위반 | 비대칭 레이트(NFR-1), hand=NPU/depth=GPU 분리, INT8 양자화 |
| **R-5** | 기존 Recon이 per-joint confidence 미노출 → fusion 가중 설계 전제 붕괴 | FR-5 fusion 약화 | Open Q8 조기 검증, 미노출 시 2D 가시성/재투영 잔차로 대체 |
| **R-6** | Recon export(spiral-conv 커스텀 op) 변환 실패 | 통합 차질 | 변환 커버리지 조기 검증(현 repo onnx/tflite 작업 활용) |
| **R-7** | Android(노이즈) vs iPhone(LiDAR) 품질 절벽이 "동일 API"에 은폐 | 사용자 기대 불일치 | 지표 플랫폼별 분리, Android 등급 명시(§9) |

---

## 부록 A. 원 입력 FR ↔ PRD FR 매핑

| 원 입력 | PRD FR | 비고 |
|---|---|---|
| FR-1 RGB 입력 영상 처리 | FR-1 | |
| FR-2 Hand Joint Detection | FR-2 | 기존 Recon, 불변 |
| FR-3 Depth Acquisition (Android Monocular / iPhone LiDAR) | FR-3 | 플랫폼 적응형으로 통합 |
| FR-4 Depth Refinement | FR-4 + FR-5 | 샘플링/거부 + metric 재구성으로 분리 |
| FR-5 3D Joint Reconstruction | FR-5 | Root-Depth Alignment |
| FR-6 Temporal Filtering | FR-6 | |
| FR-7 Unity Visualization | — (v2) | MVP 제외, 출력 API만 PRD FR-7로 대체 |
| FR-8 XR Integration | — (v2) | MVP 제외 |
| (신규) Refinement 회귀 게이트 | FR-8 | 리뷰 반영 신규 — NFR-6/SM-C2 강제 |
| (신규) 출력 API | FR-7 | 원 FR-7과 번호 구별: 본 표의 PRD FR-7은 "출력 API" |

## 부록 B. Acceptance Criteria (요청 산출물)

각 항목은 해당 FR의 testable consequence에 근거한다.

- **AC-1 (FR-1):** AR 세션 프레임이 Recon 입력 규격으로 전처리되고, 백로그 시 stale 프레임이 폐기되며, timestamp+depth가 한 단위로 전달된다.
- **AC-2 (FR-2):** 프레임당 21개 관절의 2D·root-relative 3D·신뢰도가 출력되고, Recon 모델 가중치가 변경되지 않는다.
- **AC-3 (FR-3):** LiDAR 기기는 LiDAR Depth를, 그 외는 Monocular Depth를 자동 선택하며, Depth Map이 RGB 관절 좌표로 매핑 가능한 정렬 정보와 함께 제공된다.
- **AC-4 (FR-4):** 관절 깊이가 이웃 robust 통계+신뢰도 마스킹으로 추정되고, 가림/flying-pixel 샘플이 거부되며, 무효 관절이 표시된다.
- **AC-5 (FR-5):** intrinsics로 metric unproject가 이뤄지고, **shift-only** 전역 앵커링 + robust fitting으로 outlier에 견디며, 본 길이 제약 위반 보정이 거부된다.
- **AC-6 (FR-6):** 정지/저속 jitter가 감소(SM-3)하고 고속 lag이 악화되지 않는다(SM-C1).
- **AC-7 (FR-7):** 단일 API로 Refined Joint 스트림(보정 적용 플래그 포함)을 Android·iOS 동일 시그니처로 구독할 수 있고, 플랫폼 분기가 캡슐화된다.
- **AC-8 (FR-8):** 유효 depth 부재·정렬 실패·제약 위반 시 보정을 기각하고 RGB-only를 출력하며, 출력에 보정 적용 여부 플래그가 부착된다(NFR-6/SM-C2 강제).
- **AC-9 (NFR):** 타깃 기기에서 ≥30 FPS, End-to-End Latency <50ms, 런타임 메모리 <1GB, 지속 사용 시 발열 throttling으로 FPS가 무너지지 않는다.
- **AC-10 (SM/평가):** camera-space non-PA GT 데이터셋 + 런타임 depth in-loop(GT depth 입력 금지)로 **플랫폼별 분리** 측정해 Z-error·MPJPE·jitter가 Phase-0 확정 목표를 달성한다.
- **AC-11 (Phase-0 게이트):** G-1 베이스라인 측정·G-2 LiDAR 이득·G-3 Android 이득 검증을 통과해야 전체 구현에 착수하며, 그 결과로 SM 절대 수치와 Android 등급이 확정된다.

> **산출물 매핑(요청 Deliverables):** *System Requirement* → §8 NFR + §9 Platform/Hardware + §10 Integration · *Functional Requirement* → §4(FR-1~8) · *Non-Functional Requirement* → §8(NFR-1~7) · *Acceptance Criteria* → 본 부록 B(AC-1~11).
