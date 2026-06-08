# Domain Feasibility & Correctness Review — Depth-Aware Mobile Hand Pose Estimation

- **리뷰 대상:** `prd.md`, `addendum.md` (prd-handmesh_etr-2026-06-08)
- **리뷰어 관점:** 모바일 3D 손 포즈 추정 + 뎁스 센싱 도메인 전문가 (기술적 타당성/정확성)
- **날짜:** 2026-06-08

---

## 0. Verdict

전체 아키텍처(RGB Recon은 불변, 그 위에 플랫폼 적응형 depth refinement 후처리)는 **도메인적으로 건전하고, 전역 Root-Depth Alignment를 v1 백본으로 잡은 선택은 정확히 옳다** — root-relative 포즈의 가장 큰 오차원인 절대 scale·root-depth 모호성을 직접 겨냥하기 때문이다. 그러나 **PRD는 두 개의 잠재적 치명 결함을 충분히 인지하지 못한다**: (1) Android monocular metric depth를 손 영역으로 scale-align하는 경로가 순환적(circular)일 위험, (2) LiDAR 256×192에서 손가락이 sub-pixel이라 per-joint depth 샘플이 손목/손바닥에만 신뢰 가능하다는 사실 — 이는 fingertip 개선(SM-6)의 직접적 위협이다. NFR 예산은 LiDAR 경로에서는 현실적이나 Android 듀얼모델 경로에서는 addendum의 latency 수치가 낙관적이다.

---

## 1. 계층형 Fusion 설계의 타당성 (Root-Depth Alignment as v1 backbone)

### [STRENGTH] 전역 정렬을 백본으로 선택한 것은 정확히 맞다
- Recon은 root-relative + scale-less 3D를 출력한다(Glossary 명시). RGB-only Z 오차의 지배적 원인은 **per-joint 노이즈가 아니라 전역 scale/depth 모호성**(작은 손 가까이 vs 큰 손 멀리)이다. addendum B-3의 weighted scale-and-shift LSQ `(s*,t*)=argmin Σ w_j(s·ẑ_j+t−z_j^depth)²` + RANSAC/MAD는 RootNet 계열의 표준이고 이 모호성을 직접 해소한다. 단 1~2개의 신뢰 가능한 metric 앵커(손목 깊이)만 있어도 21개 관절 전체가 metric으로 끌려온다 — depth가 sub-pixel/저해상도여도 효과가 큰 이유이고, **가성비 판단이 옳다.**

### [HIGH] scale **and shift** 2-파라미터 모델은 단일 평면 깊이로는 under-constrained
- B-3는 scale `s`와 shift `t`를 **동시에** 추정한다. 그러나 손이 카메라에 거의 정면(fronto-parallel)으로 놓이면 손목/손바닥 MCP 앵커들의 depth 분산이 작아 `s`와 `t`가 강하게 상관(공선성)된다 — `s·ẑ+t`에서 ẑ의 dynamic range가 좁으면 `s`는 사실상 결정되지 않고 회귀가 불안정해진다. UJ-1의 30–50cm 정면 손 시나리오가 바로 이 worst case다.
- **권장:** v1에서는 `s`를 **본 길이(bone-length) 사전값으로 고정**하고 `t`(root-depth)만 depth로 추정하는 shift-only 모드를 1차 백본으로 삼는 것이 훨씬 robust하다. 손 크기는 MANO/Recon mesh의 본 길이로 이미 알 수 있으므로 scale을 depth로 다시 풀 필요가 약하다. 이것이 결함이라기보다 PRD/addendum가 "scale도 동시 추정"을 무비판적으로 백본으로 적은 점을 명시적 리스크로 올려야 한다.

### [MEDIUM] 회귀 금지(NFR-6/SM-C2) 게이트의 판정 기준이 미정의
- FR-5/NFR-6은 "Baseline 대비 악화 금지"를 요구하지만, **언제 refinement를 채택/기각할지의 런타임 결정 규칙**이 없다. RANSAC inlier 수가 임계 미만이거나 fit residual이 크면 그 프레임은 RGB-only로 폴백해야 하는데, 이 fallback 트리거가 FR에 없다. SM-C2(가림 구간 왜곡 금지)는 측정 지표로만 존재하고 이를 **보장하는 메커니즘(FR)**이 빠져 있다.

---

## 2. LiDAR 256×192 + <30cm 손가락: sub-pixel & depth bleed

### [HIGH] PRD는 "분리 해상 불가"를 인지하나, 그 **결론을 FR로 끝까지 밀지 못함**
- §9 "공통 depth 한계"는 "<30cm 손가락을 분리 해상하지 못함 → 관절별 Z prior로만 사용"이라고 올바르게 적었다. 그러나 이 인지가 FR-4/FR-5/SM-6과 **모순**된다:
  - 30–50cm 거리·256×192·~60°FOV에서 손가락 두께(~1.5cm)는 LiDAR depth map에서 **1픽셀 미만**이다. fingertip 위치의 depth median 윈도는 거의 항상 **배경 또는 인접 손가락 깊이로 오염(depth bleed/flying pixel)**된다.
  - 따라서 **fingertip의 per-joint depth는 신뢰 신호가 아니다.** 그런데 SM-6은 "fingertip 오차 감소"를 fingertip의 depth 샘플로 달성하려는 듯 읽힌다. 실제로 fingertip 개선은 depth에서 직접 오는 게 아니라 **전역 root-depth 앵커링 + 본 길이 제약**의 간접 효과로만 와야 한다. PRD는 이 인과를 명확히 분리하지 않았다.
- **권장:** FR-4에 "손목/손바닥(palmar) MCP 관절만 metric 앵커 후보로 쓰고, 말단(PIP/DIP/TIP) 관절의 depth 샘플은 기본적으로 align에서 제외한다"는 **관절 클래스별 신뢰 정책**을 명시. SM-6은 "depth 직접 보정"이 아니라 "전역 정렬·본 제약의 부수 효과"로 재정의.

### [MEDIUM] robust median 윈도 크기·신뢰도 임계가 미지정 (적절히 testable하지 않음)
- FR-4는 "이웃 윈도 robust median + 신뢰도 마스킹"만 말하고 **윈도 크기, confidence threshold, occlusion τ_occ**가 전부 addendum에도 기호로만 있다. 256×192에서 윈도를 너무 크게 잡으면 인접 손가락/배경을 흡수하고, 너무 작으면 single-pixel 노이즈에 노출된다. 이 파라미터는 거리 의존적이라 **상수로 둘 수 없다**(픽셀당 metric 크기가 거리에 비례). FR-4 consequence에 "거리/depth에 적응하는 윈도"가 빠져 있다.

### [LOW] LiDAR confidenceMap 활용이 FR에 약하게만 반영
- addendum A는 ARKit confidenceMap(low/medium/high)을 언급하나 FR-4의 testable consequence는 "신뢰도 마스킹"으로 추상화만 함. 손 경계의 low-confidence 픽셀 일괄 거부는 fingertip을 통째로 날릴 수 있어, confidence와 occlusion-test를 **어떤 순서로 결합**하는지 명세 필요.

---

## 3. Android monocular metric depth — circular scaling 위험

### [HIGH — 핵심 결함] FR-5 scale 정렬로 monocular relative→metric화하는 것은 잠재적 순환
- FR-3은 "Monocular이 relative만 주면 §FR-5의 scale 정렬로 metric화한다"고 한다. 그런데 FR-5의 scale 정렬은 **손 관절을 앵커로** depth의 scale을 푼다. 즉:
  - depth를 손으로 scale → 그 depth로 손을 보정 → **손의 절대 Z 정보가 depth에 들어온 적이 없다.**
  - relative depth를 손 관절 위치로 metric화하면, 그 metric은 "Recon이 추정한 손 크기/위치"를 그대로 재주입한 것이다. 결과적으로 **depth는 RGB가 이미 가진 정보 외에 새 절대 Z를 추가하지 못하고**, SM-1(Z-error 30% 감소)·SM-2(MPJPE 20% 감소)의 근거가 무너진다. 이것은 **리뷰에서 지적해야 할 가장 중요한 정확성 결함이다.**
- **이 문제가 성립하지 않으려면** 둘 중 하나가 필요하고, PRD는 둘 다 명시하지 않았다:
  1. **Depth Anything V2 metric 파인튜닝 헤드가 진짜 metric을 출력**(relative가 아님). addendum A-1은 "metric 파인튜닝"을 1순위로 적었으므로, 이 경우 FR-3의 "relative→FR-5로 metric화" 폴백 문구는 **삭제**되어야 한다. metric 모델이면 align은 bias 보정 수준이지 scale 결정이 아니다.
  2. metric 헤드의 절대 정확도가 충분(±수 cm)해서 손과 무관한 독립 앵커로 기능. 이 경우 정확도 가정을 NFR/Open Q로 명시해야 한다.
- **현재 PRD는 (relative→손으로 scale)과 (metric 헤드)를 동시에 모순되게 허용**한다. 이 모호성을 반드시 해소해야 하며, 권장은 "monocular 경로는 **metric 모델 출력만 신뢰**하고, relative-only 모델은 v1 후보에서 배제(또는 hand-anchored scaling은 명시적으로 '새 Z 정보 없음'을 인정하고 SM-1을 LiDAR 기기로 한정)"이다.

### [MEDIUM] monocular metric depth의 현실적 정확도 자체가 손 보정에 부족할 수 있음
- Depth Anything V2 metric은 실내 **룸스케일** GT(Hypersim/vKITTI 등)로 파인튜닝된다. 30–50cm **근거리 손**은 학습 분포 밖이고, near-field에서 metric 오차가 수 cm~10cm급으로 커지는 것이 일반적이다. 손목 절대 Z를 ±5cm로 못 주면 SM-1 30% 감소는 Android에서 비현실적이다. PRD는 LiDAR/Monocular를 **동일 SM 목표**로 묶었는데, **Android는 별도(완화된) 목표**로 분리해야 한다.

---

## 4. FR에서 기술적으로 빠진 필수 요소

### [HIGH] 카메라 intrinsics가 입력 요구사항으로 명시되지 않음
- 2D 픽셀 + depth(z)를 **3D metric 좌표로 unproject**하려면 intrinsics(fx,fy,cx,cy)가 반드시 필요하다. addendum D는 언급하나, **FR 본문에는 intrinsics를 필수 입력으로 받는 요구가 없다.** ARKit은 ARFrame에서 제공하지만 ARCore/Recon 전처리에서 crop·resize가 일어나면 intrinsics가 변한다. FR-1(전처리)이 crop/resize를 하는데 그에 따른 **intrinsics 변환 추적**이 빠져 있다 — 이게 틀리면 모든 metric 결과가 systematically 틀어진다.

### [HIGH] Hand-region masking / RGB↔depth 픽셀 대응이 FR에 부재
- FR-4는 "관절 2D 위치에서 depth 샘플"하지만, **Recon 입력 RGB 좌표계 ↔ depth map 좌표계의 매핑**(특히 Recon이 손 bbox crop을 쓰는 경우)이 명세되지 않았다. crop된 입력의 관절 좌표를 full-frame depth로 되돌리는 변환이 누락. 또한 손 segmentation/mask 없이 윈도 median만으로는 손 경계에서 배경을 못 거른다.

### [MEDIUM] 좌/우손 처리 미명세
- Glossary/FR 전반이 "단일 손"을 가정하나(v1 OK), Recon이 좌/우손 중 무엇을 내는지, chirality에 따라 **본 길이 제약(FR-5)의 토폴로지/mirror**가 달라진다. "단일 손"이라도 좌/우 판별과 본 모델 선택이 FR-5에 필요하다.

### [MEDIUM] Depth↔RGB temporal alignment under async inference (Android)
- addendum D/E는 latest-only 스냅샷을 말하나, **Android에서 hand(NPU)와 monocular depth(GPU)가 비동기로 다른 latency**를 가지면, 두 출력이 **다른 프레임**을 가리킬 수 있다(손은 움직인다). FR-1은 "timestamp+depth 한 단위 묶음"을 말하지만, **monocular depth가 같은 RGB에서 추론되므로 자동 정렬**된다는 점(addendum 장점)과 **두 모델 출력 시점 불일치**는 다른 문제다. 빠른 손 움직임에서 stale depth로 align하면 오차가 들어온다 — FR에 "depth-joint 시점 일치 보장 또는 모션 보상" 요구가 없다.

### [LOW] Depth 단위·좌표계 규약(z 부호, camera-space 정의)이 Glossary에 없음
- Z-error/SM-1을 측정하려면 camera-space z의 정의(카메라에서 멀어지는 +)가 고정돼야 한다. addendum G가 "non-PA/camera-space"를 옳게 짚었으나, PRD 본문 Glossary에 좌표 규약이 없어 평가 하니스에서 부호 혼동 위험.

---

## 5. NFR 예산 현실성 (30 FPS 듀얼모델 / <50ms / <1GB)

### [STRENGTH] LiDAR 경로는 전적으로 현실적
- sceneDepth는 compute ~0, NPU 전체를 hand에 할애 → 30 FPS·<50ms·<1GB 모두 여유. addendum E의 판단 타당.

### [MEDIUM] Android 듀얼모델 latency 수치가 낙관적
- addendum E는 "hand ~1–3ms, depth ~1ms(양자화) → 듀얼 여유"라 적었으나:
  - Depth Anything V2 Small은 addendum A 자신이 **"~34ms iPhone15Pro NE"**로 적었다. addendum E의 "depth ~1ms"와 **자기모순**(MiDaS급 가정과 DAv2 가정이 섞임). 24.7M 파라미터 ViT 인코더가 모바일 NPU에서 1ms는 비현실적 — INT8이어도 보통 10–30ms대다.
  - hand+depth를 NPU/GPU로 분리한다 해도, depth 34ms면 **depth 단독으로 30 FPS(33ms) 예산을 거의 소진**한다. addendum E의 "격프레임 실행+직전 depth 재사용" 폴백이 사실상 필수 경로가 되고, 그러면 §3의 temporal alignment 문제가 악화된다.
- **권장:** Android의 30 FPS는 "depth를 매 프레임이 아니라 N프레임마다, hand는 매 프레임"이라는 **비대칭 레이트**를 NFR-1에 명시. <50ms end-to-end는 depth가 critical path에 있으면 위태로우므로 depth를 파이프라인 밖(1프레임 지연 허용)으로 두는 설계 가정을 명문화.

### [LOW] 메모리 1GB는 안전, 단 듀얼 런타임 오버헤드 미포함
- §9는 모델 weight(~17MB+수십MB)만 계산. 실제로는 **AR 세션(VIO) + 카메라 버퍼 + 두 런타임(delegate workspace) + Unity** 합산이 지배적이다. addendum F가 ARKit VIO 비용을 짚었으나 메모리 추정에는 반영 안 됨. 여전히 1GB 안일 가능성 높으나 "weight 기준 여유"는 오해 소지.

---

## 6. 우선순위 권고 (PRD 수정 항목)

1. **[HIGH] §3 circular scaling 해소:** FR-3의 "relative→FR-5 scale로 metric화" 문구를 제거하고 "monocular은 metric 모델 출력만 사용"으로 고정. 불가하면 SM-1(Z 30%↓)을 **LiDAR 기기 한정**으로 분리.
2. **[HIGH] §4 intrinsics + crop 좌표 변환을 FR-1/FR-4의 명시적 요구로 추가.**
3. **[HIGH] §2 fingertip은 depth 직접 샘플 대상에서 제외**하고 root/palm 앵커 + 본 제약의 간접효과로 재정의. SM-6 인과 수정.
4. **[HIGH] §1 scale-and-shift를 v1에선 shift-only(본 길이로 scale 고정)**로 권장. 정면 손 공선성 리스크 명시.
5. **[MEDIUM] §1 refinement 채택/기각 런타임 게이트(FR)** 추가 — NFR-6/SM-C2를 보장하는 메커니즘.
6. **[MEDIUM] §5 Android 비대칭 depth 레이트**를 NFR-1에 명시, addendum E의 "depth ~1ms"를 DAv2 실측(~34ms)과 정합.

---

## 7. 종합

- **건전한 핵심:** "모델 불변 + 후처리 refinement", "전역 Root-Depth Alignment 백본", LiDAR 경로 예산, 평가에서 non-PA/camera-space Z를 명시한 점, robust median + occlusion test + 본 길이 제약의 계층 구조 — 모두 도메인 표준에 부합.
- **반드시 고칠 결함:** Android monocular의 circular scaling(§3), intrinsics/crop 변환 누락(§4), fingertip depth 신뢰성 과대평가(§2). 이 셋이 미해결이면 SM-1/2/6은 특히 **Android에서 달성 불가** 위험이 크다.
- **현실적 기대치 조정:** v1의 측정 가능한 개선은 **LiDAR iPhone에서 가장 확실**하고, Android는 monocular metric 정확도 한계로 보수적 목표가 필요하다.
