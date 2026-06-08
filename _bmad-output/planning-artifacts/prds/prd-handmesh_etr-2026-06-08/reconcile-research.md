# 연구 문서 ↔ PRD/Addendum 정합성 점검 (Reconcile)

**입력 연구 문서:** `technical-mobile-depth-aware-hand-pose-estimation-system-research-2026-06-08.md`
**대상 산출물:** `prd.md`, `addendum.md` (prd-handmesh_etr-2026-06-08)
**점검일:** 2026-06-08

본 문서는 기술 연구의 load-bearing 결론이 요구사항(FR/NFR/Risk/SM)으로 충실히 반영되었는지 확인하고, **요구사항을 형성했어야 하나 누락된 발견**을 추려낸다.

---

## 1. 잘 반영된 핵심 결론 (확인)

연구의 다수 핵심 결론은 PRD/Addendum에 명시적으로 보존되었다. 누락이 아님을 분명히 한다.

- **Metric vs Relative depth 임계성** — Glossary(Depth Map, Monocular Depth), FR-3 Consequence("가능한 한 metric"), FR-5 scale 정렬, Addendum A("metric 필요"), Q5(라이선스)로 충실 반영.
- **LiDAR 손가락 sub-pixel 한계** — §9 공통 depth 한계(<30cm·256×192·Z prior로만), FR-4(이웃 median, 단일 픽셀 불신), Addendum A로 반영.
- **Android depth-from-motion 취약(정지 카메라+움직이는 손)** — §9 Android, §12 Assumptions, Addendum A로 반영. Monocular을 1차 경로로 정책화.
- **모델 라이선스 게이트(WiLoR NC-ND, UniDepth NC)** — Addendum A "라이선스 게이트", Q5로 반영.
- **FreiHAND PA vs absolute-Z 캐비엇** — Addendum G에 "PA는 형상 정확도이지 절대 Z 아님 → non-PA/camera-space로 Z-error 측정"으로 정확히 반영. SM-2도 "절대, non-PA" 명시.
- **런타임/NPU 현실(NNAPI deprecated, LiteRT-QNN, hand=NPU/depth=GPU 분리)** — NFR-3, Addendum E로 반영.
- **MobRecon post-processing·모델 불변·spiral-conv export 리스크** — NFR-5, §10, Addendum로 반영.

---

## 2. 누락/약화된 발견 (요구사항에 반영했어야 함)

다음은 연구에서 load-bearing이나 PRD/Addendum에서 **누락되었거나 약하게 처리되어** FR/NFR/Risk/SM을 형성했어야 하는 항목이다.

### G-1. (중요) ARCore depth FOV crop → 좌표 매핑 단순 스케일 불가
연구 §통합 패턴 2는 "ARCore depth는 카메라 FOV의 **crop**(aspect ratio 다를 수 있음) → 단순 스케일 불가, `transformCoordinates2d()` 사용"을 명시한다. ARKit은 사전 정합(스케일만)이나 ARCore는 다르다. PRD FR-3 Consequence는 "정렬 정보(intrinsics/transform)와 함께 제공"으로 뭉뚱그려, **Android 경로의 비자명한 좌표 정합 위험**이 testable consequence나 리스크로 노출되지 않음. (Addendum D에는 언급되나 PRD 본문 FR/리스크로 승격 안 됨.) → FR-3에 Android crop 정합 별도 consequence, 또는 Open Question 추가 권장.

### G-2. (중요) 정확도 사다리 — SOTA 대비 실시간 모델 격차 ~1mm PA
연구는 "WiLoR/Hamba ↔ 실시간군 격차가 **~1mm PA에 불과**"라는 결론으로 "MobRecon급으로 충분"을 정당화한다. 이는 **'왜 Recon 모델을 교체/재학습하지 않는가'의 핵심 근거**이나 PRD는 "기존 자산 보존"만 말하고 이 정량 근거를 드롭. NFR-5/Non-Goal의 rationale로 보강 가치.

### G-3. (중요) MediaPipe world-landmark z를 metric GT로 신뢰 금지
연구 §2는 "MediaPipe world landmarks의 z는 약함/근사, **Google이 깊이 정확도 미공개 → metric 소스 아님, prior/sanity-bound로만**"을 강조. PRD는 MediaPipe를 Recon 대안/depth prior 소스로 검토하지 않았고, 만약 향후 baseline·prior로 끌어들일 경우의 **안티패턴 경고가 누락**. (현 repo는 MobRecon이라 직접 영향은 낮으나, prior 후보로 들어올 위험 존재.) → Addendum 또는 Q에 "MediaPipe z를 metric으로 쓰지 말 것" 명시 권장.

### G-4. (보통) 단안 depth metric 오차 정량(>12% vs 센서 ~1%) 미반영
연구 §통합 2/§기술스택은 "단안 metric 오차 >12% vs HW 센서 ~1%(실내)"를 명시. 이는 **Android/비-Pro 경로의 정확도 상한**을 정량화하며, SM 목표 수치(SM-1 Z-error 30% 감소 등)의 **현실성 검증과 플랫폼별 차등 기대치** 설정의 근거. PRD는 "Monocular 정확도 한계"를 정성적으로만 언급. → SM 또는 §9에 플랫폼별 정확도 기대 차등(LiDAR vs Monocular)을 정량 앵커로 추가 권장.

### G-5. (보통) ARKit "depth만 원해도 VIO 전체 비용" → 발열/배터리 NFR 부재
연구 §통합 5는 ARKit이 LiDAR depth만 써도 **VIO 월드트래킹 전체 비용·발열·배터리**를 지불하므로 소비를 throttle하라고 명시. PRD는 SM-C3(메모리/발열 counter-metric)로 발열을 부분 인지하나, **AR 세션 자체의 상시 비용**이라는 원인과 throttle 요구가 NFR/제약으로 명시되지 않음. (Addendum F에 원인은 있으나 NFR 미승격.) → 전력/발열 NFR 또는 §9 제약 추가 권장.

### G-6. (보통) TrueDepth(전면)가 근거리 손엔 LiDAR보다 고해상 — 제외 근거 비대칭
연구는 "정면 근거리 손은 **TrueDepth가 LiDAR보다 고해상**(단 세대별 캘리브 불일치)"을 명시. PRD는 전면 TrueDepth 경로를 단순 v1 제외(§2.2/§5)했으나, **근거리 손 정확도 측면에서 오히려 우월할 수 있다는 트레이드오프**를 기록하지 않아 v2 우선순위 판단 근거가 소실. → Addendum H(v2 후보)에 "TrueDepth 근거리 우월성·캘리브 불일치" 노트 추가 권장.

### G-7. (경미) ARCore acquire 이미지 close 누락 → 크래시(maxImages=16)
연구 §통합 5는 "acquire한 이미지 반드시 close, 초과 시 IllegalStateException"을 경고. Addendum F에 있으나 PRD FR-1(프레임 수집) 또는 리스크에는 없음. 구현 크래시 리스크라 FR-1 consequence로 승격 가치(경미).

### G-8. (경미) bilinear 금지·nearest/segment 보간 규칙
연구 §통합 1은 해상도 불일치 샘플링에서 "**bilinear 금지**(불연속에서 가짜 깊이), nearest 또는 segment 내 보간"을 명시. Addendum B-1에 보존되나 FR-4 consequence에는 "robust 통계(median)"만 있고 bilinear 금지가 빠짐. testable rule이므로 FR-4에 추가 가치(경미).

---

## 3. 종합 판단

PRD+Addendum은 연구의 **1차 임계 결론(metric/relative, LiDAR sub-pixel, Android motion 취약, 라이선스, FreiHAND PA, NPU 현실)을 모두 포착**했다. 누락은 주로 **2차 구현·정확도·전력 디테일**이며, 그중 **G-1(ARCore FOV crop), G-2(정확도 사다리 근거), G-3(MediaPipe z 경고)**가 요구사항/리스크로 승격할 가치가 가장 크다. 나머지(G-4~G-8)는 Addendum 또는 Open Question 보강 수준으로 충분하다.
