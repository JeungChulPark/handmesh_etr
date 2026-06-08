# Adversarial Review — Depth-Aware Mobile Hand Pose Estimation PRD

리뷰 대상: `prd.md` (2026-06-08, draft) + `addendum.md`
리뷰어 입장: 냉소적·적대적. 이 프로젝트를 무너뜨릴 숨은 가정·말장난·검증 불가능한 요구사항·스코프 폭발을 찾는다.

---

## Verdict

이 PRD는 잘 구조화되어 있고 위험을 상당 부분 스스로 인지하고 있다(addendum G의 PA 경고, SM-C2 회귀 게이트 등). **그러나 제품의 핵심 가치 가설 — "Android에서 단안 metric depth로 절대 Z를 줄인다" — 가 실제로 성립하는지가 전혀 검증되지 않았고, 성공 지표(30%/20%/40%)는 근거 없이 박혀 있으며, 그 지표를 측정할 데이터셋/하니스가 확정되지 않았다.** 즉 "측정 가능하게 개선한다"는 약속의 측정 대상·기준·도구가 모두 미정인 상태에서 정량 목표만 단단해 보인다. Android와 iPhone을 "같은 API"로 묶는 결정은 정확도 절벽을 API 시그니처 뒤에 숨긴다.

---

## CRITICAL — 제품을 무너뜨릴 수 있는 문제

### C-1. [CRITICAL] Android 단안 metric depth의 가치 가설이 미검증 — 핵심 value prop이 통째로 흔들린다

Vision(§1)은 "단안 Depth 추정으로 metric 깊이를 얻어 Z를 견고하게 앵커링"한다고 단언한다. 그러나:

- Addendum A의 1순위 후보 Depth Anything V2 Small + metric 파인튜닝조차 모바일 실내 단안 metric depth의 **상대 오차는 일반적으로 두 자릿수(%)** 수준이다. 손이 카메라에서 ~30–50cm(UJ-1) 떨어져 있다면, 12% rel error는 절대 Z로 **수 cm**의 오차다.
- RGB-only Baseline의 Z 모호성이 만드는 오차도 정확히 그 "수 cm" 스케일이다(§1이 인정). 즉 **노이즈가 큰 depth prior로 노이즈가 큰 RGB Z를 보정**하는 상황이며, 순개선이 양수라는 보장이 전혀 없다.
- PRD 어디에도 "단안 depth의 절대 Z 오차 < RGB-only의 절대 Z 오차"임을 사전에 입증하는 **타당성(feasibility) 단계**가 없다. SM-1(Z-error ≥30% 감소)은 이 가설이 참이라는 전제 위에 서 있다.
- §9는 "<30cm 손가락을 분리 해상하지 못함 → Z prior로만 사용"이라고 스스로 한계를 적어놓고도, 그 prior가 root 단계에서 순개선을 준다는 증거는 제시하지 않는다. Root-Depth Alignment(FR-5)는 wrist/palm 깊이로 전역 scale·translation을 잡는데, 그 wrist 깊이 자체가 단안 추정의 가장 부정확한 부분(절대 metric)일 수 있다.

**결론:** Android 경로는 "depth가 RGB보다 절대 Z에서 더 정확하다"는, 문헌상 결코 자명하지 않은 가정에 전적으로 의존한다. 이 feasibility를 MVP 이전 spike로 못 박지 않으면, 개발 후반에 "Android는 개선이 없거나 오히려 악화"라는 결론에 도달할 수 있고, 그러면 Android value prop 전체가 증발한다. **이것이 이 프로젝트의 단일 최대 리스크이며, PRD는 이를 Open Question으로도 올려두지 않았다.**

### C-2. [CRITICAL] 성공 지표 30%/20%/40%는 임의 수치 — 게다가 측정 대상·도구가 모두 미확정

- SM-1/2/3의 30%/20%/40%는 `[ASSUMPTION]` 태그가 붙어 있고 "베이스라인 확정 후 수정"이라 적혀 있다. 즉 **현재 PRD에는 검증 가능한 성공 기준이 사실상 존재하지 않는다.** Q2(베이스라인 수치)와 Q4(데이터셋·metric Z GT 확보)가 미해결인 상태에서 정량 목표는 장식이다.
- 더 나쁜 점: 세 지표의 상대 크기 관계(Z 30% < jitter 40%, MPJPE 20%)에 아무 논거가 없다. jitter는 temporal filter로 비교적 쉽게 큰 폭 감소가 가능(1-Euro 표준)하지만, 절대 Z-error 30% 감소는 C-1의 가설이 강하게 성립해야만 가능하다. 즉 **가장 어려운 SM-1에 임의의 큰 목표가, 가장 쉬운 SM-3에 더 큰 목표가** 붙어 있어 난이도와 목표가 역상관일 수 있다.
- "측정 가능하게 감소"(SM-6, FR-6 consequence 다수)는 측정 가능성을 주장만 할 뿐 임계값이 없다 — counter-metric도 마찬가지로 "악화시키지 않아야 한다"만 있고 허용 오차(예: ±x mm 이내는 동률)가 없어 게이트로 작동 불가.

### C-3. [CRITICAL] 평가 갭 — 오프라인 벤치마크가 제품이 고치겠다는 것(절대 Z)을 실제로 측정하지 못할 위험

Addendum G가 "FreiHAND PA 지표는 형상 정확도이지 절대 Z가 아님 → non-PA/camera-space 지표 필요"라고 **정확히 지적**한 것은 칭찬할 만하다. 그러나 PRD 본문은 이 함정을 충분히 닫지 못했다:

- FreiHAND는 합성 배경 단일 손 데이터셋으로 표준 리더보드가 **PA-MPJPE 중심**이다. non-PA camera-space Z GT를 이 데이터셋들에서 일관되게·정확하게 뽑을 수 있는지(특히 metric 절대 깊이 GT)는 Q4로 미해결.
- 본 제품의 핵심 이득은 **절대 root depth / metric scale**인데, MANO 기반 GT가 root-relative로 정의된 데이터셋에서는 측정할 대상 자체가 없다. 즉 평가 하니스가 "제품이 고치는 축"을 못 보는 채로 PA 지표만 좋아질 수 있다(형상은 그대로이므로 PA는 변화 없음 → "개선 없음"으로 잘못 결론나거나, 반대로 PA가 변하지 않는 걸 근거로 "회귀 없음"이라 자축).
- **데이터셋 분포 ≠ 제품 분포.** FreiHAND/DexYCB의 depth GT는 깨끗한 스튜디오/RGB-D 센서다. 그런데 Android 런타임 depth는 단안 신경망이다. 오프라인 평가에서 **GT depth를 입력으로 쓰면** 단안 depth의 실제 오차가 평가에서 빠져버려 SM-1이 비현실적으로 좋게 나온다. 평가는 반드시 "런타임과 동일한 단안 depth 모델을 데이터셋 RGB에 돌려" 측정해야 하는데, PRD/하니스 명세에 이 구분이 없다(부록 B AC-9도 모호).

---

## HIGH — 심각하지만 치명적이진 않은 문제

### H-1. [HIGH] 크로스플랫폼 "동일 API"가 정확도 절벽을 숨긴다

FR-7/AC-7은 Android·iOS가 "동일 시그니처 API"로 Refined Joint를 낸다고 한다. UJ-2의 Mina는 "한 줄의 분기도 작성하지 않는다." 문제:

- LiDAR(256×192 metric, conf map, 사실상 GT급)와 단안 추정(>10% rel error)은 **출력 품질이 질적으로 다르다.** 같은 타입의 데이터가 흘러도 신뢰도가 천지차이인데, API가 이를 동질적으로 보이게 만든다.
- SM-1/2/3는 플랫폼 분리 없이 정의돼 있다. iPhone Pro에서 50% 개선, Android에서 0% 개선이면 평균은 "달성"으로 보이지만 Android 사용자는 가치를 못 받는다. **지표를 플랫폼별로 분리하지 않으면 Android 실패가 iPhone 성공에 가려진다.**
- 소비 앱이 "보정됐다"고 신뢰하고 가상 객체를 붙였는데 Android에서 어긋나면, 캡슐화된 API는 디버깅 불가능한 블랙박스가 된다. per-joint confidence는 노출되지만(FR-7), "이 플랫폼의 depth는 본질적으로 못 믿는다"는 플랫폼 레벨 품질 신호는 API 계약에 없다.

### H-2. [HIGH] 지연 예산 50ms·30FPS는 addendum 실측만 믿으면 비현실적

- Addendum E는 "hand ~1–3ms, depth ~1ms(양자화) → 30FPS 듀얼 여유"라고 한다. 그러나 이 숫자는 **순수 inference 커널 시간**만이다. End-to-End Latency(NFR-1 정의: 캡처→Refined Joint)는 카메라 캡처, AR 프레임 콜백, 색공간 변환·전처리, NPU/GPU 큐 대기, depth↔RGB 정렬·언프로젝트, RANSAC fit, temporal filter, Unity 마샬링까지 포함한다. 1–3ms inference가 50ms 예산의 신빙성을 보장하지 않는다.
- Android에서 hand=NPU / depth=GPU **동시 실행**을 전제로 예산을 맞춘다(NFR-3, addendum E). 실제 모바일 SoC에서 NPU/GPU 동시 점유는 열·전력·메모리 대역 경합을 일으키고, 듀얼 모델 동시 추론이 깔끔히 분리된다는 보장은 벤더·드라이버 의존이다. `[ASSUMPTION]`로 묶여 있으나 50ms 예산 전체가 이 가정에 걸려 있다.
- ARKit은 "depth만 원해도 VIO 전체 비용"(addendum F)을 강제한다. AR 세션 오버헤드가 30FPS 예산을 갉아먹는데, 이건 우리 모듈이 통제할 수 없는 비용이다.
- SM-C3(발열)는 카운터 지표로만 있고 임계·측정 프로토콜이 없다. 지속 30FPS+듀얼 추론은 thermal throttling을 부르며, 5분 후 FPS가 무너지면 "≥30FPS 유지"는 거짓이 된다. 측정은 "순간"인가 "지속"인가가 미정.

### H-3. [HIGH] 불변 Recon 모델에 대한 숨은 의존성

PRD는 Recon을 블랙박스·불변(NFR-5, FR-2)으로 다루지만 그 위에 모든 게 올라간다:

- FR-2는 Recon이 **Joint Confidence(c_vis)** 를 준다고 가정한다. 현 repo의 MobRecon export가 per-joint 히트맵 신뢰도를 실제로 노출하는가? fusion 가중(FR-5, addendum B-4)이 c_vis에 의존하는데, 만약 export가 신뢰도를 안 내놓으면 fusion 설계가 붕괴한다. **이건 가정이 아니라 검증해야 할 사실인데 Open Question에도 없다.**
- Recon은 root-relative만 출력한다(Glossary). Root-Depth Alignment는 Recon의 **상대 포즈가 충분히 정확**하다고 가정한다 — depth는 scale·translation만 고치고, 손가락 굽힘 등 상대 구조 오차는 못 고친다. 만약 RGB-only의 주 오차원이 root depth가 아니라 상대 포즈(예: 가림 시 손가락 굽힘 오추정)라면, depth 보정은 SM-2(MPJPE)에 거의 무력하다.
- §10이 "spiral-conv 등 커스텀 op 변환 커버리지 조기 검증 필요"라 적은 건 좋으나, 이게 안 되면 그라운드 제로다. git status에 onnx/tflite 변환이 진행 중(WIP)임이 보이는데, **export가 미완성인 자산에 정확도 보정을 올리는 순서 의존성**이 PRD 일정에 반영돼 있지 않다.

### H-4. [HIGH] SM-C2(가림 구간 회귀 금지)와 핵심 가치가 직접 충돌

- 가림은 손 추적에서 가장 흔하고 중요한 케이스다. 그런데 가려진 관절은 depth 샘플이 본질적으로 틀린다(앞 물체/손바닥 깊이를 읽음). FR-4가 occlusion test(`|z_depth−z_rgb|>τ`)로 거부한다지만, 이 테스트는 **z_rgb가 맞다고 가정**해야 작동한다 — z_rgb가 틀렸을 때(바로 보정하려는 그 상황) 가림 거부가 정상 샘플을 버리거나 불량 샘플을 통과시킬 수 있다.
- 즉 "depth가 가장 도움될 어려운 케이스(가림·손가락 끝)"에서 depth가 가장 안 믿기며, 그 결과 폴백(RGB 예측)으로 돌아가면 **그 관절에서는 RGB-only와 동일** = 개선 0. 개선은 쉬운 케이스(잘 보이는 손)에만 몰리고, 정작 SM-6(fingertip)·SM-1은 어려운 케이스에서 정체된다. 제품 서사("손가락 끝·Z를 고친다")와 실제 가능 영역이 어긋난다.

---

## MEDIUM — 스코프·일관성·말장난

### M-1. [MEDIUM] "재학습 없음"이 만드는 천장이 정량 목표를 제약

NFR-5/Non-Goal: 모델 불변. 이는 자산 보존엔 좋지만, **상대 포즈·히트맵 품질을 못 고친다**는 뜻이다. learned RGB-D fusion(addendum H, v2)을 막아두고 후처리 기하 보정만으로 MPJPE 20%를 노리는데, 후처리는 root/scale에 강하고 관절 구조에 약하다. SM-2(절대 MPJPE 20%)가 후처리만으로 달성 가능한지에 대한 근거가 없다.

### M-2. [MEDIUM] MVP 응집성 — "출력 API까지"인데 가치 입증은 시각화에 의존

v1은 데이터 API까지만(FR-7), 시각화·XR 인터랙션은 v2. 그런데 UJ-1(반지 정렬)의 climax는 시각적 정렬이고, §6.2가 "데모 임팩트 큰 항목"이라 스스로 인정한다. **v1만으로는 제품 가치를 사람이 체감할 데모가 없고**, 정량 지표(C-2/C-3 미해결)도 못 보여주면 v1 종료 시점에 "성공"을 입증할 수단이 빈약하다. MVP가 측정(연구자 UJ-3)에 전적으로 기대는데 그 측정 인프라가 Q4 미정이다.

### M-3. [MEDIUM] 비-Pro iPhone 폴백 정책 미정이 스코프를 흔든다

Q3(비-Pro iPhone: Monocular 일원화 vs LiDAR 전용 분기)이 열려 있다. 이건 사소한 디테일이 아니라 **테스트 매트릭스·QA 범위·플랫폼별 지표 정의**를 바꾼다. "단일 API" 약속과 결합하면, 같은 iOS 안에서도 LiDAR/단안 두 품질 등급이 공존하는데 이를 어떻게 노출/문서화할지 미정.

### M-4. [MEDIUM] 동기화·정렬을 과소평가

Addendum D는 ARCore depth가 "FOV crop + stale fallback" 가능하다고 인정한다. depth와 RGB가 다른 timestamp/시점이면 관절별 depth 샘플(FR-4)이 잘못된 픽셀을 읽는다 — 빠른 손 움직임에서 1프레임(33ms) 어긋남도 cm 단위 오차다. FR-3 consequence는 "정렬 가능한 최신 프레임"이라 적지만, 정렬 실패 시 품질 저하 한계·거부 정책이 testable하게 없다.

### M-5. [MEDIUM] Counter-metric이 게이트로 동작 불가

SM-C1/C2/C3 모두 "악화시키지 않아야 한다" 형태로 임계·측정법이 없다. addendum이 알고리즘은 제시하나, "고속 lag 악화 없음"을 어떤 시퀀스·어떤 ms 기준으로 판정하는지 미정이면 NFR-6(회귀 금지 게이트)는 강제력 없는 선언이다.

### M-6. [MEDIUM] "30FPS 처리량"과 "50ms 지연"의 이중 정의가 검수 빠져나갈 구멍

NFR-1이 둘을 구분한 건 정확하다. 그러나 파이프라이닝으로 "1프레임 초과 가능"(Glossary)이면, 처리량 30FPS를 맞추면서 지연이 누적될 수 있다. AC-8은 둘 다 요구하지만 "어느 부하·어느 기기·지속 시간"이 없어, 벤치를 유리한 조건에서 돌려 둘 다 통과시키는 게 가능하다.

---

## LOW — 정리되면 좋은 것

- **L-1.** Working title 미확정(§제목). 사소하나 draft 상태 신호.
- **L-2.** "metric을 목표로 한다"(Glossary Depth Map) — 목표일 뿐 보장 아님이 곳곳에 반복되는데, 단안 경로가 끝내 relative에 머물면 FR-5 scale 정렬이 단 2개 robust 관절(wrist/palm)로 전역 scale을 잡아야 해 매우 취약(2점 fit). 이 취약성이 명시 안 됨.
- **L-3.** FR-6 "필터 on/off 동시 노출" `[ASSUMPTION]` — 좋은 평가 기능이나 런타임 비용(이중 출력) 명시 없음.
- **L-4.** SM-4/5의 "타깃 기기"가 NFR-3 미확정(Q1)에 의존 — 기기 미정이면 성능 지표도 미정. 순환 의존.
- **L-5.** §9 메모리 "1GB 예산 여유"는 모델 가중치만 계산. 런타임 텐서·이미지 버퍼(latest-only 16장, addendum F)·AR 세션·Unity 힙은 미포함. <1GB가 여유라는 결론은 근거 부족.

---

## 검증/확정해야 할 것 (우선순위)

1. **[C-1 차단]** MVP 착수 전 Android 단안 depth feasibility spike: 데이터셋 RGB에 실제 단안 모델을 돌려 "단안 depth 절대 Z 오차 vs RGB-only 절대 Z 오차"를 비교. 순개선 음수면 Android 전략 재고. → Open Question에 추가 필요.
2. **[C-2/C-3 차단]** Q2(베이스라인 수치)·Q4(데이터셋·metric Z GT·런타임 depth 적용 방식) 선해결. 평가 하니스는 반드시 런타임과 동일한 단안 depth를 입력에 사용.
3. **[H-3]** Recon export가 per-joint confidence를 실제로 노출하는지, spiral-conv op 변환이 되는지 즉시 검증.
4. **[H-1]** SM-1/2/3를 **플랫폼별로 분리 정의**(Android / iPhone-LiDAR). 평균 뒤에 Android 실패를 숨기지 말 것.
5. **[H-4/M-5]** Counter-metric에 정량 임계·시퀀스·판정 프로토콜 부여. 가림 케이스 별도 슬라이스 평가.
6. **[H-2]** End-to-End 50ms를 inference가 아닌 capture→output 풀 경로로, 지속(>5분) 발열 조건에서 측정하도록 AC-8 강화.

---

*리뷰 종료. 이 PRD의 가장 큰 약점은 "무엇을 측정해 성공이라 부를지"가 미정인 채 정량 목표만 단단해 보이는 것, 그리고 Android value prop의 물리적 타당성을 한 번도 검증하지 않은 채 Vision에 단언으로 적은 것이다.*
