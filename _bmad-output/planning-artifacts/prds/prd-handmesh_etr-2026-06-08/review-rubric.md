# PRD Quality Review — Depth-Aware Mobile Hand Pose Estimation

## Overall verdict

이 PRD는 강한 전략적 thesis(RGB의 Z 모호성을 모델 재학습 없이 후처리 Depth 보정으로 해소)를 일관되게 끌고 가며, FR마다 testable consequence, 명시적 Non-Goals, counter-metric, 부록 AC/매핑까지 갖춘 보기 드물게 well-structured한 문서다. 가장 큰 리스크는 핵심 성공 지표(SM-1~3, SM-6)의 목표 수치가 전부 `[ASSUMPTION]`이고 RGB-only Baseline 실측치(Q2)가 미확정이라 정확도 측면의 "done"이 아직 검증 불가라는 점 — 즉 green-light-to-build가 아니라 baseline 측정이 선행돼야 하는 PRD다. 이 의존성만 reviewer가 명확히 인지하면 다운스트림(아키텍처/스토리) 분해에 바로 투입 가능한 수준이다.

## Decision-readiness — strong

의사결정자가 행동 가능한 형태로 trade-off가 정직하게 드러나 있다. "모델 재학습하지 않는다"는 핵심 베팅이 Vision(§1, "기존 Recon Hand Tracking을 교체하지 않고")과 NFR-5에서 결정으로 명시되며, 포기한 것(learned RGB-D fusion, 모델 정확도 자체 개선)도 §5/§6.2/addendum H에 분명히 적혀 있다. Open Questions(§11)는 진짜 열려 있다 — Q2(Baseline 수치), Q6(보정 알고리즘 범위: 전역 정렬만 vs 관절별 fusion까지)는 수사적 질문이 아니라 실제 미결 결정이며, Q6는 §4.3 Notes·addendum B와 연결돼 결정 지점이 추적된다. `[NOTE FOR PM]` 콜아웃도 안전한 체크포인트가 아니라 실제 긴장 지점(§4.2 "비-Pro iPhone 폴백 정책 확정 필요", §6.2 "데모 임팩트 큰 항목 우선 재검토")에 위치한다.

한 가지 짚을 점: Q3(비-Pro iPhone 폴백)는 §2.2·§4.2·§9·Assumptions Index에서 이미 "Monocular 일원화"로 사실상 결정된 것처럼 서술되는데 Q11에서는 여전히 열린 질문으로 남아 있어, 결정/미결정 상태가 모호하다.

### Findings
- **medium** Q3 폴백 정책의 결정 상태 모호 (§11 Q3 vs §2.2/§4.2/§9) — 본문은 비-Pro iPhone을 Monocular로 폴백한다고 단정적으로 기술(`§4.2 "비-Pro iPhone → Monocular 폴백"`, Assumptions Index 4행)하지만 Q3는 "Monocular 일원화 vs LiDAR 기기 전용 분기"를 여전히 열린 질문으로 둔다. *Fix:* §4.2 결정이 잠정인지 확정인지 명시하고, 확정이면 Q3를 닫거나 "검증 필요한 가정"으로 재분류.

## Substance over theater — strong

furniture가 거의 없다. JTBD(§2.1)와 UJ(§2.3)는 각각 실제 FR을 견인한다 — Mina의 "한 줄의 분기도 작성하지 않는다"는 FR-7의 "플랫폼 분기 캡슐화" consequence로, Dr. Park의 정량 입증은 §7 SM·addendum G로 직결된다. NFR이 boilerplate가 아니라 product-specific threshold를 가진다(NFR-1 `<50ms`/`≥30 FPS`, NFR-2 `<1GB`, NFR-3 Snapdragon 8 Gen·Apple A17/A18). Vision(§1)은 이 카테고리의 어느 PRD에도 swap되지 않는, 도메인 고유의 문제 진술(작은 손 가까이 vs 큰 손 멀리의 투영 모호성)로 시작한다. Non-Users(§2.2)도 막연한 페르소나 나열이 아니라 실제 scope 경계(TrueDepth 셀피, 비-LiDAR 하드웨어 depth 기대)를 긋는다. 차별화·혁신 theater 섹션을 억지로 넣지 않은 점도 좋다.

## Strategic coherence — strong

명확한 thesis가 있고 모든 feature가 그 arc를 따른다. thesis: "RGB의 Z 모호성은 본질적 한계이므로 모델을 건드리지 말고 플랫폼 적응형 metric depth로 후처리 앵커링하라." Feature 우선순위가 이 thesis에서 도출된다 — addendum B/§4.3가 "전역 Root-Depth Alignment를 백본으로 우선, 관절별 fusion은 선택"으로 가성비 기준 우선순위를 명시하며, 이는 "쉬운 것부터"가 아니라 thesis(Z 앵커링이 가치 핵심)에서 나온 결정이다. Success Metric이 thesis를 검증한다 — SM-1(Z-error)·SM-2(MPJPE)·SM-3(jitter)는 활동 지표가 아니라 가설 검증 지표이고, counter-metric(SM-C1 lag, SM-C2 가림 왜곡, SM-C3 메모리/발열)이 각 SM에 대응돼 명시됐다. MVP scope kind는 problem-solving형(Z 오차 해소)으로 일관되며 scope 로직이 일치한다.

## Done-ness clarity — adequate

FR 레벨의 functional done-ness는 강하다. 모든 FR이 "Consequences (testable)"를 가지며 부록 B AC로 roundtrip된다. "graceful/reasonable/user-friendly" 류의 빈 형용사가 거의 없고, 대부분 검증 조건이 구체적이다(FR-3 "RGB 관절 좌표계로 매핑 가능한 정렬 정보와 함께", FR-4 "이웃 윈도 robust 통계+신뢰도 마스킹"). NFR도 수치 bound를 가진다.

약점은 두 군데다. 첫째, 정확도 SM(SM-1/2/3/6)의 목표 수치가 전부 `[ASSUMPTION]`(30%/20%/40%)이고 RGB-only Baseline 절대치가 Q2로 미확정이라, "정확도 done"이 현재로선 측정 불가다 — AC-9도 "확정 수치는 §Q2 후 고정"으로 미루며 이 의존성을 인정한다. 이건 PRD의 본질적 단계 문제(baseline 측정 선행)이지 작성 결함은 아니나, done-ness에 직접 영향을 준다. 둘째, 임계값에 의존하는 consequence들이 구체 bound 없이 정성적으로만 남는다 — FR-4 "샘플 깊이가 RGB 예측과 임계 이상 불일치", FR-6 "측정 가능하게 감소"는 addendum(τ_occ, 1-Euro fcmin/β)에 메커니즘은 있으나 PRD 본문 AC만으로는 pass/fail 경계가 정의되지 않는다.

### Findings
- **high** 핵심 정확도 SM이 전부 잠정치이고 Baseline 미확정 (§7 SM-1~3·SM-6, §11 Q2, 부록 B AC-9) — "≥30%/≥20%/≥40% 감소"가 모두 `[ASSUMPTION]`이고 RGB-only Baseline의 현재 MPJPE/Z-error/jitter가 측정되지 않아, 제품의 1차 가치(정확도)에 대한 done 판정이 불가능하다. *Fix:* Baseline 측정을 build 착수 전 명시적 선행 작업(또는 Phase 0)으로 PRD에 못 박고, 측정 후 목표치를 확정해 `[ASSUMPTION]`을 해제.
- **medium** 거부/평활 임계 consequence가 본문에 정량 bound 부재 (§4.3 FR-4, §4.4 FR-6) — "임계 이상 불일치", "과도한 lag 유발하지 않음", "측정 가능하게 감소"는 정성적이라 AC만으로 pass/fail 경계가 없다. *Fix:* addendum의 τ_occ·1-Euro 파라미터를 AC 수준의 검증 기준(예: 정지 시퀀스 Z 표준편차 임계)으로 본문/부록 B에 끌어올리거나, "튜닝 후 확정" 임을 명시.

## Scope honesty — strong

omission이 일관되게 명시적이다. §5 Non-Goals가 실제 work를 하며(모델 재학습 제외, 시각화/XR 인터랙션 v2 이연, 양손·손-객체 제외, 데스크톱 GPU 제외, TrueDepth 제외, 비-LiDAR 하드웨어 depth 보장 안 함), §6.2가 MVP 제외를 v2/v2+로 등급화한다. `[ASSUMPTION]` 태그가 사용자가 직접 확정하지 않은 추론(후면 카메라 가정, 빈 결과 정책, 폴백 정책, 목표 수치)에 붙고 §12 Assumptions Index에 색인된다. De-scoping(원 FR-7/8 → v2)이 부록 A 매핑 표로 투명하게 추적된다.

open-items density는 stakes 대비 적정선이나 다소 높은 편이다 — Open Questions 7개 + `[ASSUMPTION]` 8건(+ 인라인 몇 건) + `[NOTE FOR PM]` 2건. launch-grade green-light PRD치고는 미결 항목이 많지만, 대부분이 "baseline/기기목록/데이터셋 확정" 같은 build 전 자연 선행 항목이라 blocker라기보다 명시적 의존성으로 정직하게 노출된 것에 가깝다. 다만 정확도 목표 미확정(Q2)은 high-stakes 항목이므로 위 done-ness finding으로 승계한다.

## Downstream usability — strong

이 PRD는 chain-top(아키텍처 → 스토리로 공급)이라 이 dimension이 무겁게 적용되며, 잘 버틴다. §3 Glossary가 존재하고 도메인 명사(Recon, Refined Joint, Root-Depth Alignment, Joint Confidence, Depth Source 등)가 FR·UJ·SM 전반에서 동일하게 사용된다. FR/SM/NFR/AC ID가 연속·고유하며 cross-reference("Validates FR-4, FR-5", "Counterbalances SM-3", "Realizes UJ-1")가 해소된다. 각 FR 섹션이 단독으로 의미를 가지며 "see above"가 아니라 Glossary 용어로 참조한다. UJ는 모두 named protagonist(Jin/Mina/Dr. Park)를 가지고 context를 인라인으로 운반한다. 부록 A(원 FR↔PRD FR 매핑)와 부록 B(AC)는 다운스트림 추출을 명시적으로 돕는다.

작은 흠: §4.5 FR-7 "Out of Scope (v1)"가 "원 FR-7/원 FR-8"을 언급하는데, 같은 "FR-7" 토큰이 본 PRD의 FR-7과 원 입력 FR-7 두 의미로 쓰여 빠르게 읽을 때 혼동 가능(부록 A·"원" 접두어로 구제되긴 함).

### Findings
- **low** "FR-7" 토큰의 이중 지칭 (§4.5, §5, §6.2, 부록 A) — 본 PRD FR-7(출력 API)과 원 입력 FR-7(Unity 시각화, v2)이 같은 토큰을 공유한다. *Fix:* 원 입력 참조는 일관되게 "원-FR7"/"orig FR-7"처럼 prefix를 고정.

## Shape fit — strong

shape가 제품에 맞다. 이 제품은 개발자(통합 표면)·연구자(정량 입증)·최종 XR 사용자를 동시에 가지는 multi-stakeholder 기술 capability spec이며, PRD는 그에 맞게 named-protagonist UJ(load-bearing)와 capability-style FR을 둘 다 채택했다 — 과형식화(단일 오퍼레이터 도구에 UJ 남발)도 과소형식화(consumer인데 UJ 없음)도 아니다. SM이 user-facing 정확도 지표와 operational 성능 지표(FPS/메모리)를 적절히 혼합한다. Brownfield 성격(기존 Recon/MobRecon repo 위 구축)이 §3 Glossary("현 repo의 MobRecon 계열")·§10("현 repo의 onnx/tflite 변환 작업과 정합")에서 정확히 참조되고, 신규(보정 모듈)와 기존(Recon, 불변)이 FR-2 "불변" 표기로 명확히 구분된다.

## Mechanical notes

- **Glossary drift:** 거의 없음. 용어가 case/plural 일관. "Refinement Module"(§3, §4.5 정의)이 "Depth Refinement Module"(§0)과 표현이 다르나 동일 대상 — 사소.
- **ID 연속성:** FR-1~7, SM-1~6 + SM-C1~3, NFR-1~6, UJ-1~3, AC-1~9 모두 연속·고유. 미해소 cross-ref 없음. SM/NFR/FR의 Validates/Counterbalances/Realizes 링크 전부 실재 ID로 해소됨.
- **Assumptions Index roundtrip:** §12 인덱스 8개 항목이 인라인 `[ASSUMPTION]`과 대체로 대응. 인라인 태그 일부(예: §4.4 FR-6 `[ASSUMPTION]`, §5의 무라벨 `[ASSUMPTION]` 들)는 라벨 텍스트 없이 태그만 있어 인덱스 항목과 1:1 매칭이 텍스트로는 느슨함 — 인라인 태그에 짧은 라벨을 달면 roundtrip이 견고해짐.
- **UJ protagonist:** UJ-1 Jin, UJ-2 Mina, UJ-3 Dr. Park 모두 named·context 인라인. floating UJ 없음. UJ-1만 full 6단계(Persona/Entry/Path/Climax/Resolution/Edge case), UJ-2·3은 축약형이나 stakeholder 성격상 적정.
- **Required sections:** Vision·Target User·Glossary·Features(FR)·Non-Goals·MVP Scope·Success Metrics·NFR·Constraints·Dependencies·Open Questions·Assumptions Index·AC 부록 모두 present. launch-grade 기준 누락 섹션 없음.
