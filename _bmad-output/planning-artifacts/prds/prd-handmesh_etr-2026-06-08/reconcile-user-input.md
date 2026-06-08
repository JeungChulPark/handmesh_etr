# 원 입력 ↔ PRD 정합성 점검 (Reconciliation)

원 입력명: **Depth 기반 3D Joint Refinement Module (handmesh_etr, 2026-06-08)**

대상 문서:
- `prd.md`
- `addendum.md`

---

## 1. Objective

| 원 입력 | PRD 표현 | 판정 |
|---|---|---|
| 기존 Recon Hand Tracking 결과를 활용해 Depth 기반 3D Joint Refinement Module 추가 | §1 Vision, §3 Glossary(Depth Refinement / Refinement Module), §4.3 | ✅ 충실 반영. "교체하지 않고 후처리로 덧붙임", "모델 재학습 안 함"으로 *활용* 의도를 강화 표현. |

---

## 2. User Stories

| 원 입력 | PRD 표현 | 판정 |
|---|---|---|
| (1) XR user — 가상 객체 정렬 | UJ-1(Jin), §2.1 JTBD, §1 Vision | ✅ 반영 |
| (2) Mobile user — 실시간, latency < 50ms | §1, §8 NFR-1, SM-5(<50ms), UJ는 직접 persona 없음 | ⚠️ **부분 distortion**: "실시간/지연" 의도는 NFR·SM으로 보존됐으나, 원래 *모바일 최종 사용자*가 주체였던 스토리가 PRD에서는 "제품 통합자/개발자"(Mina, §2.1 4번째 JTBD) 관점으로 치환됨. 50ms는 정확히 보존. |
| (3) Researcher — Android·iOS 크로스플랫폼 | UJ-3(Dr. Park), §2.1, §부록 평가 하니스 | ✅ 반영. "크로스플랫폼 동작 + 정량 입증"으로 확장. |

---

## 3. Functional Requirements

| 원 FR | PRD 매핑 (부록 A) | 판정 |
|---|---|---|
| FR-1 RGB 입력 영상 처리 | FR-1 | ✅ |
| FR-2 Hand Joint Detection | FR-2 (Recon 불변) | ✅ |
| FR-3 Depth Acquisition (Android Monocular / iPhone LiDAR) | FR-3 (플랫폼 적응형 통합) | ✅ 반영 + addendum A에 모델 후보 보존 |
| FR-4 Depth Refinement | FR-4 + FR-5로 **분할** | ✅ 정당한 분할(샘플링/거부 vs metric 재구성). 누락 없음 |
| FR-5 3D Joint Reconstruction | FR-5 (Root-Depth Alignment) | ✅ |
| FR-6 Temporal Filtering | FR-6 | ✅ + addendum C(1-Euro) 보존 |
| **FR-7 Unity Visualization** | — (v2 이연), 출력 API만 PRD FR-7로 대체 | ⚠️ **명시적 이연**됨(§5, §6.2, 부록 A, addendum H). 단, PRD의 "FR-7"이 원 FR-7과 **번호는 같지만 의미가 다름**(원=시각화, PRD=출력 API). 의도적 처리이나 혼동 위험. |
| **FR-8 XR Integration** | — (v2 이연) | ⚠️ **명시적 이연**됨(§5, §6.2, 부록 A, addendum H). silent drop 아님. |

> 종합: 원 FR-1~6은 완전 보존(FR-4는 2개로 분할). FR-7/FR-8은 **silent drop이 아니라 명시적으로 v2 이연**되었고, 그 자리에 "출력 API"(원 입력에 없던 신규 FR-7)가 들어감.

---

## 4. Non-Functional Requirements

| 원 NFR | PRD 매핑 | 판정 |
|---|---|---|
| NFR-1 30 FPS 이상 | NFR-1 (≥30 FPS + <50ms 분리) | ✅ |
| NFR-2 Memory < 1GB | NFR-2 (<1GB) | ✅ |
| NFR-3 Mobile Device Support | NFR-3 (구체 기기 + ASSUMPTION) | ✅ |
| NFR-4 Unity AR Foundation Support | NFR-4 | ✅ |

PRD 추가분: **NFR-5 모델 불변성**, **NFR-6 정확도 회귀 금지** — 원 입력에 없으나 Objective("기존 결과 활용")에서 파생된 **정당한 추가**.

---

## 5. Success Metrics

| 원 지표 | PRD 매핑 | 판정 |
|---|---|---|
| Mean Joint Error 감소 | SM-1 Z-error 감소 / SM-6 fingertip | ✅ (구체화) |
| MPJPE 감소 | SM-2 MPJPE 감소 | ✅ |
| Jitter 감소 | SM-3 Jitter 감소 | ✅ |
| FPS 유지 | SM-4 ≥30 FPS / SM-5 <50ms | ✅ |

PRD 추가분: SM-C1~C3 Counter-metrics(lag/가림 왜곡/메모리·발열) — 원 입력 없음, **정당한 품질 가드레일 추가**.
주의: SM 절대 목표 수치(30%/20%/40%)는 원 입력에 없고 PRD가 `[ASSUMPTION]`으로 도입 — 베이스라인 확정 전 잠정치임이 명시됨(§Q2).

---

## 6. Deliverables

| 원 산출물 | PRD 매핑 | 판정 |
|---|---|---|
| System Requirement | §9 Platform & Hardware, §10 Integration | ✅ (System Requirement 절 명칭 부재) |
| Functional Requirement | §4 Features (FR-1~7) | ✅ |
| Non Functional Requirement | §8 Cross-Cutting NFRs | ✅ |
| Acceptance Criteria | 부록 B (AC-1~9) | ✅ |

⚠️ "System Requirement" 산출물이 **명시적 단일 절로 분리되지 않고** §9/§10에 분산됨. 내용은 커버되나 산출물 라벨 추적성 약함.

---

## 7. PRD가 추가한 요구사항 (원 입력에 없음) — 정당성

- **NFR-5 모델 불변성 / NFR-6 회귀 금지** — Objective의 "기존 결과 활용"에서 직접 파생. **정당**.
- **SM-C1~C3 Counter-metrics** — 보정이 역효과를 내지 않도록 하는 가드레일. **정당**.
- **SM 절대 목표 수치(30/20/40%)** — `[ASSUMPTION]` 태그 + Open Q2로 잠정 처리. **정당하나 미확정**.
- **비-Pro iPhone Monocular 폴백** — 원 입력은 "iPhone=LiDAR"만 명시. PRD가 LiDAR 미탑재 기기 현실을 반영해 폴백 추가. **정당**(Open Q3로 확정 대기).
- **출력 API(신 FR-7)** — 원 FR-7/FR-8 이연으로 생긴 최소 전달 표면. **정당**.

---

## 8. 핵심 갭/주의 (요약)

1. **User Story (2) 주체 치환**: 원래 *모바일 최종 사용자* 스토리가 PRD에선 개발자/통합자 JTBD로 흡수됨(50ms·실시간 의도 자체는 NFR/SM에 보존). 정성적 의도 일부 약화.
2. **FR-7 번호 재사용 혼동**: PRD FR-7 = "출력 API"로 원 FR-7(Visualization)과 번호 충돌. 부록 A에서 매핑은 명시되나 추적 시 혼동 가능.
3. **FR-7/FR-8 이연은 silent drop 아님**: §5·§6.2·부록 A·addendum H 4곳에서 명시적 v2 이연 처리됨. ✅
4. **"System Requirement" 산출물 라벨 부재**: 내용은 §9/§10에 분산 커버되나 독립 절·라벨 없음 → 산출물 체크리스트 추적성 저하.
5. **silent drop 없음**: 원 입력의 모든 FR/NFR/메트릭/산출물 항목은 반영 또는 명시적 이연으로 계정됨. 정성적 손실은 (1)의 사용자 주체 치환이 유일하게 주목할 지점.
