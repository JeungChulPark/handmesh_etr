# 온디바이스 핸드 그랩 — iPhone/iPad에서 가상 오브젝트 잡기

`unity/HandGrab/`(PC→UDP 데모)의 iOS 버전. PC 스트리밍 없이 **기기 위에서 도는
hybrid_B 모델**(Apple Vision 2D + Sentis 백본 Z + LiDAR 루트)의 3D 관절로
가상 오브젝트를 **잡고 · 옮기고 · 돌리고 · 던진다**.

```
ARKit RGB + LiDAR ─ DualStreamHandProvider(crop-only) ─┐
Apple Vision 손 랜드마크 ─ HandVisionBridge/HandBboxBridge ─┼─ HybridBHandProvider ── AbsJoints
                                                        │        (OnPose 이벤트)
                                          HandSphereDriver (관절/본 시각화)
                                          HandGrabController (핀치 잡기)   ← 이 폴더
```

## 구성 요소 (이 폴더)

| 파일 | 역할 |
|---|---|
| `HandGrabController.cs` | 핀치(엄지4–검지8) 판정 + 손바닥 프레임 회전 + 잡기/놓기/던지기. `HybridBHandProvider.OnPose` 이벤트로 구동되므로 Script Execution Order 설정 불필요. 데모 오브젝트(큐브/구/캡슐, 6cm)를 첫 포즈 시점에 카메라 앞 40cm에 자동 생성. |
| `GrabbableObject.cs` | 잡을 수 있는 오브젝트 마커. Collider 필수, Rigidbody는 선택(잡는 동안 kinematic, 놓으면 손 속도 전달 → 던지기). 잡는 동안 노란 틴트. |

씬(`DepthRefinement.unity`)에는 **`[Grab]` GameObject로 이미 배치·와이어링 완료**
(_hybrid → RGBViewer의 HybridBHandProvider, _dual → DualStreamHandProvider,
_cameraTransform → Main Camera). 직접 만든 오브젝트를 잡고 싶으면 Collider 있는
아무 오브젝트에 `GrabbableObject`를 붙이고 `[Grab]`의 *Spawn Demo Objects*를 꺼도 된다.

## 조작

| 동작 | 방법 |
|---|---|
| 잡기 | 엄지 끝–검지 끝 3.5cm 이내로 모으기 (오브젝트에서 10cm 이내) |
| 옮기기 | 핀치 유지한 채 손 이동 |
| 돌리기 | 핀치 유지한 채 손목 비틀기 (손바닥 프레임 회전 적용) |
| 놓기/던지기 | 핀치 풀기 (5.5cm 히스테리시스) — 놓는 순간 손 속도 전달 |

트래킹이 0.4초 이상 끊기면 잡고 있던 오브젝트는 제자리에 놓인다(던지기 없음).
오브젝트가 카메라에서 2.5m 이상 멀어지면 다시 앞으로 리스폰.

## 튜닝 (Inspector — `[Grab]`)

- `Grab Dist / Release Dist` — 핀치 임계값(m). 모델 지터가 크면 간격을 넓힐 것
- `Smoothing` — 핀치 포즈 지수 스무딩 (0 = 반응 빠름·지터, 0.9 = 부드러움·지연)
- `Grab Radius` — 핀치 지점→오브젝트 표면 최대 거리
- `Spawn Distance / Object Size` — 데모 오브젝트 배치

## iOS 빌드 → Xcode → 기기 실행

**요구사항**: LiDAR 탑재 기기 (iPhone 12 Pro 이상 Pro 모델 / iPad Pro 2020+),
iOS 15+, macOS + Xcode 15+, Unity 6000.3 (iOS Build Support 모듈 포함).

1. **Unity에서 Xcode 프로젝트 생성**
   - `File ▸ Build Profiles ▸ iOS` → Switch Platform
   - Scenes In Build에 `DepthRefinement.unity`만 있는지 확인
   - `Build` → 출력 폴더 지정 (예: `build/ios/`)
   - 참고: 리눅스/윈도우 에디터에서도 Xcode **프로젝트 생성까지는** 가능 —
     생성된 폴더를 통째로 Mac으로 복사해서 이후 단계 진행
2. **Xcode에서 열기** — `Unity-iPhone.xcodeproj`
   - `Unity-iPhone` 타깃 ▸ *Signing & Capabilities* ▸ Team 선택
     (Bundle Identifier `com.ETRI.DepthRefinement`는 이미 설정됨,
     필요하면 팀에 맞게 변경)
   - 카메라 권한 문구(`NSCameraUsageDescription`)는 Unity Player Settings에서
     이미 주입됨 ("Used for AR hand tracking")
   - `HandPoseVision.swift`(Apple Vision 손 검출 플러그인)는 Unity가 자동으로
     UnityFramework에 포함시킴 — 별도 설정 불필요
3. **기기 연결 후 Run(⌘R)** — 처음엔 기기에서
   *설정 ▸ 일반 ▸ VPN 및 기기 관리*에서 개발자 신뢰 필요할 수 있음
4. **확인**: 카메라에 손이 잡히면 관절 스피어 + 본이 뜨고, 카메라 앞에 떠 있는
   큐브/구/캡슐을 핀치로 잡아 옮기고 돌릴 수 있다. 잡히면 오브젝트가 노랗게 변함.

### 흔한 문제

- **관절은 뜨는데 잡히지 않음** — 데모 오브젝트가 시야 뒤에 있을 수 있음.
  기기를 든 채 몸을 돌려 처음 실행 방향을 보거나, 2.5m 리스폰을 기다릴 것
- **손이 안 잡힘** — 밝은 곳에서 손을 카메라에서 30–60cm에 둘 것.
  Occlusion Manager의 Environment Depth Mode가 **Best**(NEURAL)인지 확인
  (PERFORMANCE는 루트가 ~90mm 틀어짐, docs/ZED_HYBRID_B_DEPLOY.md)
- **핀치가 튐** — `Smoothing`을 0.5–0.7로 올리거나 `Release Dist`를 키울 것
