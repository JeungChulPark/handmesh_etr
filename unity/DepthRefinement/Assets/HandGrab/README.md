# HandGrab — 학습 모델로 Unity에서 가상 오브젝트 잡기/돌리기

hybrid lifter(FastViT-SA12 replay_ft)의 실시간 3D 손 관절을 UDP로 받아,
핀치(엄지-검지)로 sphere/cube 등을 **잡고 · 옮기고 · 돌리고 · 던지는** 데모.

> **iOS(iPhone/iPad) 온디바이스 버전**은 여기가 아니라
> `unity/DepthRefinement/Assets/Runtime/Grab/` — PC 스트리밍 없이 기기에서 도는
> hybrid_B 모델로 동일한 그랩 인터랙션을 한다. 빌드 절차 포함:
> `Assets/Runtime/Grab/HAND_GRAB_IOS.md`. 이 폴더(UDP)는 데스크톱/에디터 테스트용.

```
ZED / 캡처 리플레이 ──> unity_stream_hand.py (PyTorch 추론) ──UDP 9750──> Unity
iPad RGB-D 스트림  ──> infer_ipad_stream.py (PyTorch 추론) ──UDP 9750──> Unity
                                                 │              │
                                                 └─UDP 9760────┤ (JPEG 영상)
                                                                │
                                              HandStreamReceiver (수신+스무딩)
                                              VideoStreamReceiver(카메라 영상 배경)
                                              HandSkeleton      (21관절 시각화)
                                              PinchGrabber      (잡기/회전/던지기)
```

> **iPad를 센서로 쓸 때** — 이 씬이 곧 "서버 쪽 유니티"다. 서버를
> ```bash
> python infer_ipad_stream.py --udp 127.0.0.1:9750 --udp-video 127.0.0.1:9760
> ```
> 으로 실행하면 iPad 영상이 유니티 배경에 깔리고(HandGrabDemo의 `Show Video`,
> 기본 켜짐) 그 위에서 3D 손으로 오브젝트를 잡는다. 영상 스트림이 없으면
> 배경은 자동으로 숨겨지므로 ZED 워크플로에는 영향이 없다.

## 실행 순서

**1. Python 스트리머** (이 repo 루트에서):

```bash
# ZED 라이브
python unity_stream_hand.py

# 카메라 없이 — 홀드아웃 세션 리플레이로 테스트
python unity_stream_hand.py --source replay --replay_dir rgbd_captures_04

# 다른 PC의 Unity로 보낼 때
python unity_stream_hand.py --udp <unity-pc-ip>:9750
```

**2. Unity** (2021.3+ / 6000 모두 지원):

1. `unity/HandGrab/` 폴더를 프로젝트의 `Assets/` 아래에 복사
2. 새 빈 씬에서 빈 GameObject 생성 → `HandGrabDemo` 컴포넌트 추가
3. Play — 나머지(카메라, 조명, 테이블, sphere/cube/capsule, 핸드 리그)는
   코드가 전부 생성함

## 조작

| 동작 | 방법 |
|---|---|
| 잡기 | 엄지 끝과 검지 끝을 3.5cm 이내로 모으기 (핀치) |
| 옮기기 | 핀치 유지한 채 손 이동 |
| **돌리기** | 핀치 유지한 채 손목 비틀기 (손바닥 프레임 회전이 그대로 적용) |
| 놓기/던지기 | 핀치 풀기 — 놓는 순간 손 속도가 오브젝트에 전달됨 |

## 녹화 · 거리 표시

- **Game 창 자동 녹화** (`GameViewRecorder`, `HandGrabDemo.recordGameView` 기본 on) —
  Play를 누르는 순간부터 멈출 때까지 `unity/DepthRefinement/Recordings/HandGrab_<날짜_시각>.mp4`로
  저장. 영상 배경·손·오브젝트·거리 숫자까지 화면 그대로 담기고, 좌상단 `● REC mm:ss.cc f=N`
  스탬프로 문제 구간을 시간으로 찾을 수 있다. 실제 시간 기준 CFR(기본 30fps)이라 재생 속도 = 실제 속도.
  - **ffmpeg 필요**: `brew install ffmpeg` (없으면 `*_frames/` JPEG 시퀀스로 대신 저장)
  - 영상이 위아래 뒤집혀 나오면 `GameViewRecorder.verticalFlip`을 On/Off로 바꿀 것
  - Play 중 Game 창 크기를 바꾸면 새 파일로 이어서 저장
- **바닥(Table) 제거** — `HandGrabDemo.showTable` 기본 off (오브젝트는 공중에 떠 있음)
- **부호 있는 거리** (`GrabProximityIndicator.signedByDepth` 기본 on) — 카메라 시선 방향 기준으로
  손(핀치)이 오브젝트 중심보다 **카메라에서 먼 쪽(사용자 몸 쪽)이면 `+`**, 오브젝트 중심을
  넘어 **카메라 쪽으로 가면 `−`** (몸 쪽에서 손을 뻗으면 `+ → −`).
  두 번째 줄 `depth ±x cm`는 오브젝트 중심과 손의 깊이 차이 자체 (0 = 같은 깊이).

## 잡기 동작 보정

- **놓을 때 밀림 없음** (`PinchGrabber.firmPinchDistance`, 기본 4.5cm) — 손가락이 벌어지기
  시작하면 물체는 그 자리에 멈추고, 놓기가 확정되면 **마지막으로 꽉 쥐고 있던 위치**에 그대로 놓인다.
  놓기 전에 다시 쥐면 멈춘 위치에서 이어서 따라온다.
- **손이 앞에 있으면 반투명** (`HandGrabDemo.fadeWhenHandInFront` → `HandOcclusionFade`) —
  손 관절 중 하나라도 물체 중심보다 카메라에 가깝고 화면에서 겹치면 불투명도 `fadedAlpha`(0.35)로.
  거리 표시용 와이어프레임 박스는 그대로 보인다.
- **잡고 있는 동안 다른 물체 박스는 회색** — 박스 색(회색→노랑→초록)은 "지금 핀치하면 잡힌다"는
  뜻인데, 이미 하나를 쥐고 있으면 다른 물체는 잡을 수 없으므로 색을 바꾸지 않는다.
- **자동 복귀** (`HandGrabDemo.autoReturn`, 기본 off) — 켜면 놓은 뒤 `returnDelay`(3초)가 지나면 가운데 줄의
  제자리로 부드럽게 돌아간다. 화면 밖으로 나갔거나 카메라에 `minCameraDistance`(12cm)보다 가까우면
  `lostReturnDelay`(0.5초) 뒤 바로 복귀. `R`(재배치)도 이 제자리를 기준으로 한다.

## 튜닝 포인트

- `HandStreamReceiver.mirrorX` — 전면 카메라면 켜두는 게 자연스러움 (기본 on)
- `HandStreamReceiver.minCutoff/beta` — One Euro 스무딩 (지터↔지연 트레이드오프)
- `PinchGrabber.grabDistance/releaseDistance` — 핀치 임계값 (히스테리시스)
- `PinchGrabber.grabRadius` — 핀치 지점에서 오브젝트 표면까지 최대 거리
- 좌표계: 패킷은 카메라 프레임(m), Unity 쪽에서 y-플립. `HandRig`의
  Transform을 움직이면 손 전체를 씬 안에서 재배치/스케일 가능

## 패킷 포맷 (UDP JSON)

```json
{"t": 1783500000.1234, "det": 1, "fps": 21.3, "j": [x0,y0,z0, ..., x20,y20,z20]}
{"t": 1783500000.1567, "det": 0}
```

관절 순서는 MediaPipe/FreiHAND (0 wrist, 4/8/12/16/20 = 손끝), 단위 m,
카메라 프레임(x 오른쪽, y 아래, z 전방) 절대좌표 (root = 센서 손목 depth).
