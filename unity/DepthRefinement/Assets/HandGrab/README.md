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
