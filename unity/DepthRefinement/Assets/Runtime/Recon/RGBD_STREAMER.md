# 서버 추론 모드 — iPad RGB-D 스트리밍 (RgbdStreamer)

iPad/iPhone(Unity AR Foundation)에서 **영상(RGB) + LiDAR 뎁스**를 PC로 보내고,
PC가 hybrid lifter(FastViT-SA12 replay_ft)로 추론해서 보여주는 파이프라인.
기존 **온디바이스(Sentis) 추론 모드는 그대로 유지**되며, `HandPoseModeController`로
두 모드를 런타임에 전환한다.

```
┌─ iPad (Unity) ──────────────────────┐   ┌─ 서버 PC ─────────────────────────────┐
│ RgbdStreamer (30fps)                │   │ infer_ipad_stream.py (PyTorch 추론)   │
│   RGB(JPEG) + LiDAR(uint16mm)      ─┼──►│   ├─ OpenCV 창(오버레이+START/STOP)   │
│   + intrinsics + 회전값(rotK)       │TCP│   ├─UDP 9750 관절─► Unity HandGrab    │
│                                     │9776   ├─UDP 9760 영상─► (가시화+인터랙션) │
│ (선택) ServerHandProvider          ◄┼───┤   └─UDP 9751 회신: 기본 OFF           │
└─────────────────────────────────────┘UDP└───────────────────────────────────────┘
```

**기본 배선** = iPad는 센서 역할만(30fps 전송), 추론·가시화·인터랙션은 전부 서버 PC
(Python 추론 + Unity HandGrab 씬). iPad로의 관절 회신은 기본 꺼짐(`--udp-back 9751`로
다시 켤 수 있음).

## 모드

| 모드 | 추론 위치 | 활성 컴포넌트 |
|---|---|---|
| **OnDevice** (기존) | 기기(Sentis, hybrid_B/ds_anchor_gate) | `HybridBHandProvider` / `DualStreamHandProvider` |
| **Server** (신규) | PC(PyTorch, replay_ft) | `RgbdStreamer` + `ServerHandProvider` |

`HandSphereDriver` / `HandGrabController`는 **enabled 상태인 프로바이더를 자동 선택**
(우선순위: server > hybrid B > dual-stream)하므로 씬 수정 없이 두 모드에서 동작한다.

## 씬 설정 (한 번만)

기존 RGB-D 뷰어/그랩 씬(RGBD_VIEWER_HIERARCHY.md 참고)에 컴포넌트 3개 추가:

1. 아무 GameObject에 **`RgbdStreamer`** 추가
   - `Host` = 서버 PC의 IP (예: `192.168.0.10`), `Port` = 9776
   - `Stream Fps` 15, `Rgb Max Size` 640, `Jpg Quality` 80 (기본값이면 ~2.4MB/s)
   - `Auto Start`는 켜두면 Server 모드 진입 시 자동 시작
2. 같은 곳에 **`ServerHandProvider`** 추가 (`Port` 9751)
3. 같은 곳에 **`HandPoseModeController`** 추가
   - `Mode` 기본값 = OnDevice (기존 동작 그대로)
   - UI 버튼의 OnClick에 `ToggleMode` / `SetServer` / `SetOnDevice` 바인딩
   - 상태 라벨에는 `Status` 문자열 사용

**전송 확인 UI (선택)** — 기존 뷰어 Canvas를 그대로 활용:
- 기존 UI(ARCameraBackground 카메라 화면, DepthView 뎁스 뷰)는 Server 모드에서도
  그대로 동작한다 — 모드 컨트롤러는 추론 컴포넌트만 전환한다.
- **`RgbdStreamerStatusLabel`**: Canvas에 Text 하나 추가하고 붙이면
  `● SENDING 29.8 fps / sent 1234 dropped 2` 식으로 전송 상태 표시
  (초록=전송중, 노랑=서버 PAUSED, 회색=미접속). RgbdRecorderStatusLabel과 같은 패턴.
- **`RgbdStreamPreview`**: 서버로 나가는 **바로 그 프레임**을 PiP로 표시
  (rotK 보정해 upright로 회전 — 서버가 rotate_frame 후 보는 그림과 동일).
  아무 GameObject에 추가만 하면 좌하단에 자동 생성되고, 기존 Canvas의 RawImage를
  `Target`에 할당하면 그 자리에 그린다. 전송 중이 아닐 때는 자동으로 숨는다.

## 실행 (Server 모드)

**1. PC에서 서버 실행** (repo 루트):

```bash
python infer_ipad_stream.py                      # :9776 수신, START 버튼 대기
                                                 # (관절→9750, 영상→9760 로컬 Unity로 기본 전달)
python infer_ipad_stream.py --autostart          # 버튼 없이 바로 전송 시작
python infer_ipad_stream.py --udp-back 9751      # iPad에서도 가시화할 때만
python infer_ipad_stream.py --selftest           # iPad 없이 루프백 배선 테스트
```

**START/STOP 버튼 (원격 제어)**: GUI 창 우상단 버튼(또는 스페이스바)으로 전송을
제어한다. 같은 TCP 소켓 역방향으로 1바이트 명령('S'/'P')이 내려가며 —
- **START** → iPad가 영상 전송 시작 + (HandPoseModeController의 `Remote Mode
  Switch`가 켜져 있으면) 자동으로 **Server 모드로 전환**
- **STOP** → 전송 중지 + **OnDevice 모드로 복귀**
- 이를 위해 iPad는 OnDevice 모드에서도 제어 연결(TCP)만 유지한다(프레임은 안 감).
  원치 않으면 `Remote Mode Switch`를 끄면 OnDevice 모드에서 연결도 끊는다.
- GUI 실행 시 기본은 PAUSED 상태(버튼을 눌러야 전송), 헤드리스(--no-gui)는 바로 시작.

- 방화벽에서 TCP 9776 인바운드 허용 필요. iPad와 PC는 같은 네트워크에 있어야 한다.
- iPad로의 관절 회신은 **기본 꺼짐** — 기기에서도 가시화/인터랙션이 필요할 때만
  `--udp-back 9751`을 주고 씬에 `ServerHandProvider`를 둔다(없어도 무해, idle).
- iOS **Local Network 권한**: `Assets/Editor/IOSPlistPostprocessor.cs`가 빌드 시
  `NSLocalNetworkUsageDescription`을 Info.plist에 자동 주입한다. 첫 접속 시 뜨는
  권한 팝업에서 허용할 것 — 거부했으면 설정 > 개인정보 보호 > 로컬 네트워크에서 재허용.

**2. iPad 앱에서** Server 모드로 전환(버튼 또는 인스펙터) → PC 창에 오버레이가 뜨고,
기기에서는 `ServerHandProvider`가 받은 관절로 스피어/그랩이 그대로 동작한다.

**종료**: 창 우상단 **QUIT** 버튼, `q`/ESC 키, 또는 창의 X 버튼 — 어느 쪽이든 서버가
완전히 종료되며(창이 다시 생기지 않음), 종료 직전 'P'를 보내 iPad를 OnDevice 모드로
돌려놓는다.

## 가상 오브젝트 인터랙션 (Server 모드에서도 동일)

OnDevice 모드에서 쓰던 그랩 스크립트가 **씬 수정 없이 Server 모드에서도 동작**한다 —
관절 소스만 `ServerHandProvider`(UDP 9751 회신)로 바뀔 뿐, 이후 파이프라인은 동일:

- **`HandGrabController`** (권장, 핀치로 잡기/돌리기/던지기): enabled 소스 자동 선택
  (server > hybrid B > dual). `Spawn Demo Objects`가 켜져 있으면 첫 포즈에서
  cube/sphere/capsule이 카메라 앞에 생성된다. 씬에 없으면 아무 GameObject에
  Add Component만 하면 됨(전부 자동 연결).
- **`HandSphereDriver`** (관절 스피어/본 시각화): 동일하게 자동 선택.
- **`SimpleHandGrab`** (미니 데모): 포즈 게이트가 server/hybrid 중 enabled 쪽을 따름.

전제: 서버의 `--udp-back`이 켜져 있고(기본 9751), 씬에 `ServerHandProvider`가 있을 것.
서버 왕복 지연(~0.1s)이 있으므로 기기를 크게 움직이며 잡을 때는 OnDevice 모드보다
오브젝트가 약간 늦게 따라온다.

## 와이어 포맷 (little-endian, 프레임당 1패킷)

```
u32 magic 'RGBD' | u32 version=1 | u32 jpegSize | u32 depthSize
u32 rgbW | u32 rgbH | u32 depthW | u32 depthH
f32 fx,fy,cx,cy (센서 풀해상도 intrW x intrH 기준) | u32 intrW | u32 intrH
u32 rotK (k*90° CW로 이미지가 upright; Screen.orientation 매핑은
          DualStreamHandProvider.EffectiveRotation과 동일)
f64 tSec (기기 unix time)
[JPEG bytes, 센서 방향] [uint16 mm 뎁스, row-major depthH x depthW, 0=invalid]
```

서버는 뎁스를 RGB 해상도로 NEAREST 업샘플 → intrinsics 스케일 →
`rotate_frame(bgr, depth, K, rotK)`로 upright 정렬 후 추론한다
(`--rotate 0..3`으로 강제 오버라이드 가능).
