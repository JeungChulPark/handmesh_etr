"""iPad/iPhone (Unity RgbdStreamer.cs) -> PC live hand-pose inference server.

Receives synchronized RGB + LiDAR depth frames over TCP from the Unity app
(unity/DepthRefinement, `RgbdStreamer` component), runs the deploy-best HYBRID
lifter (FastViT-SA12 replay_ft) on each frame and shows the overlay — the same
model + per-frame path as unity_stream_hand.py, with the iPad as the sensor
instead of a ZED. This is the SERVER-inference mode; the on-device (Sentis)
mode needs no PC and is switched with HandPoseModeController in the Unity app.

Wire format, one little-endian packet per frame (must match RgbdStreamer.cs):
    u32 magic 'RGBD' | u32 version | u32 jpegSize | u32 depthSize
    u32 rgbW | u32 rgbH | u32 depthW | u32 depthH
    f32 fx,fy,cx,cy (at intrW x intrH) | u32 intrW | u32 intrH
    u32 rotK (k*90deg CW makes the image upright) | f64 tSec (device clock)
    v2+: f32 camPx,camPy,camPz | f32 camQx,camQy,camQz,camQw | u32 poseOk
         (ARKit camera pose in Unity WORLD space; poseOk=1 while ARSession tracks)
    [jpegSize bytes JPEG] [depthSize bytes uint16 MILLIMETERS, row-major, 0=invalid]

The device camera pose is forwarded in the outgoing UDP joint packet as "cp" (position
[x,y,z]) and "cq" (quaternion [x,y,z,w]) on EVERY frame (also det=0), so the desktop
Unity HandGrab scene can world-lock its virtual camera to the physical iPad
(StreamedCameraDriver.cs on the HandRig; set HandGrabDemo.followDeviceCamera).

The inferred joints are streamed out as the unity_stream_hand.py UDP JSON packet.
DEFAULT wiring = desktop Unity on this PC does the visualization + interaction:
joints go to 127.0.0.1:9750 (--udp) and the camera frames to 127.0.0.1:9760
(--udp-video) for the unity/HandGrab scene. Sending joints back to the iPad for
on-device rendering is OFF by default (enable with --udp-back 9751 +
ServerHandProvider.cs when the device should also render/interact).

The GUI window has a START/STOP button (space bar works too). It sends a 1-byte
command back down the TCP link ('S'/'P'): START makes the device begin sending
frames — and, when HandPoseModeController's remoteModeSwitch is on, auto-switches
the app into Server mode; STOP pauses the stream and returns it to OnDevice mode.
With a GUI the server starts PAUSED (press START); headless starts enabled.

Examples
--------
  python infer_ipad_stream.py                        # GUI + joints/video -> local Unity
  python infer_ipad_stream.py --udp none --udp-video none   # OpenCV overlay only
  python infer_ipad_stream.py --udp-back 9751        # + reply joints to the iPad too
  python infer_ipad_stream.py --selftest             # loopback wiring test, no iPad
"""
import os
import json
import time
import socket
import struct
import argparse
import threading

import numpy as np
import cv2
import torch

from utils import load_cfg
from train_hybrid import HybridLifter
from infer_rgbd_femtobolt import HandDetector, project, draw_skeleton
from infer_rgbd_captures import rotate_frame
from unity_stream_hand import infer_joints

MAGIC = 0x44424752                    # "RGBD" little-endian
HDR_FMT = "<8I4f3Id"
HDR_SIZE = struct.calcsize(HDR_FMT)   # 68 bytes
POSE_FMT = "<7fI"                     # v2 extension: cam pos xyz, quat xyzw, poseOk
POSE_SIZE = struct.calcsize(POSE_FMT)  # 32 bytes


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf.extend(chunk)
    return bytes(buf)


class IPadSource:
    """TCP server for RgbdStreamer packets. A background thread decodes incoming frames
    and keeps only the LATEST one (stale frames are dropped, so inference always runs on
    the newest frame instead of building up latency). Survives client reconnects.

    Control channel (server -> device, same TCP socket): 1 byte, 'S' = start sending
    frames, 'P' = pause. The current state is pushed on connect and on every toggle;
    on the device RgbdStreamer gates capture on it and HandPoseModeController can
    auto-switch the app between OnDevice/Server modes."""

    def __init__(self, host="0.0.0.0", port=9776, start_enabled=True):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self._srv.settimeout(1.0)
        self._cond = threading.Condition()
        self._latest = None
        self._running = True
        self._enabled = start_enabled
        self._conn = None
        self.client_ip = None
        self.frames_in = 0
        self.frames_dropped = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        print(f"[ipad] listening on tcp://{host}:{port}", flush=True)

    def _serve(self):
        while self._running:
            try:
                conn, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.client_ip = addr[0]
            self._conn = conn
            print(f"[ipad] client connected: {addr[0]}:{addr[1]}", flush=True)
            try:
                conn.sendall(b"S" if self._enabled else b"P")   # push current state
                self._read_client(conn)
            except (ConnectionError, OSError, struct.error) as e:
                print(f"[ipad] client disconnected ({e})", flush=True)
            finally:
                self._conn = None
                conn.close()

    def _read_client(self, conn):
        while self._running:
            hdr = recv_exact(conn, HDR_SIZE)
            (magic, ver, jsz, dsz, _rw, _rh, dw, dh,
             fx, fy, cx, cy, iw, _ih, rotk, t_dev) = struct.unpack(HDR_FMT, hdr)
            if magic != MAGIC:
                raise ConnectionError(f"bad magic 0x{magic:08x} (stream desync)")
            pose = None                       # (pos[3], quat[4]) in Unity world space
            if ver >= 2:
                px, py, pz, qx, qy, qz, qw, pose_ok = struct.unpack(
                    POSE_FMT, recv_exact(conn, POSE_SIZE))
                if pose_ok:
                    pose = ((px, py, pz), (qx, qy, qz, qw))
            jpg = recv_exact(conn, jsz)
            dep = recv_exact(conn, dsz) if dsz else None

            bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            if dep is not None and dw * dh * 2 == dsz:
                d = np.frombuffer(dep, "<u2").reshape(dh, dw).astype(np.float32) * 1e-3
                depth = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)  # -> RGB res
            else:
                depth = np.zeros((H, W), np.float32)
            # intrinsics arrive at the full sensor resolution iw x ih; the JPEG is the same
            # image downscaled (no crop) -> uniform scale
            s = W / float(iw) if iw else 1.0
            K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]], np.float32)

            with self._cond:
                if self._latest is not None:
                    self.frames_dropped += 1
                self._latest = (bgr, depth, K, int(rotk), float(t_dev), pose)
                self.frames_in += 1
                self._cond.notify_all()

    @property
    def streaming(self):
        return self._enabled

    def set_streaming(self, on):
        """Toggle the device's frame flow (and, with remoteModeSwitch, its inference mode)."""
        self._enabled = bool(on)
        conn = self._conn
        if conn is not None:
            try:
                conn.sendall(b"S" if self._enabled else b"P")
            except OSError:
                pass                      # reader thread will notice the dead socket
        print(f"[ipad] streaming {'STARTED' if self._enabled else 'PAUSED'}"
              f"{' (no device connected yet)' if conn is None else ''}", flush=True)

    def read(self, timeout=1.0):
        """Newest undecoded frame, or None if nothing arrived within timeout."""
        with self._cond:
            if self._latest is None:
                self._cond.wait(timeout)
            got, self._latest = self._latest, None
        return got

    def close(self):
        self._running = False
        conn = self._conn
        if conn is not None:
            try:
                conn.sendall(b"P")   # put the device back to OnDevice mode on shutdown
            except OSError:
                pass
        try:
            self._srv.close()
        except OSError:
            pass


WIN = "RGB-D hand pose (iPad)"


def draw_buttons(img, streaming):
    """Draw START/STOP + QUIT (top-right); returns their rects for hit-testing."""
    h, w = img.shape[:2]
    bx1, by1, bx2, by2 = w - 330, 12, w - 142, 64              # START/STOP
    qx1, qy1, qx2, qy2 = w - 130, 12, w - 12, 64               # QUIT
    color = (40, 40, 200) if streaming else (40, 160, 40)      # red = stop, green = start
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
    cv2.putText(img, "STOP" if streaming else "START", (bx1 + 18, by1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.rectangle(img, (qx1, qy1), (qx2, qy2), (60, 60, 60), -1)
    cv2.rectangle(img, (qx1, qy1), (qx2, qy2), (255, 255, 255), 2)
    cv2.putText(img, "QUIT", (qx1 + 20, qy1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return (bx1, by1, bx2, by2), (qx1, qy1, qx2, qy2)


def selftest_feeder(port, frames=6):
    """Pretend to be the iPad: push synthetic packets over loopback TCP."""
    time.sleep(0.3)
    conn = socket.create_connection(("127.0.0.1", port))
    bgr = np.full((480, 640, 3), 40, np.uint8)
    cv2.circle(bgr, (320, 240), 60, (0, 120, 255), -1)
    jpg = cv2.imencode(".jpg", bgr)[1].tobytes()
    depth = np.full((192, 256), 500, np.uint16)          # 0.5 m everywhere
    hdr = struct.pack(HDR_FMT, MAGIC, 2, len(jpg), depth.nbytes, 640, 480, 256, 192,
                      600.0, 600.0, 320.0, 240.0, 640, 480, 0, time.time())
    pose = struct.pack(POSE_FMT, 0.1, 1.2, -0.3, 0.0, 0.0, 0.0, 1.0, 1)  # v2 camera pose
    for _ in range(frames):
        conn.sendall(hdr + pose + jpg + depth.tobytes())
        time.sleep(0.05)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="mobrecon_ckpt/hybrid_sa12_replay_ft_bb_uv2d/best.pt")
    ap.add_argument("--cfg", default="./configs_rgbd.yaml")
    ap.add_argument("--backbone", default="fastvit", choices=["densestack", "fastvit"])
    ap.add_argument("--fastvit_variant", default="fastvit_sa12")
    ap.add_argument("--mode", default="backbone", choices=["backbone", "locked"])
    ap.add_argument("--listen", default="0.0.0.0:9776", help="host:port for the iPad TCP stream")
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 1, 2, 3],
                    help="override the device-reported k*90CW upright rotation")
    ap.add_argument("--udp", default="127.0.0.1:9750",
                    help="forward joints to host:port for the desktop Unity HandGrab scene "
                         "(unity_stream_hand packet; 'none' = off)")
    ap.add_argument("--udp-video", dest="udp_video", default="127.0.0.1:9760",
                    help="send the camera frames as JPEG datagrams to host:port for the "
                         "Unity video background ('none' = off)")
    ap.add_argument("--udp-back", dest="udp_back", type=int, default=0,
                    help="UDP port ON THE DEVICE to send joints back to (0 = off; only "
                         "needed when the iPad itself should render/interact)")
    ap.add_argument("--save", default=None, help="dir to dump overlay PNGs + metrics.jsonl")
    ap.add_argument("--no-gui", dest="gui", action="store_false", help="headless")
    ap.add_argument("--autostart", action="store_true",
                    help="start with streaming ENABLED (GUI default: paused until the "
                         "START button / space is pressed; headless default: enabled)")
    ap.add_argument("--selftest", action="store_true",
                    help="loopback wiring test: no iPad, no GUI, exits after a few frames")
    args = ap.parse_args()
    if args.selftest:
        args.gui = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.cfg)
    model = HybridLifter(cfg, mode=args.mode, backbone=args.backbone,
                         fastvit_variant=args.fastvit_variant).to(device).eval()
    st = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(st.get("model_state_dict", st))
    print(f"[ipad] loaded {args.ckpt} (epoch {st.get('epoch','?')}, "
          f"MPJPE {st.get('best_test_mpjpe', float('nan')):.2f}mm)", flush=True)
    det = HandDetector()

    host, port = args.listen.rsplit(":", 1)
    # GUI: wait for the START button. Headless: no button, so stream immediately.
    start_enabled = args.autostart or not args.gui
    src = IPadSource(host, int(port), start_enabled=start_enabled)

    ui = {"btn": None, "quit_btn": None, "quit": False}

    def toggle_streaming():
        src.set_streaming(not src.streaming)

    def inside(rect, x, y):
        return rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if inside(ui["btn"], x, y):
            toggle_streaming()
        elif inside(ui["quit_btn"], x, y):
            ui["quit"] = True

    def handle_key():
        """Returns True when the user asked to quit (key, QUIT button, or closed window)."""
        key = cv2.waitKey(1) & 0xFF
        if ui["quit"] or key in (27, ord("q")):
            return True
        if key == ord(" "):
            toggle_streaming()
        try:    # window closed with the title-bar X: do NOT respawn it, shut down instead
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            return True
        return False

    if args.gui:
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, on_mouse)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_fixed = None
    if args.udp and args.udp.lower() not in ("none", "off", "0"):
        h, p = args.udp.rsplit(":", 1)
        udp_fixed = (h, int(p))
        print(f"[ipad] forwarding joints to udp://{h}:{p} (Unity HandGrab)", flush=True)
    udp_video = None
    if args.udp_video and args.udp_video.lower() not in ("none", "off", "0"):
        h, p = args.udp_video.rsplit(":", 1)
        udp_video = (h, int(p))
        print(f"[ipad] video feed to udp://{h}:{p} (Unity background)", flush=True)
    if args.udp_back:
        print(f"[ipad] replying joints to <device-ip>:{args.udp_back}/udp", flush=True)

    feeder = None
    max_frames = None
    if args.selftest:
        max_frames = 5
        feeder = threading.Thread(target=selftest_feeder, args=(int(port),), daemon=True)
        feeder.start()

    metrics_f = None
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        metrics_f = open(os.path.join(args.save, "metrics.jsonl"), "w")

    t_prev, fps, frame_i, n_det, t_wait = time.time(), 0.0, 0, 0, 0.0
    try:
        while True:
            got = src.read(timeout=0.2 if args.gui else 1.0)
            if got is None:
                if args.selftest and feeder is not None and not feeder.is_alive():
                    break
                if time.time() - t_wait > 5:
                    print("[ipad] waiting for frames...", flush=True)
                    t_wait = time.time()
                if args.gui:                      # idle screen: status + START/STOP button
                    canvas = np.full((360, 640, 3), 30, np.uint8)
                    if src.client_ip is None:
                        msg = "waiting for the iPad to connect..."
                    elif not src.streaming:
                        msg = f"iPad {src.client_ip} connected - press START"
                    else:
                        msg = f"iPad {src.client_ip}: waiting for frames..."
                    cv2.putText(canvas, msg, (20, 190), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (200, 200, 200), 2)
                    ui["btn"], ui["quit_btn"] = draw_buttons(canvas, src.streaming)
                    cv2.imshow(WIN, canvas)
                    if handle_key():
                        break
                continue
            bgr, depth, K, rotk, t_dev, pose = got
            k = args.rotate if args.rotate is not None else rotk
            bgr, depth, K = rotate_frame(bgr, depth, K, k)

            if udp_video is not None:   # clean upright frame; one JPEG per datagram (<64KB)
                vh, vw = bgr.shape[:2]
                s_v = 640.0 / max(vh, vw)
                small = bgr if s_v >= 1.0 else cv2.resize(
                    bgr, (int(vw * s_v), int(vh * s_v)), interpolation=cv2.INTER_AREA)
                ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok and len(enc) > 64000:
                    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 40])
                if ok and len(enc) <= 64000:
                    sock.sendto(enc.tobytes(), udp_video)

            joints, _uv = infer_joints(model, det, bgr, depth, K, device)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            if joints is not None:
                n_det += 1
                pkt = {"t": round(now, 4), "det": 1, "fps": round(fps, 1),
                       "j": [round(float(v), 5) for v in joints.reshape(-1)]}
            else:
                pkt = {"t": round(now, 4), "det": 0}
            if pose is not None:   # device camera pose rides along on EVERY frame, so the
                cp, cq = pose      # desktop Unity camera keeps tracking even without a hand
                pkt["cp"] = [round(v, 5) for v in cp]
                pkt["cq"] = [round(v, 6) for v in cq]
            data = json.dumps(pkt).encode()
            if udp_fixed is not None:
                sock.sendto(data, udp_fixed)
            if args.udp_back and src.client_ip and src.client_ip != "127.0.0.1":
                sock.sendto(data, (src.client_ip, args.udp_back))

            if args.gui or args.save:
                overlay = bgr.copy()
                if joints is not None:
                    overlay = draw_skeleton(overlay, project(joints, K))
                    cv2.putText(overlay, f"wrist_z={joints[0,2]*100:5.1f}cm", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"{fps:4.1f} FPS  ipad={src.client_ip or '?'}  rot k={k}  "
                            f"net-drop {src.frames_dropped}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if args.save:
                    cv2.imwrite(os.path.join(args.save, f"{frame_i:05d}.png"), overlay)
                    if metrics_f is not None:
                        rec = {"frame": frame_i, "detected": joints is not None,
                               "t_dev": round(t_dev, 4)}
                        if joints is not None:
                            rec["abs_joints"] = [[round(float(v), 4) for v in q] for q in joints]
                        metrics_f.write(json.dumps(rec) + "\n")
                if args.gui:
                    ui["btn"], ui["quit_btn"] = draw_buttons(overlay, src.streaming)
                    cv2.imshow(WIN, overlay)
                    if handle_key():
                        break
            frame_i += 1
            if frame_i % 60 == 0:
                print(f"[ipad] frame {frame_i}  {fps:4.1f} FPS  (hand {n_det}/{frame_i})",
                      flush=True)
            if max_frames is not None and frame_i >= max_frames:
                break
    finally:
        src.close()
        det.close()
        sock.close()
        if metrics_f is not None:
            metrics_f.close()
        if args.gui:
            cv2.destroyAllWindows()

    if args.selftest:
        ok = frame_i > 0
        print(f"[selftest] {'OK' if ok else 'FAILED'}: processed {frame_i} frames "
              f"(received {src.frames_in}, hand detected in {n_det})", flush=True)
        raise SystemExit(0 if ok else 1)
    print(f"[ipad] done: {frame_i} frames, hand in {n_det}", flush=True)


if __name__ == "__main__":
    main()
