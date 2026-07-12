#if ARFOUNDATION_PRESENT
using System;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Concurrent;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;   // GraphicsFormat (off-thread JPG encode)
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Streams synchronized RGB + LiDAR depth frames to a PC over TCP for LIVE server-side
    /// hand-pose inference (<c>infer_ipad_stream.py</c>). This is the network twin of
    /// <see cref="RgbdRecorder"/>: the same sensor-orientation RGB (downscaled, JPEG) and
    /// metric depth are produced, but instead of writing cap_NNNN files to disk each frame
    /// is framed into one binary packet and pushed down a persistent TCP connection.
    ///
    /// Packet, little-endian, one per frame (server parser: HDR_FMT in infer_ipad_stream.py):
    ///   u32 magic 'RGBD' (0x44424752) | u32 version=2 | u32 jpegSize | u32 depthSize
    ///   u32 rgbW | u32 rgbH | u32 depthW | u32 depthH
    ///   f32 fx, fy, cx, cy   (intrinsics at intrW x intrH — the FULL sensor resolution)
    ///   u32 intrW | u32 intrH
    ///   u32 rotK             (k*90 deg CW that makes the sensor image upright; from
    ///                         Screen.orientation, same mapping as DualStreamHandProvider)
    ///   f64 tSec             (device unix time, seconds)
    ///   f32 camPx, camPy, camPz          (v2: ARKit camera pose in Unity WORLD space, m)
    ///   f32 camQx, camQy, camQz, camQw   (v2: camera rotation quaternion, Unity xyzw)
    ///   u32 poseOk           (v2: 1 = ARSession is tracking, 0 = pose stale/unavailable)
    ///   [jpegSize bytes: JPEG, sensor raster (row 0 = top)]
    ///   [depthSize bytes: uint16 MILLIMETERS, row-major depthH x depthW, 0 = invalid]
    ///
    /// The camera pose lets the PC-side Unity scene world-lock its virtual camera to the
    /// physical device (StreamedCameraDriver in the HandGrab scene): joints are inferred
    /// in the camera frame, so receiver-at-device-pose puts hands at their real positions.
    ///
    /// Threading mirrors RgbdRecorder: the main thread only grabs + copies the camera buffers;
    /// JPEG encoding, packet assembly and the (blocking) socket write run on a background
    /// worker. The frame queue is tiny and drops when full, so a slow network can never stall
    /// the AR loop — the newest frames win.
    /// </summary>
    public sealed class RgbdStreamer : MonoBehaviour
    {
        const uint Magic = 0x44424752;   // "RGBD" little-endian

        [SerializeField] ARCameraManager _cameraManager;
        [SerializeField] AROcclusionManager _occlusionManager;

        [Header("Server (PC running infer_ipad_stream.py)")]
        [SerializeField] string _host = "192.168.0.10";
        [SerializeField] int _port = 9776;

        [Header("Stream")]
        [Tooltip("Frames per second to capture & send. 30 keeps up with the server "
                 + "(~44 FPS capacity) at ~4.8 MB/s on WiFi.")]
        [SerializeField] float _streamFps = 30f;
        [Tooltip("Max RGB long side (px) sent over the wire. 0 = full sensor resolution.")]
        [SerializeField] int _rgbMaxSize = 640;
        [Range(1, 100)][SerializeField] int _jpgQuality = 80;
        [Tooltip("Start streaming as soon as the scene runs (otherwise call StartStreaming()).")]
        [SerializeField] bool _autoStart = true;

        public bool IsStreaming { get; private set; }
        /// <summary>TCP connection currently established.</summary>
        public bool Connected => _connected;
        /// <summary>Frames actually written to the socket this session.</summary>
        public int SentCount { get; private set; }
        /// <summary>Frames dropped (queue full / disconnected) this session.</summary>
        public int DroppedCount { get; private set; }
        /// <summary>Server-side gate: the PC's START/STOP button sends 'S'/'P' back down the
        /// TCP link; frames only flow while true. Defaults true for servers that never send
        /// commands.</summary>
        public bool RemoteEnabled => _remoteSend;
        /// <summary>Fires on the MAIN thread when the server toggles 'S' (true) / 'P' (false).
        /// HandPoseModeController uses it to auto-switch the device between modes.</summary>
        public event Action<bool> OnRemoteCommand;

        /// <summary>The EXACT frame being streamed (sensor orientation, downscaled), for an
        /// on-device preview UI. Only updated while some <see cref="RgbdStreamPreview"/> set
        /// <see cref="previewEnabled"/> — costs one texture upload per sent frame.</summary>
        public Texture2D PreviewTexture { get; private set; }
        /// <summary>rotK of the latest captured frame (k*90° CW makes it upright).</summary>
        public uint LastRotK { get; private set; } = 1;
        [NonSerialized] public bool previewEnabled;
        /// <summary>One-line status for a debug UI label.</summary>
        public string Status =>
            $"{(IsStreaming ? (_remoteSend ? "streaming" : "paused (server)") : "stopped")} " +
            $"{(_connected ? $"-> {_host}:{_port}" : "(disconnected)")}  " +
            $"sent {SentCount}  dropped {DroppedCount}";

        sealed class FrameJob
        {
            public byte[] rgbaTopDown;    // sensor-order RGBA (row 0 = top), worker flips + JPEGs
            public int w, h;
            public byte[] depthMm;        // uint16 LE millimeters, row-major, or null
            public int dW, dH;
            public float fx, fy, cx, cy;
            public int intrW, intrH;
            public uint rotK;
            public double tSec;
            public Vector3 camPos;        // ARKit camera pose, Unity world space
            public Quaternion camRot;
            public uint poseOk;
        }

        BlockingCollection<FrameJob> _queue;
        Thread _worker;
        TcpClient _client;
        NetworkStream _stream;
        volatile bool _connected;
        volatile bool _remoteSend = true;   // server 'S'/'P' command gate (worker thread writes)
        volatile bool _componentActive;     // mirrors enabled for the worker thread
        bool _lastRemoteSend = true;        // main-thread copy, for OnRemoteCommand edges
        int _lastConnectAttemptMs = int.MinValue;

        NativeArray<byte> _rgba;     // converted RGBA scratch (sensor), main thread only
        ushort[] _depthMm;           // millimeter scratch, main thread only
        float _timer;

        void Awake()
        {
            if (_cameraManager == null) _cameraManager = FindFirstObjectByType<ARCameraManager>();
            if (_occlusionManager == null) _occlusionManager = FindFirstObjectByType<AROcclusionManager>();

            // capacity 2 = latest-wins: a slow link drops frames instead of building latency
            _queue = new BlockingCollection<FrameJob>(boundedCapacity: 2);
            _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "RgbdStreamer" };
            _worker.Start();
        }

        void Start()
        {
            if (_autoStart) StartStreaming();
        }

        void OnEnable() => _componentActive = true;
        void OnDisable() => _componentActive = false;

        void OnDestroy()
        {
            IsStreaming = false;
            _queue?.CompleteAdding();
            try { _worker?.Join(2000); } catch { /* best effort */ }
            CloseClient();
            if (_rgba.IsCreated) _rgba.Dispose();
            _queue?.Dispose();
        }

        // ---- public API (bindable to UI buttons) -----------------------------------------

        public void StartStreaming() { IsStreaming = true; _timer = 0f; }
        public void StopStreaming() { IsStreaming = false; }
        public void ToggleStreaming() { if (IsStreaming) StopStreaming(); else StartStreaming(); }

        /// <summary>Change the server address at runtime (e.g. from an input field).</summary>
        public void SetHost(string host) { if (!string.IsNullOrWhiteSpace(host)) _host = host.Trim(); }

        // ---- capture (main thread) --------------------------------------------------------

        void Update()
        {
            // surface server START/STOP edges on the main thread (mode auto-switch hook)
            bool rs = _remoteSend;
            if (rs != _lastRemoteSend)
            {
                _lastRemoteSend = rs;
                OnRemoteCommand?.Invoke(rs);
            }

            if (!IsStreaming || !rs) return;   // capture only while the server wants frames
            _timer += Time.unscaledDeltaTime;
            float interval = 1f / Mathf.Max(0.5f, _streamFps);
            if (_timer < interval) return;
            _timer = 0f;
            CaptureFrame();
        }

        void CaptureFrame()
        {
            if (_cameraManager == null) return;
            if (!AcquireRgb(out byte[] rgba, out int rgbW, out int rgbH)) return;

            float fx = 0, fy = 0, cx = 0, cy = 0; int intrW = 0, intrH = 0;
            if (_cameraManager.TryGetIntrinsics(out XRCameraIntrinsics intr))
            {
                fx = intr.focalLength.x; fy = intr.focalLength.y;
                cx = intr.principalPoint.x; cy = intr.principalPoint.y;
                intrW = intr.resolution.x; intrH = intr.resolution.y;
            }

            AcquireDepthMm(out byte[] depthMm, out int dW, out int dH);   // null on non-LiDAR

            // ARKit camera pose: the transform holding ARCameraManager IS the AR camera,
            // driven to the device pose (Unity world space) by the TrackedPoseDriver.
            Transform camT = _cameraManager.transform;
            uint poseOk = ARSession.state == ARSessionState.SessionTracking ? 1u : 0u;

            LastRotK = RotKFromOrientation(Screen.orientation);
            if (previewEnabled)
            {
                if (PreviewTexture == null || PreviewTexture.width != rgbW || PreviewTexture.height != rgbH)
                    PreviewTexture = new Texture2D(rgbW, rgbH, TextureFormat.RGBA32, false);
                PreviewTexture.LoadRawTextureData(rgba);   // top-down rows; UI flips via uvRect
                PreviewTexture.Apply(false);
            }

            var job = new FrameJob
            {
                rgbaTopDown = rgba, w = rgbW, h = rgbH,
                depthMm = depthMm, dW = dW, dH = dH,
                fx = fx, fy = fy, cx = cx, cy = cy, intrW = intrW, intrH = intrH,
                rotK = LastRotK,
                tSec = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
                camPos = camT.position, camRot = camT.rotation, poseOk = poseOk,
            };
            if (!_queue.TryAdd(job)) DroppedCount++;   // backpressure: never stall the AR loop
        }

        /// <summary>Same mapping as DualStreamHandProvider.EffectiveRotation: k*90 deg CW that
        /// makes the raw sensor image upright for the current screen orientation.</summary>
        static uint RotKFromOrientation(ScreenOrientation o)
        {
            switch (o)
            {
                case ScreenOrientation.Portrait: return 1;
                case ScreenOrientation.PortraitUpsideDown: return 3;
                case ScreenOrientation.LandscapeLeft: return 0;
                case ScreenOrientation.LandscapeRight: return 2;
                default: return 1;
            }
        }

        bool AcquireRgb(out byte[] rgba, out int outW, out int outH)
        {
            rgba = null; outW = outH = 0;
            if (!_cameraManager.TryAcquireLatestCpuImage(out XRCpuImage img)) return false;
            using (img)
            {
                int srcW = img.width, srcH = img.height;
                int longSide = Mathf.Max(srcW, srcH);
                float s = (_rgbMaxSize > 0 && longSide > _rgbMaxSize) ? (float)_rgbMaxSize / longSide : 1f;
                int w = Mathf.Max(1, Mathf.RoundToInt(srcW * s));
                int h = Mathf.Max(1, Mathf.RoundToInt(srcH * s));

                var p = new XRCpuImage.ConversionParams(img, TextureFormat.RGBA32, XRCpuImage.Transformation.None)
                {
                    inputRect = new RectInt(0, 0, srcW, srcH),
                    outputDimensions = new Vector2Int(w, h),
                };
                int need = img.GetConvertedDataSize(p);
                if (!_rgba.IsCreated || _rgba.Length < need)
                {
                    if (_rgba.IsCreated) _rgba.Dispose();
                    _rgba = new NativeArray<byte>(need, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                }
                img.Convert(p, new NativeSlice<byte>(_rgba));

                rgba = new byte[need];
                NativeArray<byte>.Copy(_rgba, rgba, need);
                outW = w; outH = h;
            }
            return true;
        }

        bool AcquireDepthMm(out byte[] bytes, out int w, out int h)
        {
            bytes = null; w = h = 0;
            if (_occlusionManager == null) return false;
            if (!_occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage img)) return false;
            using (img)
            {
                w = img.width; h = img.height;
                int n = w * h;
                if (_depthMm == null || _depthMm.Length != n) _depthMm = new ushort[n];

                var raw = img.GetPlane(0).data;
                if (img.format == XRCpuImage.Format.DepthUint16)
                {
                    var src = raw.Reinterpret<ushort>(1);
                    int m = Mathf.Min(src.Length, n);
                    for (int i = 0; i < m; i++) _depthMm[i] = src[i];              // already mm
                }
                else // DepthFloat32 (ARKit) or best-effort
                {
                    var src = raw.Reinterpret<float>(1);
                    int m = Mathf.Min(src.Length, n);
                    for (int i = 0; i < m; i++)
                    {
                        float v = src[i];
                        _depthMm[i] = (v > 0f && !float.IsNaN(v) && !float.IsInfinity(v))
                            ? (ushort)Mathf.Min(v * 1000f, 65535f) : (ushort)0;
                    }
                }

                bytes = new byte[n * 2];
                Buffer.BlockCopy(_depthMm, 0, bytes, 0, n * 2);   // little-endian uint16
            }
            return true;
        }

        // ---- worker thread: JPEG encode + packet assembly + socket write -------------------

        void WorkerLoop()
        {
            byte[] flip = null;
            var packet = new MemoryStream();
            var bw = new BinaryWriter(packet);   // BinaryWriter is little-endian by spec
            while (!_queue.IsCompleted)
            {
                _queue.TryTake(out FrameJob job, 200);   // null job = idle tick

                if (!_componentActive)                   // component disabled: hold no connection
                {
                    if (_connected) CloseClient();
                    continue;
                }
                // Keep the link up and read server commands even while idle, paused or in
                // OnDevice mode, so the server's START button can reach us at any time.
                if (!EnsureConnected()) { if (job != null) DroppedCount++; continue; }
                PollCommands();
                if (job == null) continue;
                if (!_remoteSend) { DroppedCount++; continue; }

                try
                {
                    int stride = job.w * 4;
                    int n = stride * job.h;
                    if (flip == null || flip.Length < n) flip = new byte[n];
                    // sensor top-down -> bottom-up rows that EncodeArrayToJPG expects
                    for (int y = 0; y < job.h; y++)
                        Buffer.BlockCopy(job.rgbaTopDown, (job.h - 1 - y) * stride, flip, y * stride, stride);
                    byte[] jpeg = ImageConversion.EncodeArrayToJPG(
                        flip, GraphicsFormat.R8G8B8A8_SRGB, (uint)job.w, (uint)job.h, 0, _jpgQuality);

                    packet.SetLength(0);
                    bw.Write(Magic);
                    bw.Write(2u);                                   // version
                    bw.Write((uint)jpeg.Length);
                    bw.Write((uint)(job.depthMm?.Length ?? 0));
                    bw.Write((uint)job.w); bw.Write((uint)job.h);
                    bw.Write((uint)job.dW); bw.Write((uint)job.dH);
                    bw.Write(job.fx); bw.Write(job.fy); bw.Write(job.cx); bw.Write(job.cy);
                    bw.Write((uint)job.intrW); bw.Write((uint)job.intrH);
                    bw.Write(job.rotK);
                    bw.Write(job.tSec);
                    bw.Write(job.camPos.x); bw.Write(job.camPos.y); bw.Write(job.camPos.z);
                    bw.Write(job.camRot.x); bw.Write(job.camRot.y);
                    bw.Write(job.camRot.z); bw.Write(job.camRot.w);
                    bw.Write(job.poseOk);
                    bw.Write(jpeg);
                    if (job.depthMm != null) bw.Write(job.depthMm);
                    bw.Flush();

                    _stream.Write(packet.GetBuffer(), 0, (int)packet.Length);
                    SentCount++;
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[RgbdStreamer] send failed ({e.Message}) — will reconnect");
                    CloseClient();
                    DroppedCount++;
                }
            }
        }

        /// <summary>Drain server -> device control bytes: 'S' = send frames, 'P' = pause.
        /// infer_ipad_stream.py sends the current state on connect and on button toggles.</summary>
        void PollCommands()
        {
            try
            {
                while (_stream != null && _stream.DataAvailable)
                {
                    int b = _stream.ReadByte();
                    if (b == 'S') _remoteSend = true;
                    else if (b == 'P') _remoteSend = false;
                    else if (b < 0) { CloseClient(); return; }   // server closed the socket
                }
            }
            catch (Exception)
            {
                CloseClient();
            }
        }

        bool EnsureConnected()
        {
            if (_client != null && _client.Connected) return true;
            // retry at most every 2 s (Environment.TickCount: worker thread can't use Time.time)
            if (unchecked(Environment.TickCount - _lastConnectAttemptMs) < 2000) return false;
            _lastConnectAttemptMs = Environment.TickCount;
            try
            {
                CloseClient();
                var c = new TcpClient { NoDelay = true, SendTimeout = 3000 };
                IAsyncResult ar = c.BeginConnect(_host, _port, null, null);
                if (!ar.AsyncWaitHandle.WaitOne(1500)) { c.Close(); return false; }
                c.EndConnect(ar);
                _client = c;
                _stream = c.GetStream();
                _connected = true;
                Debug.Log($"[RgbdStreamer] connected to {_host}:{_port}");
                return true;
            }
            catch (Exception)
            {
                _connected = false;
                return false;
            }
        }

        void CloseClient()
        {
            _connected = false;
            try { _stream?.Close(); } catch { }
            try { _client?.Close(); } catch { }
            _stream = null;
            _client = null;
        }
    }
}
#endif
