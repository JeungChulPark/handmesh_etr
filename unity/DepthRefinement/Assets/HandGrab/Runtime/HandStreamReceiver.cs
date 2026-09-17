using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Receives hand-joint packets from unity_stream_hand.py over UDP and exposes
    /// the latest 21 joints in Unity world space (meters).
    ///
    /// Packet: {"t":<sec>,"det":1,"fps":21.3,"j":[63 floats]} — camera frame,
    /// x right / y down / z forward. Mapping to Unity: (x, -y, z), optional
    /// x-mirror, then transformed by this GameObject's transform (put the
    /// receiver at the virtual "camera" pose in your scene).
    ///
    /// infer_ipad_stream.py additionally sends the iPad's ARKit camera pose as
    /// "cp" (world position [x,y,z]) and "cq" (rotation quaternion [x,y,z,w]),
    /// exposed here via <see cref="CameraPosition"/>/<see cref="CameraRotation"/>.
    /// Add <see cref="StreamedCameraDriver"/> to apply it to this transform (and
    /// the Main Camera) for a world-locked view that follows the device.
    /// </summary>
    public class HandStreamReceiver : MonoBehaviour
    {
        public const int JointCount = 21;

        [Tooltip("UDP port unity_stream_hand.py sends to")]
        public int port = 9750;

        [Tooltip("Flip X so the hand moves like a mirror (natural for a front-facing camera)")]
        public bool mirrorX = true;

        [Header("One Euro smoothing")]
        public bool smooth = true;
        public float minCutoff = 1.2f;
        public float beta = 0.05f;

        [Tooltip("Seconds without a detected hand before IsTracked goes false")]
        public float trackingTimeout = 0.3f;

        [Header("Jump rejection")]
        [Tooltip("A new frame whose wrist moved faster than this (m/s, camera frame) is treated " +
                 "as an outlier (wrong LiDAR root depth / detector misfire) and ignored")]
        public float maxWristSpeed = 3.0f;
        [Tooltip("Accept the new position anyway after this many consecutive outliers — the " +
                 "hand really did move there")]
        public int jumpConfirmFrames = 3;

        /// <summary>Latest joints in world space; valid only while IsTracked.</summary>
        public Vector3[] Joints { get; } = new Vector3[JointCount];
        public bool IsTracked { get; private set; }
        public float StreamFps { get; private set; }
        /// <summary>Packets / hand detections / rejected jumps received so far.</summary>
        public int PacketCount { get; private set; }
        public int HandCount { get; private set; }
        public int RejectedJumps { get; private set; }
        /// <summary>Time.time of the last packet (hand or not), -999 = never.</summary>
        public float LastPacketTime { get; private set; } = -999f;

        /// <summary>True once a packet carried the device camera pose ("cp"/"cq").</summary>
        public bool HasCameraPose { get; private set; }
        /// <summary>Streamed ARKit camera pose, Unity world space (v2 iPad stream).</summary>
        public Vector3 CameraPosition { get; private set; }
        public Quaternion CameraRotation { get; private set; } = Quaternion.identity;
        /// <summary>Device camera vertical FOV (deg) of the streamed upright frame, 0 = unknown.
        /// The desktop camera must adopt it or virtual objects slide against the video.</summary>
        public float DeviceFovDeg { get; private set; }

        [Serializable]
        class Packet
        {
            public double t;
            public int det;
            public float fps;
            public float[] j;
            public float[] cp;   // device camera world position [x,y,z] (optional)
            public float[] cq;   // device camera rotation quaternion [x,y,z,w] (optional)
            public float fov;    // device camera vertical FOV, degrees (optional, 0 = absent)
        }

        UdpClient _udp;
        Thread _thread;
        volatile bool _running;
        readonly object _lock = new object();
        float[] _latest;                 // raw 63 floats, camera frame
        float[] _latestCp, _latestCq;    // raw device camera pose, or null
        float _latestFov;                // device vertical FOV (deg), 0 = not received
        double _latestT;
        float _latestFps;
        float _lastDetTime = -999f;
        double _prevT = double.NaN;      // packet time already consumed
        Vector3 _acceptedWrist;          // raw camera-frame wrist of the last accepted frame
        double _acceptedT;
        int _outliers;                   // consecutive rejected frames
        OneEuroFilterV3[] _filters;

        void OnEnable()
        {
            _filters = new OneEuroFilterV3[JointCount];
            for (int i = 0; i < JointCount; i++)
                _filters[i] = new OneEuroFilterV3(minCutoff, beta);

            _udp = new UdpClient(port);
            _udp.Client.ReceiveTimeout = 500;
            _running = true;
            _thread = new Thread(ReceiveLoop) { IsBackground = true };
            _thread.Start();
            Debug.Log($"[HandStreamReceiver] listening on udp:{port}");
        }

        void OnDisable()
        {
            _running = false;
            _udp?.Close();
            _thread?.Join(500);
            _udp = null;
            _thread = null;
        }

        void ReceiveLoop()
        {
            var any = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    byte[] data = _udp.Receive(ref any);
                    var pkt = JsonUtility.FromJson<Packet>(System.Text.Encoding.UTF8.GetString(data));
                    if (pkt == null) continue;
                    lock (_lock)
                    {
                        _latestFps = pkt.fps;
                        _latestT = pkt.t;
                        _latest = (pkt.det == 1 && pkt.j != null && pkt.j.Length == JointCount * 3)
                            ? pkt.j : null;
                        if (pkt.cp != null && pkt.cp.Length == 3 &&
                            pkt.cq != null && pkt.cq.Length == 4)
                        {
                            _latestCp = pkt.cp;
                            _latestCq = pkt.cq;
                        }
                        if (pkt.fov > 1f) _latestFov = pkt.fov;
                    }
                }
                catch (SocketException) { /* timeout / closed — keep polling */ }
                catch (Exception e) { Debug.LogWarning($"[HandStreamReceiver] {e.Message}"); }
            }
        }

        void Update()
        {
            float[] raw, cp, cq;
            float fov;
            double t;
            lock (_lock)
            {
                raw = _latest;
                cp = _latestCp;
                cq = _latestCq;
                fov = _latestFov;
                t = _latestT;
                StreamFps = _latestFps;
            }

            if (cp != null && cq != null)   // pose freezes at the last value on dropout
            {
                CameraPosition = new Vector3(cp[0], cp[1], cp[2]);
                CameraRotation = new Quaternion(cq[0], cq[1], cq[2], cq[3]);
                HasCameraPose = true;
            }
            if (fov > 1f) DeviceFovDeg = fov;

            bool fresh = t != _prevT;   // Update runs faster than packets arrive
            if (fresh)
            {
                _prevT = t;
                PacketCount++;
                LastPacketTime = Time.time;
                if (raw != null)
                {
                    HandCount++;
                    if (IsJump(raw, t)) raw = null;   // keep showing the last good pose
                }
            }
            else if (raw != null && _outliers > 0)
            {
                raw = null;   // same (rejected) packet again
            }

            if (raw != null)
            {
                // only a NEW packet counts as a detection: when the stream stops, _latest keeps
                // the last hand forever, which froze the skeleton on screen (recording 141958,
                // 22-36 s: packets 0/s yet the hand stayed "tracked")
                if (fresh) _lastDetTime = Time.time;
                for (int i = 0; i < JointCount; i++)
                {
                    // camera frame (x right, y down, z fwd) -> Unity (x right, y up, z fwd)
                    var p = new Vector3(mirrorX ? -raw[i * 3] : raw[i * 3],
                                        -raw[i * 3 + 1],
                                        raw[i * 3 + 2]);
                    if (smooth) p = _filters[i].Filter(p, t);
                    Joints[i] = transform.TransformPoint(p);
                }
            }

            bool tracked = Time.time - _lastDetTime < trackingTimeout;
            if (!tracked && IsTracked)
                foreach (var f in _filters) f.Reset();
            IsTracked = tracked;
        }

        // Wrist speed gate on the RAW camera-frame joints. Re-acquiring after a tracking
        // loss is always accepted; a run of jumpConfirmFrames outliers is accepted too
        // (filters reset so the hand snaps instead of sliding across the gap).
        bool IsJump(float[] raw, double t)
        {
            var wrist = new Vector3(raw[0], raw[1], raw[2]);
            bool reacquire = Time.time - _lastDetTime >= trackingTimeout;
            if (!reacquire)
            {
                float dt = Mathf.Max((float)(t - _acceptedT), 1f / 60f);
                if ((wrist - _acceptedWrist).magnitude / dt > maxWristSpeed
                    && ++_outliers < jumpConfirmFrames)
                {
                    RejectedJumps++;
                    return true;
                }
                if (_outliers >= jumpConfirmFrames)
                    foreach (var f in _filters) f.Reset();
            }
            _outliers = 0;
            _acceptedWrist = wrist;
            _acceptedT = t;
            return false;
        }
    }
}
