using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// SERVER-inference hand pose source. While <see cref="RgbdStreamer"/> pushes this device's
    /// RGB + LiDAR frames to the PC, <c>infer_ipad_stream.py</c> runs the hybrid lifter there
    /// and sends the joints back over UDP (<c>--udp-back</c>, default port 9751). This component
    /// receives them and exposes the SAME surface as <see cref="HybridBHandProvider"/> —
    /// <see cref="AbsJoints"/> in the display-orientation CV camera frame (+X right, +Y down,
    /// +Z forward, metres), <see cref="HasPose"/>, and <see cref="OnPose"/> fired on the main
    /// thread — so HandSphereDriver / HandGrabController consume either source unchanged.
    ///
    /// Packet (JSON, one datagram per frame, same as unity_stream_hand.py):
    ///   {"t":&lt;sec&gt;,"det":1,"fps":21.3,"j":[x0,y0,z0, ... x20,y20,z20]}
    /// </summary>
    public sealed class ServerHandProvider : MonoBehaviour
    {
        const int JointCount = 21;

        [Tooltip("UDP port infer_ipad_stream.py sends joints back to (--udp-back).")]
        [SerializeField] int _port = 9751;
        [Tooltip("Seconds without a detected hand before HasPose goes false.")]
        [SerializeField] float _timeout = 0.3f;

        /// <summary>Absolute camera-space joints (metres, CV frame). Valid while <see cref="HasPose"/>.</summary>
        public Vector3[] AbsJoints { get; } = new Vector3[JointCount];
        public bool HasPose { get; private set; }
        /// <summary>Inference FPS reported by the server (for a debug label).</summary>
        public float ServerFps { get; private set; }
        public event Action OnPose;

        [Serializable]
        class Packet { public double t; public int det; public float fps; public float[] j; }

        UdpClient _udp;
        Thread _thread;
        volatile bool _running;
        readonly object _lock = new object();
        float[] _pending;               // latest 63 floats (null = no hand in that frame)
        float _pendingFps;
        bool _fresh;
        float _lastPoseTime = -999f;

        void OnEnable()
        {
            try
            {
                _udp = new UdpClient(_port);
            }
            catch (SocketException e)
            {
                Debug.LogError($"[ServerHandProvider] cannot bind udp:{_port} — {e.Message}");
                enabled = false;
                return;
            }
            _udp.Client.ReceiveTimeout = 500;
            _running = true;
            _thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "ServerHandProvider" };
            _thread.Start();
            Debug.Log($"[ServerHandProvider] listening on udp:{_port}");
        }

        void OnDisable()
        {
            _running = false;
            try { _udp?.Close(); } catch { }
            try { _thread?.Join(500); } catch { }
            _udp = null;
            _thread = null;
            HasPose = false;
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
                        _pendingFps = pkt.fps;
                        _pending = (pkt.det == 1 && pkt.j != null && pkt.j.Length == JointCount * 3)
                            ? pkt.j : null;
                        _fresh = true;
                    }
                }
                catch (SocketException) { /* receive timeout / socket closed — keep polling */ }
                catch (Exception e) { Debug.LogWarning($"[ServerHandProvider] {e.Message}"); }
            }
        }

        void Update()
        {
            float[] j = null;
            bool fresh;
            lock (_lock)
            {
                fresh = _fresh;
                _fresh = false;
                if (fresh) { j = _pending; ServerFps = _pendingFps; }
            }

            if (fresh && j != null)
            {
                // NO axis change here: the server sends CV camera-frame joints, and consumers
                // (HandSphereDriver / HandGrabController) already do the CV -> Unity flip.
                for (int i = 0; i < JointCount; i++)
                    AbsJoints[i] = new Vector3(j[i * 3], j[i * 3 + 1], j[i * 3 + 2]);
                _lastPoseTime = Time.time;
                HasPose = true;
                OnPose?.Invoke();
            }
            else if (HasPose && Time.time - _lastPoseTime > _timeout)
            {
                HasPose = false;
            }
        }
    }
}
