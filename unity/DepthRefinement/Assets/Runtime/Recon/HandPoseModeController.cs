#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using UnityEngine;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Switches the app between the TWO inference modes at runtime:
    ///
    ///   OnDevice — the existing Sentis path: <see cref="HybridBHandProvider"/> /
    ///              <see cref="DualStreamHandProvider"/> run the model on the phone.
    ///              RgbdStreamer / ServerHandProvider are disabled.
    ///   Server   — <see cref="RgbdStreamer"/> pushes RGB + LiDAR to infer_ipad_stream.py;
    ///              <see cref="ServerHandProvider"/> receives the inferred joints back.
    ///              The on-device Sentis providers are disabled (no GPU cost).
    ///
    /// Consumers (HandSphereDriver / HandGrabController) follow whichever provider is
    /// enabled, so switching modes just flips component enabled-states. Bind
    /// <see cref="ToggleMode"/> / <see cref="SetOnDevice"/> / <see cref="SetServer"/> to UI
    /// buttons; on-device providers return to the enabled-state they had in the scene.
    /// </summary>
    public sealed class HandPoseModeController : MonoBehaviour
    {
        public enum InferenceMode { OnDevice, Server }

        [SerializeField] InferenceMode _mode = InferenceMode.OnDevice;

        [Tooltip("Let the PC's START/STOP button switch this device between modes remotely: " +
                 "START -> Server mode, STOP -> back to OnDevice. Keeps the RgbdStreamer " +
                 "control connection open even in OnDevice mode (no frames flow until START).")]
        [SerializeField] bool _remoteModeSwitch = true;

        [Header("On-device components (auto-found when empty)")]
        [SerializeField] HybridBHandProvider _hybrid;
        [SerializeField] DualStreamHandProvider _dual;

        [Header("Server components (auto-found when empty)")]
        [SerializeField] RgbdStreamer _streamer;
        [SerializeField] ServerHandProvider _server;

        public InferenceMode Mode => _mode;
        /// <summary>One-line status for a debug UI label.</summary>
        public string Status => _mode == InferenceMode.Server
            ? $"SERVER  {(_streamer != null ? _streamer.Status : "no RgbdStreamer in scene")}"
            : "ON-DEVICE (Sentis)";

        bool _hybridSceneEnabled, _dualSceneEnabled;   // scene-authored states, restored on OnDevice

        void Awake()
        {
            if (_hybrid == null) _hybrid = FindFirstObjectByType<HybridBHandProvider>(FindObjectsInactive.Include);
            if (_dual == null) _dual = FindFirstObjectByType<DualStreamHandProvider>(FindObjectsInactive.Include);
            if (_streamer == null) _streamer = FindFirstObjectByType<RgbdStreamer>(FindObjectsInactive.Include);
            if (_server == null) _server = FindFirstObjectByType<ServerHandProvider>(FindObjectsInactive.Include);

            _hybridSceneEnabled = _hybrid != null && _hybrid.enabled;
            _dualSceneEnabled = _dual != null && _dual.enabled;

            if (_streamer != null) _streamer.OnRemoteCommand += HandleRemoteCommand;
        }

        void OnDestroy()
        {
            if (_streamer != null) _streamer.OnRemoteCommand -= HandleRemoteCommand;
        }

        void Start() => Apply();

        // Start() order vs RgbdStreamer's autoStart is undefined; re-apply once after every
        // Start has run so the selected mode always wins the first frame.
        bool _reapplied;
        void LateUpdate()
        {
            if (_reapplied) return;
            _reapplied = true;
            Apply();
        }

        // Server pressed START (true) / STOP (false) on infer_ipad_stream.py.
        void HandleRemoteCommand(bool start)
        {
            if (!_remoteModeSwitch) return;
            var want = start ? InferenceMode.Server : InferenceMode.OnDevice;
            if (_mode != want) SetMode(want);
        }

        public void SetMode(InferenceMode m) { _mode = m; Apply(); }
        public void SetOnDevice() => SetMode(InferenceMode.OnDevice);
        public void SetServer() => SetMode(InferenceMode.Server);
        public void ToggleMode() =>
            SetMode(_mode == InferenceMode.OnDevice ? InferenceMode.Server : InferenceMode.OnDevice);

        void Apply()
        {
            bool onDevice = _mode == InferenceMode.OnDevice;
            if (_hybrid != null) _hybrid.enabled = onDevice && _hybridSceneEnabled;
            if (_dual != null) _dual.enabled = onDevice && _dualSceneEnabled;
            if (_server != null) _server.enabled = !onDevice;
            if (_streamer != null)
            {
                // With remote switching the streamer stays enabled in BOTH modes — it keeps
                // the control connection so the server's START can reach us; StopStreaming()
                // guarantees no frames flow while OnDevice.
                _streamer.enabled = !onDevice || _remoteModeSwitch;
                if (onDevice) _streamer.StopStreaming();
                else _streamer.StartStreaming();
            }
            Debug.Log($"[HandPoseMode] {_mode}");
        }
    }
}
#endif
