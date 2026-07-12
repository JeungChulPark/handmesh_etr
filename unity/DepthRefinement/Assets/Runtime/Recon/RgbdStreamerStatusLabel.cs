#if ARFOUNDATION_PRESENT
using UnityEngine;
using UnityEngine.UI;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Shows <see cref="RgbdStreamer"/> state on a UI label of the existing viewer Canvas:
    /// connection, streaming / paused-by-server / stopped, frames sent + dropped, and the
    /// measured outgoing FPS. Green while frames flow, yellow while connected-but-paused,
    /// grey when disconnected. Same pattern as <see cref="RgbdRecorderStatusLabel"/>; works
    /// with either a legacy UI <see cref="Text"/> or a TextMeshPro label (auto-detected).
    /// </summary>
    public sealed class RgbdStreamerStatusLabel : MonoBehaviour
    {
        [SerializeField] RgbdStreamer _streamer;
        [Tooltip("Legacy UI Text to write to. Leave empty if using TextMeshPro.")]
        [SerializeField] Text _label;
#if TMP_PRESENT
        [Tooltip("TextMeshPro label to write to. Auto-found on this object.")]
        [SerializeField] TMPro.TMP_Text _tmpLabel;
#endif
        [SerializeField] Color _streamingColor = new Color(0.3f, 0.95f, 0.4f);
        [SerializeField] Color _pausedColor = new Color(0.95f, 0.85f, 0.25f);
        [SerializeField] Color _offColor = new Color(0.8f, 0.8f, 0.8f);
        [Tooltip("Refresh interval (s) — also the window for the FPS estimate.")]
        [SerializeField] float _refresh = 0.5f;

        float _nextRefresh;
        int _lastSent;
        float _lastSentTime;
        float _fps;

        void Awake()
        {
            if (_streamer == null) _streamer = FindFirstObjectByType<RgbdStreamer>(FindObjectsInactive.Include);
            if (_label == null) _label = GetComponent<Text>();
#if TMP_PRESENT
            if (_tmpLabel == null) _tmpLabel = GetComponent<TMPro.TMP_Text>();
#endif
        }

        void Update()
        {
            if (Time.time < _nextRefresh) return;
            _nextRefresh = Time.time + Mathf.Max(0.1f, _refresh);

            if (_streamer == null) { SetText("streamer: none", _offColor); return; }

            int sent = _streamer.SentCount;
            float now = Time.time;
            if (_lastSentTime > 0f && now > _lastSentTime)
                _fps = Mathf.Lerp(_fps, (sent - _lastSent) / (now - _lastSentTime), 0.7f);
            _lastSent = sent;
            _lastSentTime = now;

            bool sending = _streamer.IsStreaming && _streamer.RemoteEnabled && _streamer.Connected;
            string head = !_streamer.Connected ? "OFFLINE"
                        : !_streamer.IsStreaming ? "STOPPED"
                        : !_streamer.RemoteEnabled ? "PAUSED (server)"
                        : $"● SENDING {_fps:F1} fps";
            Color c = sending ? _streamingColor
                    : _streamer.Connected ? _pausedColor : _offColor;
            SetText($"{head}\nsent {sent}  dropped {_streamer.DroppedCount}", c);
        }

        void SetText(string s, Color c)
        {
            if (_label != null) { _label.text = s; _label.color = c; }
#if TMP_PRESENT
            if (_tmpLabel != null) { _tmpLabel.text = s; _tmpLabel.color = c; }
#endif
        }
    }
}
#endif
