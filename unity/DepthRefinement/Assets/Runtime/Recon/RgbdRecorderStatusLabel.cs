#if ARFOUNDATION_PRESENT
using UnityEngine;
using UnityEngine.UI;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Shows <see cref="RgbdRecorder"/> state on a UI label: recording indicator, number of
    /// captures written this session, and the output folder. Turns red while recording. Works
    /// with either a legacy UI <see cref="Text"/> or a TextMeshPro label (auto-detected).
    ///
    /// Updates only when the state changes (no per-frame allocation). Attach to the status label
    /// object, or assign the label field. Pairs with the Capture / ● REC buttons.
    /// </summary>
    public sealed class RgbdRecorderStatusLabel : MonoBehaviour
    {
        [SerializeField] RgbdRecorder _recorder;
        [Tooltip("Legacy UI Text to write to. Leave empty if using TextMeshPro.")]
        [SerializeField] Text _label;
#if TMP_PRESENT
        [Tooltip("TextMeshPro label to write to. Auto-found on this object.")]
        [SerializeField] TMPro.TMP_Text _tmpLabel;
#endif
        [SerializeField] Color _recordingColor = new Color(0.95f, 0.2f, 0.2f);
        [SerializeField] Color _idleColor = Color.white;

        int _lastCount = -1;
        bool _lastRec;
        int _lastSharpBucket = -1;
        bool _init;

        void Awake()
        {
            if (_recorder == null) _recorder = FindFirstObjectByType<RgbdRecorder>();
            if (_label == null) _label = GetComponent<Text>();
#if TMP_PRESENT
            if (_tmpLabel == null) _tmpLabel = GetComponent<TMPro.TMP_Text>();
#endif
        }

        void Update()
        {
            if (_recorder == null) { SetText("recorder: none", false); return; }

            bool rec = _recorder.IsRecording;
            int count = _recorder.Count;
            // Sharpness changes every frame; bucket it so we only rebuild the string on a
            // meaningful change (keeps the focus readout live for tuning _sharpnessMin without
            // per-frame allocation noise).
            int sharpBucket = Mathf.RoundToInt(_recorder.LastSharpness / 5f);
            if (_init && rec == _lastRec && count == _lastCount && sharpBucket == _lastSharpBucket) return;
            _init = true; _lastRec = rec; _lastCount = count; _lastSharpBucket = sharpBucket;

            string folder = string.IsNullOrEmpty(_recorder.OutputDir)
                ? "" : System.IO.Path.GetFileName(_recorder.OutputDir);
            string head = rec ? "● REC" : "IDLE";
            SetText($"{head}   {count} saved  (skip {_recorder.Skipped}, queue {_recorder.Pending})\n" +
                    $"focus {_recorder.LastSharpness:F0}\n→ {folder}", rec);
        }

        void SetText(string s, bool recording)
        {
            Color c = recording ? _recordingColor : _idleColor;
            if (_label != null) { _label.text = s; _label.color = c; }
#if TMP_PRESENT
            if (_tmpLabel != null) { _tmpLabel.text = s; _tmpLabel.color = c; }
#endif
        }
    }
}
#endif
// Works with either a legacy UI Text or a TextMeshPro label (auto-detected).
