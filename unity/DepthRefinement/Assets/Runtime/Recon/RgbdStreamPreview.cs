#if ARFOUNDATION_PRESENT
using UnityEngine;
using UnityEngine.UI;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// On-device preview of the EXACT frames <see cref="RgbdStreamer"/> is sending to the
    /// server — for checking, in the existing viewer UI, that the streamed video looks right
    /// (crop/scale/orientation). Same idea as DepthImageVisualizer's DepthView, but showing
    /// the outgoing RGB.
    ///
    /// Assign an existing RawImage (e.g. a sibling of DepthView on the viewer Canvas), or
    /// leave <see cref="_target"/> empty to auto-create a small picture-in-picture RawImage
    /// (bottom-left). The stream is sent in SENSOR orientation; the preview is rotated by
    /// rotK so it appears upright, exactly like the server sees it after rotate_frame().
    /// Hidden automatically while the streamer is not sending (stopped / server-paused).
    /// </summary>
    public sealed class RgbdStreamPreview : MonoBehaviour
    {
        [SerializeField] RgbdStreamer _streamer;
        [Tooltip("RawImage to draw into. Empty = auto-create a bottom-left PiP.")]
        [SerializeField] RawImage _target;
        [Tooltip("PiP long side in pixels when auto-creating the RawImage.")]
        [SerializeField] float _pipSize = 280f;
        [Tooltip("Rotate the preview by the frame's rotK so it shows upright.")]
        [SerializeField] bool _uprightRotation = true;

        bool _enabledPreviewOnStreamer;

        void Awake()
        {
            if (_streamer == null) _streamer = FindFirstObjectByType<RgbdStreamer>(FindObjectsInactive.Include);
        }

        void OnEnable()
        {
            if (_streamer != null)
            {
                _streamer.previewEnabled = true;
                _enabledPreviewOnStreamer = true;
            }
        }

        void OnDisable()
        {
            if (_streamer != null && _enabledPreviewOnStreamer)
                _streamer.previewEnabled = false;
            if (_target != null) _target.enabled = false;
        }

        void Update()
        {
            if (_streamer == null) return;
            var tex = _streamer.PreviewTexture;
            bool live = tex != null && _streamer.IsStreaming && _streamer.RemoteEnabled;

            if (_target == null && live) BuildPip();
            if (_target == null) return;

            if (_target.enabled != live) _target.enabled = live;
            if (!live) return;

            _target.texture = tex;
            // raw rows are top-down while UI textures are bottom-up -> flip vertically
            _target.uvRect = new Rect(0f, 1f, 1f, -1f);

            int k = _uprightRotation ? (int)_streamer.LastRotK : 0;
            _target.rectTransform.localEulerAngles = new Vector3(0f, 0f, -90f * k);

            // aspect-fit inside a square of _pipSize; k odd swaps the on-screen axes
            float aspect = tex.height > 0 ? (float)tex.width / tex.height : 1f;
            float w = aspect >= 1f ? _pipSize : _pipSize * aspect;
            float h = aspect >= 1f ? _pipSize / aspect : _pipSize;
            _target.rectTransform.sizeDelta = new Vector2(w, h);
        }

        // Small PiP on its own overlay canvas, anchored bottom-left, above the AR background.
        void BuildPip()
        {
            var canvasGo = new GameObject("StreamPreviewCanvas");
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 10;

            var imgGo = new GameObject("StreamPreview");
            imgGo.transform.SetParent(canvasGo.transform, false);
            _target = imgGo.AddComponent<RawImage>();
            var rt = _target.rectTransform;
            rt.anchorMin = rt.anchorMax = new Vector2(0f, 0f);
            rt.pivot = new Vector2(0.5f, 0.5f);
            // pivot-anchored with margin; sized every frame in Update
            rt.anchoredPosition = new Vector2(_pipSize * 0.5f + 16f, _pipSize * 0.5f + 16f);
        }
    }
}
#endif
