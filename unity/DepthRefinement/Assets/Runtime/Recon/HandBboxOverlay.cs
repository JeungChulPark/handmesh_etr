#if SENTIS_PRESENT && ARFOUNDATION_PRESENT
using UnityEngine;
using UnityEngine.UI;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Draws the detected hand region (from <see cref="HandBboxBridge"/>) as a translucent
    /// rectangle over a target RawImage (the RGB panel) — a debug overlay to confirm the bbox
    /// tracks the hand and to tune the bridge's Rotation/Mirror. Uses the box in the DETECTOR
    /// display frame, so it lines up with the upright RGB preview.
    ///
    /// Creates one child RawImage anchored to the box's normalized region of the target, so it
    /// follows the target's size automatically (no pixel math).
    /// </summary>
    public sealed class HandBboxOverlay : MonoBehaviour
    {
        [SerializeField] HandBboxBridge _bridge;
        [Tooltip("RawImage to overlay the box on (the RGB panel, e.g. RgbView).")]
        [SerializeField] RawImage _target;
        [Tooltip("Box fill colour (use low alpha so the hand stays visible).")]
        [SerializeField] Color _color = new Color(1f, 0.9f, 0.1f, 0.28f);

        RawImage _box;

        void Start()
        {
            if (_bridge == null) _bridge = FindFirstObjectByType<HandBboxBridge>();
            if (_target == null) { Debug.LogError("[HandBboxOverlay] No target RawImage assigned."); enabled = false; return; }

            var go = new GameObject("HandBbox");
            go.transform.SetParent(_target.transform, false);
            _box = go.AddComponent<RawImage>();
            _box.texture = Texture2D.whiteTexture;   // solid colour fill
            _box.color = _color;
            _box.raycastTarget = false;
            _box.enabled = false;
        }

        void Update()
        {
            if (_box == null) return;
            if (_bridge == null || !_bridge.HasBox)
            {
                _box.enabled = false;
                return;
            }
            _box.enabled = true;

            Rect b = _bridge.BoxDisplayNormalized;       // normalized, origin top-left
            var rt = _box.rectTransform;
            // UI anchors are bottom-up, the box is top-down -> flip Y.
            rt.anchorMin = new Vector2(b.xMin, 1f - b.yMax);
            rt.anchorMax = new Vector2(b.xMax, 1f - b.yMin);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }
    }
}
#endif
