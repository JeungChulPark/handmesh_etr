#if ARFOUNDATION_PRESENT
using System.Text;
using Unity.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Prints live LiDAR depth info to a UI <see cref="Text"/> (e.g. DepthLabel): the depth image
    /// size + pixel format, valid-pixel coverage, near/far/mean distance (metres), and the
    /// CENTRE-pixel distance (what's straight ahead). This is the same depth the model anchors
    /// the hand root from, so it doubles as a sanity check that LiDAR is returning sane metres.
    ///
    /// Display-only and throttled: it acquires its own depth CPU image on refresh and scans it
    /// (the image is small, ~256x192), without touching the model path. Needs a LiDAR device
    /// (iPhone/iPad Pro) and an <c>AROcclusionManager</c> with Environment Depth enabled.
    ///
    /// Attach to the DepthLabel object and assign the Text. Uses legacy <see cref="Text"/> to
    /// match the project; see the TextMeshPro note at the bottom for a TMP label.
    /// </summary>
    public sealed class DepthInfoLabel : MonoBehaviour
    {
        [SerializeField] AROcclusionManager _occlusionManager;
        [Tooltip("Legacy UI Text to write to (DepthLabel). Leave empty if using TextMeshPro.")]
        [SerializeField] Text _label;
#if TMP_PRESENT
        [Tooltip("TextMeshPro label to write to (DepthLabel). Auto-found on this object.")]
        [SerializeField] TMPro.TMP_Text _tmpLabel;
#endif
        [Tooltip("Seconds between refreshes (the depth image is scanned only on refresh).")]
        [SerializeField] float _refreshInterval = 0.5f;

        readonly StringBuilder _sb = new StringBuilder(256);
        float _timer;

        void Awake()
        {
            if (_occlusionManager == null) _occlusionManager = FindFirstObjectByType<AROcclusionManager>();
            if (_label == null) _label = GetComponent<Text>();
#if TMP_PRESENT
            if (_tmpLabel == null) _tmpLabel = GetComponent<TMPro.TMP_Text>();
            if (_label == null && _tmpLabel == null)
                Debug.LogError("[DepthInfoLabel] No Text/TMP_Text assigned (DepthLabel).");
#else
            if (_label == null) Debug.LogError("[DepthInfoLabel] No Text assigned (DepthLabel).");
#endif
        }

        void SetText(string s)
        {
            if (_label != null) _label.text = s;
#if TMP_PRESENT
            if (_tmpLabel != null) _tmpLabel.text = s;
#endif
        }

        void Update()
        {
            _timer += Time.unscaledDeltaTime;
            if (_timer < _refreshInterval) return;
            _timer = 0f;
            Refresh();
        }

        void Refresh()
        {
            if (_occlusionManager == null ||
                !_occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage img))
            {
                SetText("depth: no LiDAR frame\n(needs a Pro device + Environment Depth)");
                return;
            }

            using (img)
            {
                int w = img.width, h = img.height, total = w * h;
                int ci = (h / 2) * w + (w / 2);       // centre pixel
                var raw = img.GetPlane(0).data;

                int valid = 0;
                float minV = float.MaxValue, maxV = 0f, sum = 0f, center = 0f;

                if (img.format == XRCpuImage.Format.DepthUint16)
                {
                    var src = raw.Reinterpret<ushort>(1);
                    int n = src.Length;
                    for (int i = 0; i < n; i++)
                    {
                        float m = src[i] * 0.001f;     // mm -> m
                        if (m > 0f) { valid++; sum += m; if (m < minV) minV = m; if (m > maxV) maxV = m; }
                    }
                    if (ci >= 0 && ci < n) center = src[ci] * 0.001f;
                }
                else // DepthFloat32 (ARKit) or best-effort
                {
                    var src = raw.Reinterpret<float>(1);
                    int n = src.Length;
                    for (int i = 0; i < n; i++)
                    {
                        float m = src[i];
                        if (m > 0f && !float.IsNaN(m) && !float.IsInfinity(m))
                        { valid++; sum += m; if (m < minV) minV = m; if (m > maxV) maxV = m; }
                    }
                    if (ci >= 0 && ci < n) center = src[ci];
                }

                float cov = total > 0 ? (float)valid / total : 0f;
                float mean = valid > 0 ? sum / valid : 0f;

                _sb.Clear();
                _sb.Append("depth ").Append(w).Append('x').Append(h)
                   .Append("  ").Append(img.format).Append('\n');
                _sb.Append("cov ").Append((cov * 100f).ToString("0")).Append("%  valid ")
                   .Append(valid).Append('/').Append(total).Append('\n');
                if (valid > 0)
                {
                    _sb.Append("range ").Append(minV.ToString("0.00")).Append('-')
                       .Append(maxV.ToString("0.00")).Append(" m  mean ")
                       .Append(mean.ToString("0.00")).Append(" m\n");
                    _sb.Append("center ").Append(center.ToString("0.00")).Append(" m");
                }
                else _sb.Append("range: no valid depth");

                SetText(_sb.ToString());
            }
        }
    }
}
#endif
// Works with either a legacy UI Text or a TextMeshPro label on DepthLabel (auto-detected).
