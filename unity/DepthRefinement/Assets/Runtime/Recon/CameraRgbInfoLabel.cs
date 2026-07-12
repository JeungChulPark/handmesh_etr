#if ARFOUNDATION_PRESENT
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// Prints live camera/depth EXTRACTION info to a UI <see cref="Text"/> (e.g. RGBLabel):
    /// the raw sensor CPU image size + pixel format, the camera intrinsics resolution / focal /
    /// principal point, the LiDAR depth image size, the on-screen display texture size, plus
    /// screen size + orientation and FPS.
    ///
    /// This is the actual data the model reads — handy for confirming what resolution/format the
    /// pipeline gets and whether sensor vs display dimensions differ. Display-only: it acquires
    /// its own CPU image handles (throttled) and does NOT touch the model path.
    ///
    /// Attach to the RGBLabel object (or any object) and assign the Text. Uses the legacy
    /// <see cref="Text"/> to match the rest of the project; for a TextMeshPro label see the note
    /// at the bottom of this file.
    /// </summary>
    public sealed class CameraRgbInfoLabel : MonoBehaviour
    {
        [SerializeField] ARCameraManager _cameraManager;
        [Tooltip("Optional — for the LiDAR depth image size line.")]
        [SerializeField] AROcclusionManager _occlusionManager;
        [Tooltip("Optional — to also show the on-screen RGB display texture size.")]
        [SerializeField] CameraRgbDisplay _rgbDisplay;
        [Tooltip("Legacy UI Text to write to (RGBLabel). Leave empty if using TextMeshPro.")]
        [SerializeField] Text _label;
#if TMP_PRESENT
        [Tooltip("TextMeshPro label to write to (RGBLabel). Auto-found on this object.")]
        [SerializeField] TMPro.TMP_Text _tmpLabel;
#endif

        [Tooltip("Seconds between refreshes. The CPU image is acquired only on refresh, so this " +
                 "is cheap; FPS is still smoothed every frame.")]
        [SerializeField] float _refreshInterval = 0.5f;

        readonly StringBuilder _sb = new StringBuilder(256);
        float _timer;
        float _fps;

        void Awake()
        {
            if (_cameraManager == null) _cameraManager = FindFirstObjectByType<ARCameraManager>();
            if (_occlusionManager == null) _occlusionManager = FindFirstObjectByType<AROcclusionManager>();
            if (_rgbDisplay == null) _rgbDisplay = FindFirstObjectByType<CameraRgbDisplay>();
            if (_label == null) _label = GetComponent<Text>();
#if TMP_PRESENT
            if (_tmpLabel == null) _tmpLabel = GetComponent<TMPro.TMP_Text>();
            if (_label == null && _tmpLabel == null)
                Debug.LogError("[CameraRgbInfoLabel] No Text/TMP_Text assigned (RGBLabel).");
#else
            if (_label == null) Debug.LogError("[CameraRgbInfoLabel] No Text assigned (RGBLabel).");
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
            float dt = Time.unscaledDeltaTime;
            if (dt > 0f) _fps = Mathf.Lerp(_fps, 1f / dt, 0.1f);   // smoothed FPS

            _timer += dt;
            if (_timer < _refreshInterval) return;
            _timer = 0f;
            Refresh();
        }

        void Refresh()
        {
            _sb.Clear();
            _sb.Append("FPS ").Append(Mathf.RoundToInt(_fps)).Append('\n');

            // Raw sensor RGB CPU image (what the model crops from).
            if (_cameraManager != null && _cameraManager.TryAcquireLatestCpuImage(out XRCpuImage img))
            {
                using (img)
                    _sb.Append("RGB  ").Append(img.width).Append('x').Append(img.height)
                       .Append("  ").Append(img.format)
                       .Append("  planes ").Append(img.planeCount).Append('\n');
            }
            else _sb.Append("RGB  : no image\n");

            // Camera intrinsics (resolution may differ from the CPU image).
            if (_cameraManager != null && _cameraManager.TryGetIntrinsics(out XRCameraIntrinsics intr))
            {
                _sb.Append("intr ").Append(intr.resolution.x).Append('x').Append(intr.resolution.y)
                   .Append("  f(").Append(intr.focalLength.x.ToString("0")).Append(',')
                   .Append(intr.focalLength.y.ToString("0")).Append(")  c(")
                   .Append(intr.principalPoint.x.ToString("0")).Append(',')
                   .Append(intr.principalPoint.y.ToString("0")).Append(")\n");
            }

            // LiDAR depth CPU image.
            if (_occlusionManager != null &&
                _occlusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage d))
            {
                using (d)
                    _sb.Append("dpth ").Append(d.width).Append('x').Append(d.height)
                       .Append("  ").Append(d.format).Append('\n');
            }

            // On-screen display texture (after downscale/rotation in CameraRgbDisplay).
            if (_rgbDisplay != null && _rgbDisplay.Texture != null)
                _sb.Append("tex  ").Append(_rgbDisplay.Texture.width).Append('x')
                   .Append(_rgbDisplay.Texture.height).Append('\n');

            // Screen + orientation (helps diagnose sensor-vs-display mismatch).
            _sb.Append("scrn ").Append(Screen.width).Append('x').Append(Screen.height)
               .Append("  ").Append(Screen.orientation);

            SetText(_sb.ToString());
        }
    }
}
#endif
// Works with either a legacy UI Text or a TextMeshPro label on RGBLabel (auto-detected).
