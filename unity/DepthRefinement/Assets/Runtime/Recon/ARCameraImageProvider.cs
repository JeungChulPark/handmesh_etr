#if ARFOUNDATION_PRESENT
using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;   // XRCpuImage

namespace HandMesh.DepthRefinement
{
    /// <summary>
    /// iOS/Android camera-frame extractor for on-device model inference (the cross-platform
    /// replacement for Android-only ARCore <c>TextureReader</c>/<c>Frame.AcquireCameraImage</c>).
    ///
    /// AR Foundation's <see cref="ARCameraManager"/> abstracts ARKit (iOS) and ARCore (Android)
    /// behind ONE API, so the SAME code path that feeds your TFLite/Sentis model on Android
    /// also runs on iPhone. Each AR frame we:
    ///   1. acquire the CPU image  -> <see cref="ARCameraManager.TryAcquireLatestCpuImage"/>
    ///      (iOS delivers YUV 420; Convert() turns it into RGBA),
    ///   2. downscale to a square model input (<see cref="ModelInputSize"/>) in native code,
    ///   3. rotate sensor-landscape -> screen-upright (portrait) on the CPU,
    ///   4. expose it as <see cref="OutputTexture"/> (for Sentis <c>TextureConverter.ToTensor</c>)
    ///      and as a normalized NHWC float[] (for a TFLite interpreter input).
    ///
    /// Allocation-free in steady state (persistent buffers). Drop this on the AR camera object,
    /// assign the <see cref="ARCameraManager"/>, and read <see cref="OutputTexture"/> /
    /// <see cref="GetInputTensorNHWC"/> each frame (or subscribe to <see cref="OnCameraImage"/>).
    /// </summary>
    public sealed class ARCameraImageProvider : MonoBehaviour
    {
        public enum Rotation { None = 0, CW90 = 90, Rot180 = 180, CW270 = 270 }

        [SerializeField] ARCameraManager _cameraManager;

        [Tooltip("Square side fed to the model (matches the .tflite/.onnx input, e.g. 256).")]
        [SerializeField] int _modelInputSize = 256;

        [Tooltip("Auto-pick rotation from Screen.orientation (portrait/landscape). " +
                 "Disable to force a fixed rotation below.")]
        [SerializeField] bool _autoRotateFromScreen = true;

        [Tooltip("Used when Auto Rotate is off. iOS back-camera portrait is usually CW90.")]
        [SerializeField] Rotation _rotation = Rotation.CW90;

        [Tooltip("Mirror horizontally (enable for the front/selfie camera).")]
        [SerializeField] bool _mirrorHorizontal = false;

        [Tooltip("Region of the SENSOR image (pre-rotation, normalized 0..1) to crop before " +
                 "resizing to the square input. Default = full frame. Set this to a hand bbox " +
                 "(e.g. from MediaPipe) to run the model on the hand crop, like on Android.")]
        [SerializeField] Rect _cropRectNormalized = new Rect(0, 0, 1, 1);

        /// <summary>Sensor-space (pre-rotation) crop, normalized 0..1. Settable at runtime
        /// from a hand detector. Default = full frame.</summary>
        public Rect CropRectNormalized { get => _cropRectNormalized; set => _cropRectNormalized = value; }

        /// <summary>Upright square RGBA frame, ready for the model. Null until the first frame.</summary>
        public Texture2D OutputTexture => _output;

        /// <summary>True once at least one camera frame has been converted.</summary>
        public bool HasFrame { get; private set; }

        public int ModelInputSize => _modelInputSize;

        /// <summary>Raised after each frame is converted; arg = <see cref="OutputTexture"/>.</summary>
        public event Action<Texture2D> OnCameraImage;

        Texture2D _output;
        NativeArray<byte> _convBuffer; // size*size*4, sensor orientation (from Convert)
        byte[] _rotated;               // size*size*4, upright
        float[] _nhwc;                 // size*size*3, normalized RGB for TFLite

        void OnEnable()
        {
            if (_cameraManager == null) _cameraManager = FindObjectOfType<ARCameraManager>();
            if (_cameraManager == null)
            {
                Debug.LogError("[ARCameraImageProvider] No ARCameraManager assigned/found.");
                enabled = false;
                return;
            }
            _cameraManager.frameReceived += OnFrameReceived;
        }

        void OnDisable()
        {
            if (_cameraManager != null) _cameraManager.frameReceived -= OnFrameReceived;
            HasFrame = false;
            if (_convBuffer.IsCreated) _convBuffer.Dispose();
        }

        void OnFrameReceived(ARCameraFrameEventArgs _)
        {
            // Grab the latest CPU image (the camera background, in sensor orientation).
            if (!_cameraManager.TryAcquireLatestCpuImage(out XRCpuImage image)) return;
            using (image)
            {
                ConvertAndOrient(image);
            }
            HasFrame = true;
            OnCameraImage?.Invoke(_output);
        }

        void ConvertAndOrient(XRCpuImage image)
        {
            int size = _modelInputSize;

            // 1) YUV -> RGBA32, downscaled to size x size, in native code (cheap, no GC).
            var transformation = _mirrorHorizontal
                ? XRCpuImage.Transformation.MirrorX
                : XRCpuImage.Transformation.None;
            var p = new XRCpuImage.ConversionParams(image, TextureFormat.RGBA32, transformation)
            {
                inputRect = SensorInputRect(image.width, image.height),
                outputDimensions = new Vector2Int(size, size),
            };
            int need = image.GetConvertedDataSize(p);
            EnsureNative(ref _convBuffer, need);
            // AR Foundation 6.x exposes Convert(ConversionParams, NativeSlice<byte>) (and an
            // IntPtr overload); there is no NativeArray overload. NativeSlice is a safe type,
            // so this compiles without allowUnsafeCode.
            image.Convert(p, new NativeSlice<byte>(_convBuffer));

            // 2) Rotate sensor-landscape -> screen-upright on the CPU (256^2 is trivial).
            int rot = _autoRotateFromScreen ? RotationFromScreen() : (int)_rotation;
            EnsureManaged(ref _rotated, size * size * 4);
            RotateRGBA(_convBuffer, _rotated, size, rot);

            // 3) Upload to the output texture (feed Sentis directly, or read NHWC for TFLite).
            if (_output == null || _output.width != size || _output.height != size)
                _output = new Texture2D(size, size, TextureFormat.RGBA32, false);
            _output.LoadRawTextureData(_rotated);
            _output.Apply(false);
        }

        /// <summary>
        /// Returns the current frame as a normalized NHWC RGB float[] (length size*size*3,
        /// values [0,1], channel order R,G,B) — the usual TFLite input layout. Feed it with
        /// <c>interpreter.SetInputTensorData(inputIndex, GetInputTensorNHWC())</c>.
        /// Reuses one buffer; copy if you need to keep it.
        /// </summary>
        public float[] GetInputTensorNHWC()
        {
            int size = _modelInputSize;
            int n = size * size;
            if (_nhwc == null || _nhwc.Length != n * 3) _nhwc = new float[n * 3];
            if (_rotated == null) return _nhwc;
            const float inv = 1f / 255f;
            for (int i = 0; i < n; i++)
            {
                int s = i * 4;   // RGBA
                int d = i * 3;   // RGB
                _nhwc[d + 0] = _rotated[s + 0] * inv;
                _nhwc[d + 1] = _rotated[s + 1] * inv;
                _nhwc[d + 2] = _rotated[s + 2] * inv;
            }
            return _nhwc;
        }

        RectInt SensorInputRect(int w, int h)
        {
            Rect c = _cropRectNormalized;
            // Clamp to [0,1] and to a non-empty pixel rect.
            float x = Mathf.Clamp01(c.x), y = Mathf.Clamp01(c.y);
            float cw = Mathf.Clamp01(c.width), ch = Mathf.Clamp01(c.height);
            int px = Mathf.Clamp(Mathf.RoundToInt(x * w), 0, w - 1);
            int py = Mathf.Clamp(Mathf.RoundToInt(y * h), 0, h - 1);
            int pw = Mathf.Clamp(Mathf.RoundToInt(cw * w), 1, w - px);
            int ph = Mathf.Clamp(Mathf.RoundToInt(ch * h), 1, h - py);
            return new RectInt(px, py, pw, ph);
        }

        int RotationFromScreen()
        {
            // Camera sensor is landscape; map the current screen orientation to the CW rotation
            // that brings the image upright. Tune if your app locks orientation differently.
            switch (Screen.orientation)
            {
                case ScreenOrientation.Portrait: return 90;
                case ScreenOrientation.PortraitUpsideDown: return 270;
                case ScreenOrientation.LandscapeLeft: return 0;
                case ScreenOrientation.LandscapeRight: return 180;
                default: return (int)_rotation;
            }
        }

        // Rotate a size x size RGBA buffer clockwise by rot (0/90/180/270) degrees.
        static void RotateRGBA(NativeArray<byte> src, byte[] dst, int n, int rot)
        {
            for (int y = 0; y < n; y++)
            {
                for (int x = 0; x < n; x++)
                {
                    int sx, sy;
                    switch (rot)
                    {
                        case 90:  sx = y;         sy = n - 1 - x; break; // CW90:  dst(x,y) = src(col=y, row=n-1-x)
                        case 180: sx = n - 1 - x; sy = n - 1 - y; break;
                        case 270: sx = n - 1 - y; sy = x;         break;
                        default:  sx = x;         sy = y;         break; // 0
                    }
                    int s = (sy * n + sx) * 4;
                    int d = (y * n + x) * 4;
                    dst[d + 0] = src[s + 0];
                    dst[d + 1] = src[s + 1];
                    dst[d + 2] = src[s + 2];
                    dst[d + 3] = src[s + 3];
                }
            }
        }

        static void EnsureNative(ref NativeArray<byte> buf, int len)
        {
            if (buf.IsCreated && buf.Length >= len) return;
            if (buf.IsCreated) buf.Dispose();
            buf = new NativeArray<byte>(len, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        static void EnsureManaged(ref byte[] buf, int len)
        {
            if (buf == null || buf.Length != len) buf = new byte[len];
        }
    }
}
#endif
