using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Shows the live iPad/ZED camera feed inside the desktop Unity scene, as a fullscreen
    /// background BEHIND the 3D hand and grabbables. infer_ipad_stream.py --udp-video sends
    /// one JPEG per UDP datagram; this receives it on a background thread and uploads it to
    /// a texture on the main thread. Zero-setup: the Canvas/RawImage UI is built in code on
    /// the first frame (Screen Space - Camera at the far plane, so all 3D renders in front).
    /// The background hides itself when no frame arrives for <see cref="timeout"/> seconds.
    /// </summary>
    public class VideoStreamReceiver : MonoBehaviour
    {
        [Tooltip("UDP port infer_ipad_stream.py --udp-video sends JPEG frames to")]
        public int port = 9760;

        [Tooltip("Flip the video horizontally — keep this equal to HandStreamReceiver.mirrorX " +
                 "so the video moves the same way as the hand.")]
        public bool mirrorX = true;

        [Tooltip("Hide the background when no frame arrives for this long (s).")]
        public float timeout = 1.0f;

        public bool IsReceiving { get; private set; }

        UdpClient _udp;
        Thread _thread;
        volatile bool _running;
        readonly object _lock = new object();
        byte[] _pending;                 // latest JPEG, swapped in by the receive thread
        Texture2D _tex;
        RawImage _image;
        AspectRatioFitter _fitter;
        float _lastFrameTime = -999f;

        void OnEnable()
        {
            try
            {
                _udp = new UdpClient(port);
            }
            catch (SocketException e)
            {
                Debug.LogError($"[VideoStreamReceiver] cannot bind udp:{port} — {e.Message}");
                enabled = false;
                return;
            }
            _udp.Client.ReceiveTimeout = 500;
            _running = true;
            _thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "VideoStreamReceiver" };
            _thread.Start();
            Debug.Log($"[VideoStreamReceiver] listening on udp:{port}");
        }

        void OnDisable()
        {
            _running = false;
            try { _udp?.Close(); } catch { }
            try { _thread?.Join(500); } catch { }
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
                    byte[] data = _udp.Receive(ref any);   // one datagram = one whole JPEG
                    lock (_lock) _pending = data;          // latest wins
                }
                catch (SocketException) { /* timeout / closed — keep polling */ }
                catch (Exception e) { Debug.LogWarning($"[VideoStreamReceiver] {e.Message}"); }
            }
        }

        void Update()
        {
            byte[] jpeg;
            lock (_lock) { jpeg = _pending; _pending = null; }

            if (jpeg != null)
            {
                if (_image == null) BuildUi();
                if (_tex == null) _tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
                if (_tex.LoadImage(jpeg))                  // decodes JPEG, resizes the texture
                {
                    _image.texture = _tex;
                    if (_fitter != null && _tex.height > 0)
                        _fitter.aspectRatio = (float)_tex.width / _tex.height;
                    _lastFrameTime = Time.time;
                }
            }

            IsReceiving = Time.time - _lastFrameTime < timeout;
            if (_image != null && _image.enabled != IsReceiving)
                _image.enabled = IsReceiving;
        }

        // Fullscreen RawImage on a Screen Space - Camera canvas at the far plane, so the 3D
        // hand/objects always render in front of the video.
        void BuildUi()
        {
            var cam = Camera.main;
            var canvasGo = new GameObject("VideoBackground");
            var canvas = canvasGo.AddComponent<Canvas>();
            if (cam != null)
            {
                canvas.renderMode = RenderMode.ScreenSpaceCamera;
                canvas.worldCamera = cam;
                canvas.planeDistance = cam.farClipPlane * 0.95f;
            }
            else
            {
                canvas.renderMode = RenderMode.ScreenSpaceOverlay;   // fallback: video on top
            }

            var imgGo = new GameObject("VideoImage");
            imgGo.transform.SetParent(canvasGo.transform, false);
            _image = imgGo.AddComponent<RawImage>();
            // texture rows arrive top-down; LoadImage flips them, so only X may need mirroring
            _image.uvRect = mirrorX ? new Rect(1f, 0f, -1f, 1f) : new Rect(0f, 0f, 1f, 1f);
            var rt = _image.rectTransform;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = rt.offsetMax = Vector2.zero;
            _fitter = imgGo.AddComponent<AspectRatioFitter>();
            _fitter.aspectMode = AspectRatioFitter.AspectMode.FitInParent;
        }
    }
}
