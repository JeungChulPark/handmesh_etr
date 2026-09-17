using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Records the Game view to an MP4 from the moment Play starts until it stops, so a
    /// session can be reviewed afterwards (tracking glitches, grab misses, depth errors).
    /// Everything that is on screen is captured — video background, 3D hand, grabbables,
    /// the proximity label — plus a "REC mm:ss.cc  f=N" stamp to reference moments by time.
    ///
    /// Pipeline: end of frame → ScreenCapture.CaptureScreenshotIntoRenderTexture →
    /// AsyncGPUReadback (no main-thread stall) → worker thread → ffmpeg stdin (raw RGBA).
    /// Output is constant frame rate on the REAL clock: a frame is written as many times
    /// as output ticks elapsed since the previous one, so playback speed matches reality
    /// even when Unity renders slower or faster than <see cref="fps"/>. Game time is never
    /// slowed down (unlike Time.captureFramerate), which a live network stream requires.
    ///
    /// Needs ffmpeg (macOS: `brew install ffmpeg`). Without it the frames are written as a
    /// JPEG sequence instead. Resizing the Game view mid-recording starts a new file.
    /// Output: &lt;project&gt;/Recordings/HandGrab_yyyyMMdd_HHmmss.mp4
    /// </summary>
    public class GameViewRecorder : MonoBehaviour
    {
        public enum Flip { Auto, On, Off }

        [Tooltip("Start recording as soon as the component starts (i.e. when Play is pressed)")]
        public bool recordOnStart = true;
        [Tooltip("Output frame rate (frames are duplicated/dropped to match the real clock)")]
        public int fps = 30;
        [Tooltip("Folder for recordings; relative paths are under the Unity project root")]
        public string outputFolder = "Recordings";
        [Tooltip("ffmpeg executable. Empty = search /opt/homebrew/bin, /usr/local/bin, /usr/bin, PATH")]
        public string ffmpegPath = "";
        [Tooltip("x264 quality (lower = better/larger; 18–23 is typical)")]
        public int crf = 20;
        [Tooltip("Draw a REC time / frame stamp into the video")]
        public bool showTimestamp = true;
        [Tooltip("Vertical flip of the captured frame. Auto picks by graphics API; set On/Off " +
                 "if the saved video comes out upside down.")]
        public Flip verticalFlip = Flip.Auto;
        [Tooltip("Frames allowed to wait for the encoder before new ones are dropped")]
        public int maxQueuedFrames = 12;

        public bool IsRecording { get; private set; }
        public string CurrentFile { get; private set; }
        public int DroppedFrames => _dropped;

        struct Frame { public byte[] rgba; public int repeat; }

        // --- per-segment state ---
        int _w, _h;
        RenderTexture _rt;
        Process _ffmpeg;
        Stream _stdin;
        string _jpegDir;                // non-null = JPEG-sequence fallback
        int _jpegIndex;
        BlockingCollection<Frame> _queue;
        Thread _worker;
        double _t0;                     // realtime at segment start
        long _ticksWritten;             // output frames already accounted for
        int _pendingReadbacks;
        volatile int _dropped;
        bool _flip;

        // --- session state (spans segments) ---
        double _sessionT0;
        int _frameCount;
        Coroutine _loop;
        GUIStyle _style;

        void Start()
        {
            if (recordOnStart) StartRecording();
        }

        public void StartRecording()
        {
            if (IsRecording) return;
            _sessionT0 = Time.realtimeSinceStartupAsDouble;
            _frameCount = 0;
            IsRecording = true;
            _loop = StartCoroutine(CaptureLoop());
        }

        public void StopRecording()
        {
            if (!IsRecording) return;
            IsRecording = false;
            if (_loop != null) StopCoroutine(_loop);
            _loop = null;
            EndSegment();
        }

        void OnDisable() => StopRecording();   // Play stopped / object destroyed: finalize the file

        IEnumerator CaptureLoop()
        {
            var eof = new WaitForEndOfFrame();
            while (IsRecording)
            {
                yield return eof;   // after rendering AND OnGUI, so overlays are captured
                if (!IsRecording) break;

                if (_queue == null || Screen.width != _w || Screen.height != _h)
                {
                    EndSegment();
                    if (!BeginSegment(Screen.width, Screen.height)) { StopRecording(); yield break; }
                }

                // how many output frames this capture stands for on the real clock
                double t = Time.realtimeSinceStartupAsDouble - _t0;
                long due = (long)Math.Floor(t * fps) + 1;
                int repeat = (int)(due - _ticksWritten);
                _frameCount++;
                if (repeat <= 0) continue;          // rendering faster than fps: skip
                _ticksWritten = due;

                if (_pendingReadbacks + _queue.Count >= maxQueuedFrames) { _dropped++; continue; }

                ScreenCapture.CaptureScreenshotIntoRenderTexture(_rt);
                _pendingReadbacks++;
                var queue = _queue;                 // bind to THIS segment
                AsyncGPUReadback.Request(_rt, 0, TextureFormat.RGBA32, req =>
                {
                    _pendingReadbacks--;
                    if (req.hasError || queue.IsAddingCompleted) { _dropped++; return; }
                    byte[] buf = req.GetData<byte>().ToArray();
                    if (!queue.TryAdd(new Frame { rgba = buf, repeat = repeat })) _dropped++;
                });
            }
        }

        bool BeginSegment(int w, int h)
        {
            _w = w; _h = h;
            _flip = verticalFlip == Flip.On
                    || (verticalFlip == Flip.Auto && !SystemInfo.graphicsUVStartsAtTop);
            _rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
            _rt.Create();

            string dir = Path.IsPathRooted(outputFolder)
                ? outputFolder
                : Path.Combine(Path.GetDirectoryName(Application.dataPath) ?? ".", outputFolder);
            Directory.CreateDirectory(dir);
            string stem = Path.Combine(dir, $"HandGrab_{DateTime.Now:yyyyMMdd_HHmmss}");

            string exe = FindFfmpeg();
            if (exe != null)
            {
                CurrentFile = stem + ".mp4";
                // even dimensions are required by yuv420p/x264
                string vf = (_flip ? "vflip," : "") + "scale=trunc(iw/2)*2:trunc(ih/2)*2";
                var psi = new ProcessStartInfo
                {
                    FileName = exe,
                    Arguments = $"-y -loglevel error -f rawvideo -pix_fmt rgba -s {w}x{h} -r {fps} -i - " +
                                $"-vf \"{vf}\" -c:v libx264 -preset veryfast -crf {crf} " +
                                $"-pix_fmt yuv420p -movflags +faststart \"{CurrentFile}\"",
                    UseShellExecute = false,
                    RedirectStandardInput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };
                try
                {
                    _ffmpeg = Process.Start(psi);
                    _ffmpeg.ErrorDataReceived += (_, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data)) Debug.LogWarning($"[GameViewRecorder] ffmpeg: {e.Data}");
                    };
                    _ffmpeg.BeginErrorReadLine();
                    _stdin = _ffmpeg.StandardInput.BaseStream;
                }
                catch (Exception e)
                {
                    Debug.LogError($"[GameViewRecorder] cannot start ffmpeg ({exe}): {e.Message}");
                    _ffmpeg = null;
                }
            }

            if (_ffmpeg == null)
            {
                _jpegDir = stem + "_frames";
                Directory.CreateDirectory(_jpegDir);
                _jpegIndex = 0;
                CurrentFile = _jpegDir;
                Debug.LogWarning("[GameViewRecorder] ffmpeg not found — saving a JPEG sequence instead " +
                                 "(macOS: brew install ffmpeg)");
            }

            _queue = new BlockingCollection<Frame>();
            _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "GameViewRecorder" };
            _worker.Start();
            _t0 = Time.realtimeSinceStartupAsDouble;
            _ticksWritten = 0;
            _dropped = 0;
            Debug.Log($"[GameViewRecorder] recording {w}x{h}@{fps} → {CurrentFile}");
            return true;
        }

        void EndSegment()
        {
            if (_queue == null) return;
            _queue.CompleteAdding();
            // late readbacks are rejected by IsAddingCompleted; the worker drains the rest
            if (!_worker.Join(10000)) Debug.LogWarning("[GameViewRecorder] encoder did not drain in 10 s");

            if (_ffmpeg != null)
            {
                try { _stdin.Close(); } catch { }
                if (!_ffmpeg.WaitForExit(10000))
                {
                    Debug.LogWarning("[GameViewRecorder] ffmpeg did not finish in 10 s — killing (file may be truncated)");
                    try { _ffmpeg.Kill(); } catch { }
                }
                _ffmpeg.Dispose();
            }
            string msg = _dropped > 0 ? $" ({_dropped} frames dropped — encoder too slow)" : "";
            Debug.Log($"[GameViewRecorder] saved {CurrentFile}{msg}");

            _queue = null;   // not Disposed: late readback callbacks still query IsAddingCompleted
            _worker = null;
            _ffmpeg = null;
            _stdin = null;
            _jpegDir = null;
            if (_rt != null) { _rt.Release(); Destroy(_rt); _rt = null; }
        }

        void WorkerLoop()
        {
            try
            {
                foreach (Frame f in _queue.GetConsumingEnumerable())
                {
                    if (_jpegDir != null)
                    {
                        // EncodeArrayToJPG wants Unity's bottom-up rows; _flip means the
                        // readback already IS bottom-up (ffmpeg needed vflip), so invert here
                        byte[] jpg = ImageConversion.EncodeArrayToJPG(
                            _flip ? f.rgba : FlipRows(f.rgba, _w, _h),
                            UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_SRGB,
                            (uint)_w, (uint)_h, 0, 90);
                        // JPEG sequence has no timing: one file per capture
                        File.WriteAllBytes(Path.Combine(_jpegDir, $"{_jpegIndex++:D6}.jpg"), jpg);
                        continue;
                    }
                    for (int i = 0; i < f.repeat; i++) _stdin.Write(f.rgba, 0, f.rgba.Length);
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[GameViewRecorder] write failed: {e.Message}");
                // keep draining so CompleteAdding/Join never hangs
                foreach (Frame _ in _queue.GetConsumingEnumerable()) { }
            }
        }

        static byte[] FlipRows(byte[] src, int w, int h)
        {
            int stride = w * 4;
            var dst = new byte[src.Length];
            for (int y = 0; y < h; y++)
                Buffer.BlockCopy(src, y * stride, dst, (h - 1 - y) * stride, stride);
            return dst;
        }

        string FindFfmpeg()
        {
            if (!string.IsNullOrWhiteSpace(ffmpegPath))
                return File.Exists(ffmpegPath) ? ffmpegPath : null;
            // apps launched from the Dock/Hub don't inherit the shell PATH, so check brew first
            string[] known = { "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg" };
            foreach (string p in known)
                if (File.Exists(p)) return p;
            string exeName = Application.platform == RuntimePlatform.WindowsEditor
                             || Application.platform == RuntimePlatform.WindowsPlayer ? "ffmpeg.exe" : "ffmpeg";
            foreach (string d in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
            {
                if (string.IsNullOrEmpty(d)) continue;
                string p = Path.Combine(d, exeName);
                if (File.Exists(p)) return p;
            }
            return null;
        }

        void OnGUI()
        {
            if (!IsRecording || !showTimestamp) return;
            if (_style == null)
                _style = new GUIStyle(GUI.skin.label) { fontSize = 16, fontStyle = FontStyle.Bold };

            double t = Time.realtimeSinceStartupAsDouble - _sessionT0;
            string txt = $"● REC {(int)(t / 60):00}:{t % 60:00.00}  f={_frameCount}";
            var r = new Rect(10, 8, 320, 24);
            _style.normal.textColor = Color.black;
            GUI.Label(new Rect(r.x + 1, r.y + 1, r.width, r.height), txt, _style);
            _style.normal.textColor = new Color(1f, 0.3f, 0.3f);
            GUI.Label(r, txt, _style);
        }
    }
}
