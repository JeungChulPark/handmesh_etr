using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Zero-setup bootstrap: add this to ONE empty GameObject in an empty scene
    /// and press Play. It builds the whole demo in code — receiver + skeleton +
    /// pinch grabber at the origin (the virtual camera pose), a table, and
    /// grabbable sphere/cube/capsule in front of the camera where the hand
    /// appears (0.3–0.7 m depth for a desk-mounted ZED / iPhone).
    ///
    /// Run the streamer first:
    ///   python unity_stream_hand.py                    # live ZED
    ///   python unity_stream_hand.py --source replay    # no camera needed
    /// </summary>
    public class HandGrabDemo : MonoBehaviour
    {
        public int udpPort = 9750;
        public bool mirrorX = true;
        [Tooltip("Meters from camera to the object shelf — keep within one-handed reach of "
               + "the device (~0.3 m); at the device FOV (~67°) objects shrink fast with distance")]
        public float objectDistance = 0.32f;
        [Tooltip("Height (m) of the grabbables above the view centre. 0 = the row sits in the " +
                 "middle of the screen; objects drop onto the table 5 cm below this height.")]
        public float objectHeight = 0f;
        [Tooltip("Show the live camera feed behind the scene (infer_ipad_stream.py --udp-video). " +
                 "Harmless when no video stream is running — the background just stays hidden.")]
        public bool showVideo = true;
        [Tooltip("UDP port for the JPEG video background")]
        public int videoPort = 9760;
        [Tooltip("World-lock the desktop camera to the iPad's streamed ARKit pose " +
                 "(infer_ipad_stream.py \"cp\"/\"cq\"). Forces mirrorX off; the virtual " +
                 "camera then moves through the scene exactly like the physical device. " +
                 "Turn OFF for a fixed desk sensor (ZED) stream, which carries no pose.")]
        public bool followDeviceCamera = true;
        [Tooltip("Objects hover in place instead of falling: spawned kinematic and kept " +
                 "kinematic after release. Turn off for gravity + toss physics.")]
        public bool floatObjects = true;
        [Tooltip("Guide line + cm label from the pinch point to the nearest grabbable, " +
                 "so you can judge depth on a flat monitor (green = close enough to grab).")]
        public bool showDistanceGuide = true;
        [Tooltip("Grey table slab under the objects. Off by default — floating objects need " +
                 "no support and the slab looked out of place over the video background.")]
        public bool showTable = false;
        [Tooltip("Mirror the rendered view left-right (video + hand + objects). ON when the " +
                 "device camera faces you (iPad rear camera pointed at the user); world " +
                 "coordinates are untouched, so grabbing and distances stay correct.")]
        public bool mirrorView = true;
        [Tooltip("Size multiplier for the grabbable objects (1 = sphere/cube 8 cm)")]
        public float objectScale = 0.6f;
        [Tooltip("Centre-to-centre gap (m) of the object row, which is centred on the view")]
        public float objectSpacing = 0.08f;
        [Tooltip("Record the Game view to <project>/Recordings/*.mp4 while playing " +
                 "(GameViewRecorder; needs ffmpeg — brew install ffmpeg)")]
        public bool recordGameView = true;

        HandStreamReceiver _recv;
        Transform _table;
        readonly System.Collections.Generic.List<Transform> _grabbables = new();
        readonly System.Collections.Generic.List<Vector3> _offsets = new();   // in the shelf frame
        StreamedCameraDriver _driver;
        VideoStreamReceiver _video;
        bool _placedOnPose;
        bool _wasStreaming;
        float _recenterAt = -1f;        // Time.time of a pending auto re-center, <0 = none
        const float SettleSec = 0.3f;   // let the camera pose smoothing catch up first

        void Start()
        {
            if (followDeviceCamera) mirrorX = false;   // mirroring breaks a moving camera
            // --- camera at the sensor pose, looking down +Z ---
            var cam = Camera.main;
            if (cam == null)
            {
                cam = new GameObject("Main Camera").AddComponent<Camera>();
                cam.tag = "MainCamera";
            }
            cam.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            cam.nearClipPlane = 0.05f;
            cam.backgroundColor = new Color(0.12f, 0.12f, 0.16f);
            cam.clearFlags = CameraClearFlags.SolidColor;
            if (mirrorView && cam.GetComponent<MirroredCameraView>() == null)
                cam.gameObject.AddComponent<MirroredCameraView>();

            if (FindObjectOfType<Light>() == null)
            {
                var l = new GameObject("Sun").AddComponent<Light>();
                l.type = LightType.Directional;
                l.transform.rotation = Quaternion.Euler(50, -30, 0);
            }

            // --- hand rig (receiver + skeleton + grabber share one GameObject) ---
            var hand = new GameObject("HandRig");
            var recv = _recv = hand.AddComponent<HandStreamReceiver>();
            recv.port = udpPort;
            recv.mirrorX = mirrorX;
            hand.AddComponent<HandSkeleton>();
            hand.AddComponent<PinchGrabber>();
            if (showDistanceGuide) hand.AddComponent<GrabProximityIndicator>();

            if (followDeviceCamera)
            {
                var driver = _driver = hand.AddComponent<StreamedCameraDriver>();
                driver.receiver = recv;
                driver.driveMainCamera = true;   // HandRig carries the joints; camera follows
            }

            if (showVideo)
            {
                var video = _video = hand.AddComponent<VideoStreamReceiver>();
                video.port = videoPort;
                video.mirrorX = mirrorX;   // video must mirror the same way as the hand
            }

            // --- optional table so released objects land somewhere (just below the shelf) ---
            if (showTable)
            {
                var table = GameObject.CreatePrimitive(PrimitiveType.Cube);
                table.name = "Table";
                table.transform.position = new Vector3(0, objectHeight - 0.05f, objectDistance);
                table.transform.localScale = new Vector3(1.2f, 0.02f, 0.8f);
                table.GetComponent<MeshRenderer>().material.color = new Color(0.25f, 0.28f, 0.32f);
                _table = table.transform;
            }

            if (recordGameView && GetComponent<GameViewRecorder>() == null)
                gameObject.AddComponent<GameViewRecorder>();

            // --- grabbables: a row centred on the view (floating, or dropping onto the table) ---
            Spawn(PrimitiveType.Sphere, new Vector3(-objectSpacing, objectHeight, objectDistance),
                  0.08f * objectScale, new Color(0.85f, 0.35f, 0.3f));
            Spawn(PrimitiveType.Cube, new Vector3(0f, objectHeight, objectDistance),
                  0.08f * objectScale, new Color(0.3f, 0.55f, 0.85f));
            Spawn(PrimitiveType.Capsule, new Vector3(objectSpacing, objectHeight, objectDistance),
                  0.07f * objectScale, new Color(0.4f, 0.8f, 0.45f));
        }

        // With a world-locked camera the ARKit world origin is wherever the AR session
        // STARTED — the fixed spawn spot can be behind the user or across the room. So:
        //   * once the FIRST pose has been APPLIED to the camera, move the row in front of it
        //     (checking the receiver instead fired one frame early, while the camera was
        //     still at the origin, and left the objects off-screen);
        //   * again whenever the stream resumes after a gap — a relaunched iPad app starts a
        //     new ARKit session with a new origin;
        //   * R re-centers it any time (device drifted / objects thrown out of reach).
        void LateUpdate()
        {
            if (followDeviceCamera && _driver != null)
            {
                bool posed = _driver.HasPose;
                if (posed && !_placedOnPose)
                {
                    _placedOnPose = true;
                    _recenterAt = Time.time + SettleSec;
                }
                bool streaming = _video != null ? _video.IsReceiving : posed;
                if (streaming && !_wasStreaming && _placedOnPose)
                    _recenterAt = Time.time + SettleSec;
                _wasStreaming = streaming;
            }
            if (_recenterAt >= 0f && Time.time >= _recenterAt)
            {
                _recenterAt = -1f;
                Recenter();
            }
            if (Input.GetKeyDown(KeyCode.R)) Recenter();
        }

        /// <summary>Place the table + grabbables `objectDistance` along the camera's view
        /// ray, i.e. in the middle of the screen even when the device is tilted. The row
        /// itself stays level (gravity-aligned yaw). Held objects are left in the hand.</summary>
        public void Recenter()
        {
            var cam = Camera.main;
            if (cam == null) return;
            Transform ct = cam.transform;
            Vector3 flat = ct.forward; flat.y = 0f;
            flat = flat.sqrMagnitude < 1e-4f ? Vector3.forward : flat.normalized;
            Quaternion yaw = Quaternion.LookRotation(flat);
            Vector3 anchor = ct.position + ct.forward * objectDistance
                             + Vector3.up * objectHeight;

            if (_table != null)
                _table.SetPositionAndRotation(anchor - Vector3.up * 0.05f, yaw);
            for (int i = 0; i < _grabbables.Count; i++)
            {
                Transform g = _grabbables[i];
                if (g == null) continue;
                var grab = g.GetComponent<Grabbable>();
                if (grab != null && grab.IsHeld) continue;
                var rb = g.GetComponent<Rigidbody>();
                if (rb != null && !rb.isKinematic)
                {
#if UNITY_6000_0_OR_NEWER
                    rb.linearVelocity = Vector3.zero;
#else
                    rb.velocity = Vector3.zero;
#endif
                    rb.angularVelocity = Vector3.zero;
                }
                g.SetPositionAndRotation(anchor + yaw * _offsets[i], yaw);
            }
            Debug.Log($"[HandGrabDemo] shelf re-centered {objectDistance:0.00} m in front of the camera");
        }

        void Spawn(PrimitiveType type, Vector3 pos, float size, Color color)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = $"Grabbable_{type}";
            go.transform.position = pos;
            go.transform.localScale = Vector3.one * size;
            go.GetComponent<MeshRenderer>().material.color = color;
            var rb = go.AddComponent<Rigidbody>();
            rb.mass = 0.2f;
            rb.interpolation = RigidbodyInterpolation.Interpolate;
            rb.isKinematic = floatObjects;   // float mode: hover at the spawn point too
            go.AddComponent<Grabbable>().floatOnRelease = floatObjects;
            _grabbables.Add(go.transform);
            _offsets.Add(new Vector3(pos.x, pos.y - objectHeight, 0f));  // shelf-frame offset
        }
    }
}
