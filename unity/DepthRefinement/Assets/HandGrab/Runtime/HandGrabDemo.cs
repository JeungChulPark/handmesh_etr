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
        [Tooltip("Meters from camera to the object shelf")]
        public float objectDistance = 0.45f;
        [Tooltip("Height (m) of the grabbables relative to the camera axis. The hand tracks " +
                 "around y=0 (camera center), so keep this near 0; objects drop onto the " +
                 "table 5 cm below this height.")]
        public float objectHeight = 0.05f;
        [Tooltip("Show the live camera feed behind the scene (infer_ipad_stream.py --udp-video). " +
                 "Harmless when no video stream is running — the background just stays hidden.")]
        public bool showVideo = true;
        [Tooltip("UDP port for the JPEG video background")]
        public int videoPort = 9760;
        [Tooltip("World-lock the desktop camera to the iPad's streamed ARKit pose " +
                 "(infer_ipad_stream.py \"cp\"/\"cq\"). Forces mirrorX off; the virtual " +
                 "camera then moves through the scene exactly like the physical device.")]
        public bool followDeviceCamera = false;
        [Tooltip("Objects hover in place instead of falling: spawned kinematic and kept " +
                 "kinematic after release. Turn off for gravity + toss physics.")]
        public bool floatObjects = true;
        [Tooltip("Guide line + cm label from the pinch point to the nearest grabbable, " +
                 "so you can judge depth on a flat monitor (green = close enough to grab).")]
        public bool showDistanceGuide = true;

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

            if (FindObjectOfType<Light>() == null)
            {
                var l = new GameObject("Sun").AddComponent<Light>();
                l.type = LightType.Directional;
                l.transform.rotation = Quaternion.Euler(50, -30, 0);
            }

            // --- hand rig (receiver + skeleton + grabber share one GameObject) ---
            var hand = new GameObject("HandRig");
            var recv = hand.AddComponent<HandStreamReceiver>();
            recv.port = udpPort;
            recv.mirrorX = mirrorX;
            hand.AddComponent<HandSkeleton>();
            hand.AddComponent<PinchGrabber>();
            if (showDistanceGuide) hand.AddComponent<GrabProximityIndicator>();

            if (followDeviceCamera)
            {
                var driver = hand.AddComponent<StreamedCameraDriver>();
                driver.receiver = recv;
                driver.driveMainCamera = true;   // HandRig carries the joints; camera follows
            }

            if (showVideo)
            {
                var video = hand.AddComponent<VideoStreamReceiver>();
                video.port = videoPort;
                video.mirrorX = mirrorX;   // video must mirror the same way as the hand
            }

            // --- table so released objects land somewhere (just below the shelf height) ---
            var table = GameObject.CreatePrimitive(PrimitiveType.Cube);
            table.name = "Table";
            table.transform.position = new Vector3(0, objectHeight - 0.05f, objectDistance);
            table.transform.localScale = new Vector3(1.2f, 0.02f, 0.8f);
            table.GetComponent<MeshRenderer>().material.color = new Color(0.25f, 0.28f, 0.32f);

            // --- grabbables (floating, or dropping a few cm onto the table) ---
            Spawn(PrimitiveType.Sphere, new Vector3(-0.12f, objectHeight, objectDistance),
                  0.06f, new Color(0.85f, 0.35f, 0.3f));
            Spawn(PrimitiveType.Cube, new Vector3(0.05f, objectHeight, objectDistance),
                  0.06f, new Color(0.3f, 0.55f, 0.85f));
            Spawn(PrimitiveType.Capsule, new Vector3(0.2f, objectHeight + 0.02f, objectDistance),
                  0.05f, new Color(0.4f, 0.8f, 0.45f));
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
        }
    }
}
