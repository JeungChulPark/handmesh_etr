using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Moves this transform (and optionally the Main Camera) to the iPad's streamed
    /// ARKit camera pose ("cp"/"cq" from infer_ipad_stream.py, exposed by
    /// <see cref="HandStreamReceiver"/>), so the desktop scene is world-locked to the
    /// room the iPad tracks in. Put it on the SAME GameObject as HandStreamReceiver:
    /// the receiver maps joints from camera frame through this transform, so hands
    /// land at their real-world positions while the device moves.
    ///
    /// mirrorX is forced OFF — mirroring is only meaningful for a fixed, front-facing
    /// camera and breaks a world-locked moving one. Requires the v2 device stream
    /// (RgbdStreamer sends the pose); with a v1 stream nothing moves.
    /// </summary>
    [DefaultExecutionOrder(-50)]   // apply the pose BEFORE HandStreamReceiver maps joints
    public class StreamedCameraDriver : MonoBehaviour
    {
        [Tooltip("Receiver carrying the streamed pose (auto-found on this GameObject)")]
        public HandStreamReceiver receiver;

        [Tooltip("Also move Camera.main so the desktop view follows the device")]
        public bool driveMainCamera = true;

        [Tooltip("Exponential smoothing speed (1/s): ~20 tracks tightly, ~5 is " +
                 "cinematic, 0 = raw pose")]
        public float smoothing = 20f;

        bool _hasPose;
        Vector3 _pos;
        Quaternion _rot = Quaternion.identity;

        void OnEnable()
        {
            if (receiver == null) receiver = GetComponent<HandStreamReceiver>();
            if (receiver == null)
            {
                Debug.LogError("[StreamedCameraDriver] no HandStreamReceiver assigned/found");
                enabled = false;
                return;
            }
            if (receiver.mirrorX)
            {
                receiver.mirrorX = false;
                Debug.LogWarning("[StreamedCameraDriver] mirrorX disabled — a world-locked " +
                                 "camera cannot be mirrored");
            }
        }

        void Update()
        {
            if (!receiver.HasCameraPose) return;

            if (!_hasPose || smoothing <= 0f)
            {
                _pos = receiver.CameraPosition;
                _rot = receiver.CameraRotation;
                _hasPose = true;
            }
            else
            {
                float a = 1f - Mathf.Exp(-smoothing * Time.deltaTime);
                _pos = Vector3.Lerp(_pos, receiver.CameraPosition, a);
                _rot = Quaternion.Slerp(_rot, receiver.CameraRotation, a);
            }

            transform.SetPositionAndRotation(_pos, _rot);
            if (driveMainCamera && Camera.main != null && Camera.main.transform != transform)
                Camera.main.transform.SetPositionAndRotation(_pos, _rot);
        }
    }
}
