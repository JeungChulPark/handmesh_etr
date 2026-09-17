using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Mirrors the camera's rendered image left↔right — video background, 3D hand and
    /// grabbables together — without touching world coordinates. Use it when the device
    /// camera faces the user (iPad rear camera pointed at you while you watch the PC
    /// monitor): an un-mirrored view then moves opposite to your hand, like a webcam
    /// without "mirror" mode.
    ///
    /// Unlike <see cref="HandStreamReceiver.mirrorX"/> (which negates joint X and breaks a
    /// world-locked moving camera), this only flips the projection, so the ARKit pose,
    /// grabbing and distances stay physically correct. Built-in render pipeline: the
    /// projection is re-mirrored every frame in OnPreCull (StreamedCameraDriver keeps
    /// changing the FOV) and triangle winding is inverted while rendering.
    /// IMGUI labels are drawn un-mirrored (readable) at the mirrored screen positions,
    /// because WorldToScreenPoint uses the mirrored projection.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class MirroredCameraView : MonoBehaviour
    {
        [Tooltip("Flip the rendered image horizontally")]
        public bool mirror = true;

        Camera _cam;

        void Awake() => _cam = GetComponent<Camera>();

        void OnPreCull()
        {
            _cam.ResetProjectionMatrix();   // picks up the current fieldOfView / aspect
            if (mirror)
                _cam.projectionMatrix = _cam.projectionMatrix * Matrix4x4.Scale(new Vector3(-1f, 1f, 1f));
        }

        void OnPreRender()
        {
            if (mirror) GL.invertCulling = true;   // mirrored winding would cull front faces
        }

        void OnPostRender() => GL.invertCulling = false;

        void OnDisable()
        {
            if (_cam != null) _cam.ResetProjectionMatrix();
            GL.invertCulling = false;
        }
    }
}
