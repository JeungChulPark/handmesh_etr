using UnityEngine;

namespace HandMesh.HandGrab
{
    /// <summary>
    /// Mirrors the camera's finished image left↔right — video background, 3D hand and
    /// grabbables together — without touching world coordinates. Use it when the device
    /// camera faces the user (iPad rear camera pointed at you while you watch the PC
    /// monitor): an un-mirrored view then moves opposite to your hand, like a webcam
    /// without "mirror" mode.
    ///
    /// Implemented as a post-process blit (built-in pipeline OnRenderImage), NOT a
    /// negative-scale projection matrix: the projection trick left the Screen Space -
    /// Camera video canvas unrendered (recording 131905: hand + objects visible, video
    /// never). The blit flips whatever the camera drew, canvases included.
    ///
    /// IMGUI (OnGUI) is drawn after this and stays un-mirrored so text is readable; code
    /// that places a label at a world point must pass its screen X through
    /// <see cref="ScreenX"/>.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class MirroredCameraView : MonoBehaviour
    {
        [Tooltip("Flip the rendered image horizontally")]
        public bool mirror = true;

        /// <summary>True while a mirrored camera is rendering the view.</summary>
        public static bool Active { get; private set; }

        static readonly Vector2 FlipScale = new Vector2(-1f, 1f);
        static readonly Vector2 FlipOffset = new Vector2(1f, 0f);

        /// <summary>Screen X of a WorldToScreenPoint result, as it appears on the mirrored screen.</summary>
        public static float ScreenX(float x) => Active ? Screen.width - x : x;

        void OnEnable() => Active = mirror;

        void OnDisable() => Active = false;

        void OnRenderImage(RenderTexture src, RenderTexture dst)
        {
            Active = mirror;
            if (mirror) Graphics.Blit(src, dst, FlipScale, FlipOffset);
            else Graphics.Blit(src, dst);
        }
    }
}
