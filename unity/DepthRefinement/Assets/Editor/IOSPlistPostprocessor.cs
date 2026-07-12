#if UNITY_IOS
using System.IO;
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEditor.iOS.Xcode;

namespace HandMesh.DepthRefinement.Editor
{
    /// <summary>
    /// Injects iOS privacy keys into the exported Xcode project's Info.plist on every build,
    /// so they never have to be re-added by hand in Xcode.
    ///
    /// NSLocalNetworkUsageDescription is REQUIRED for the server-inference mode
    /// (RgbdStreamer's TCP push to the PC + ServerHandProvider's UDP receive): on iOS 14+
    /// any local-network traffic silently fails until the user grants the Local Network
    /// permission, and the grant prompt only appears when this key is present.
    /// (The camera permission comes from Player Settings > cameraUsageDescription as before.)
    /// </summary>
    public static class IOSPlistPostprocessor
    {
        [PostProcessBuild]
        public static void OnPostprocessBuild(BuildTarget target, string pathToBuiltProject)
        {
            if (target != BuildTarget.iOS) return;

            string plistPath = Path.Combine(pathToBuiltProject, "Info.plist");
            var plist = new PlistDocument();
            plist.ReadFromFile(plistPath);

            plist.root.SetString(
                "NSLocalNetworkUsageDescription",
                "실시간 손 포즈 추론을 위해 같은 네트워크의 PC 서버로 카메라 영상과 LiDAR 데이터를 전송합니다.");

            plist.WriteToFile(plistPath);
            UnityEngine.Debug.Log($"[IOSPlistPostprocessor] NSLocalNetworkUsageDescription added to {plistPath}");
        }
    }
}
#endif
