using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace HandMesh.DepthRefinement.Editor
{
    /// <summary>
    /// Headless iOS build entry point, so the Xcode project can be produced from a
    /// terminal (and therefore from CI or an agent) instead of the Editor UI:
    ///
    ///   Unity -quit -batchmode -nographics \
    ///         -projectPath unity/DepthRefinement \
    ///         -buildTarget iOS \
    ///         -executeMethod HandMesh.DepthRefinement.Editor.BuildiOS.Build \
    ///         -buildPath build -teamId ABCDE12345 -logFile -
    ///
    /// Recognised arguments (all optional):
    ///   -buildPath &lt;dir&gt;    where to emit the Xcode project (default: "build",
    ///                        relative to the Unity project root)
    ///   -teamId &lt;id&gt;        Apple Developer Team ID; also enables automatic signing
    ///   -bundleId &lt;id&gt;      override the application identifier
    ///   -development         Unity development player (profiler + script debugging)
    ///   -append              reuse the existing Xcode project instead of replacing it,
    ///                        which keeps incremental IL2CPP compiles fast
    ///
    /// Exits non-zero on failure so a shell script can branch on it; Unity's own
    /// batchmode exit code does not otherwise reflect a failed BuildPlayer.
    /// </summary>
    public static class BuildiOS
    {
        public static void Build()
        {
            string buildPath = Arg("-buildPath") ?? "build";
            string teamId = Arg("-teamId") ?? Environment.GetEnvironmentVariable("TEAM_ID");
            string bundleId = Arg("-bundleId") ?? Environment.GetEnvironmentVariable("BUNDLE_ID");
            bool development = Flag("-development");
            bool append = Flag("-append");

            if (!Path.IsPathRooted(buildPath))
            {
                // Unity's CWD in batchmode is not the project root, so anchor it there.
                buildPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", buildPath));
            }

            string[] scenes = EnabledScenes();
            if (scenes.Length == 0)
            {
                Fail("No enabled scenes in Build Settings -- nothing to build.");
                return;
            }

            // iPad is a target device, and the LiDAR/ARKit path needs the device SDK.
            PlayerSettings.iOS.targetDevice = iOSTargetDevice.iPhoneAndiPad;
            PlayerSettings.iOS.sdkVersion = iOSSdkVersion.DeviceSDK;

            if (!string.IsNullOrEmpty(bundleId))
            {
                PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.iOS, bundleId);
            }

            if (!string.IsNullOrEmpty(teamId))
            {
                // Automatic signing lets xcodebuild -allowProvisioningUpdates mint the
                // development profile; without a team here Xcode opens the project unsigned.
                PlayerSettings.iOS.appleDeveloperTeamID = teamId;
                PlayerSettings.iOS.appleEnableAutomaticSigning = true;
            }

            var options = new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = buildPath,
                target = BuildTarget.iOS,
                targetGroup = BuildTargetGroup.iOS,
                options = BuildOptions.None,
            };
            if (development) options.options |= BuildOptions.Development | BuildOptions.AllowDebugging;
            if (append) options.options |= BuildOptions.AcceptExternalModificationsToPlayer;

            Log($"scenes={string.Join(", ", scenes)}");
            Log($"buildPath={buildPath}");
            Log($"bundleId={PlayerSettings.GetApplicationIdentifier(NamedBuildTarget.iOS)}");
            Log($"teamId={(string.IsNullOrEmpty(teamId) ? "<unset>" : teamId)} " +
                $"autoSign={PlayerSettings.iOS.appleEnableAutomaticSigning} " +
                $"development={development} append={append}");

            BuildReport report = BuildPipeline.BuildPlayer(options);
            BuildSummary summary = report.summary;

            if (summary.result != BuildResult.Succeeded)
            {
                foreach (BuildStep step in report.steps)
                {
                    foreach (BuildStepMessage msg in step.messages)
                    {
                        if (msg.type == LogType.Error || msg.type == LogType.Exception)
                        {
                            Debug.LogError($"[BuildiOS] {step.name}: {msg.content}");
                        }
                    }
                }
                Fail($"build {summary.result} after {summary.totalTime} " +
                     $"({summary.totalErrors} errors, {summary.totalWarnings} warnings)");
                return;
            }

            Log($"OK in {summary.totalTime}, {summary.totalSize / (1024 * 1024)} MB -> {buildPath}");
            EditorApplication.Exit(0);
        }

        private static string[] EnabledScenes()
        {
            var scenes = new List<string>();
            foreach (EditorBuildSettingsScene s in EditorBuildSettings.scenes)
            {
                if (s.enabled) scenes.Add(s.path);
            }
            return scenes.ToArray();
        }

        private static string Arg(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == name) return args[i + 1];
            }
            return null;
        }

        private static bool Flag(string name)
        {
            return Array.IndexOf(Environment.GetCommandLineArgs(), name) >= 0;
        }

        private static void Log(string msg) => Debug.Log($"[BuildiOS] {msg}");

        private static void Fail(string msg)
        {
            Debug.LogError($"[BuildiOS] {msg}");
            EditorApplication.Exit(1);
        }
    }
}
