# Unity → iOS → iPad: what runs where

Two machines, one repo. The split is not a preference — Unity ships **no iOS Build
Support module for Linux**, and Xcode is macOS-only. So the Linux box trains and
exports, the Mac builds and installs, and git is the handoff.

```
┌──────────────────────────────┐          ┌────────────────────────────────┐
│ LINUX  (this machine)        │          │ MAC                            │
│ jucpark-device               │          │                                │
│                              │          │                                │
│  conda env handmesh_etr      │          │  Unity Hub                     │
│  PyTorch / CUDA training     │          │   └ Unity 6000.3.18f1          │
│  ONNX export                 │          │       + iOS Build Support ★    │
│  PC inference server         │  git     │  Xcode (+ Command Line Tools)  │
│                              │ ───────▶ │  Claude Code                   │
│  Unity Editor: NOT needed    │  push    │   └ agent: unity-ios-deploy    │
│  Xcode:        NOT possible  │  /pull   │                                │
│                              │          │  iPad ── USB ──┘               │
└──────────────────────────────┘          └────────────────────────────────┘
```

★ The one thing people get wrong: **the Unity *Editor* must be installed on the Mac,
with the iOS Build Support module.** The Unity *project* (`unity/DepthRefinement/`)
lives in this git repo and therefore exists on both machines — but only the Mac can
turn it into an Xcode project.

---

## Linux side — what you do here

Nothing iOS-specific. This machine's job ends at "a validated `.onnx` is committed".

```bash
# 1. train / fine-tune as usual
python scripts/train/train_hybrid.py --exp hybrid_B --mode locked

# 2. export to ONNX -- the export scripts already write into the Unity project
#    (--out defaults to unity/DepthRefinement/Assets/Models/<slot>.onnx)
python scripts/export/export_hybrid_fastvit_onnx.py \
    --ckpt mobrecon_ckpt/hybrid_sa12_replay_ft_bb_uv2d/best.pt

# 3. verify what is about to ship (opset, input/output shapes, git diff)
bash scripts/run/ship_model_to_unity.sh

# 4. hand off
git add unity/DepthRefinement/Assets/Models/
git commit -m "chore(unity): update hybrid_fastvit ONNX"
git push
```

If an ONNX was produced somewhere else, copy it into a slot first —
`bash scripts/run/ship_model_to_unity.sh /tmp/foo.onnx hybrid_fastvit` — rather than
renaming files by hand.

`Assets/Models/*.onnx` are tracked in git (16–46 MB each), so `git pull` on the Mac is
the whole transfer. Keep the filename: the sibling `.meta` holds the GUID that
`DepthRefinement.unity` references, and a rename silently unbinds the model.

**Also on Linux**, for the server-inference mode, the PC runs the pose server the iPad
talks to over the local network:

```bash
python scripts/infer/infer_ipad_stream.py    # iPad streams RGB+depth here, gets joints back
```

That is a runtime pairing, unrelated to building — the iPad needs the app installed
first, and both machines have to be on the same network.

### What is *not* possible on Linux

| | |
|---|---|
| Unity iOS Build Support module | not offered by Unity Hub on Linux |
| Xcode / `xcodebuild` / `xcrun devicectl` | macOS-only |
| Code signing, provisioning profiles | macOS-only (needs the keychain) |
| Installing to a USB-attached iPad | macOS-only |

Unity on **Windows** can generate the Xcode project, but compiling and signing it
still needs a Mac — so that path just adds a third machine.

---

## Mac side — one-time setup

1. **Xcode** from the App Store, then launch it once and accept the license.
   ```bash
   xcode-select --install          # command line tools
   sudo xcodebuild -license accept
   ```
2. **Unity Hub** → install **exactly `6000.3.18f1`** (it is pinned in
   `ProjectSettings/ProjectVersion.txt`; a different version will silently upgrade and
   dirty the project). During install tick **iOS Build Support**. If Unity is already
   installed without it: Hub → Installs → gear → *Add modules*.
   Default location, which `build_ios.sh` expects:
   ```
   /Applications/Unity/Hub/Editor/6000.3.18f1/Unity.app/Contents/MacOS/Unity
   ```
   Anywhere else: pass `UNITY=/path/to/Unity`.
3. **Sign in to Unity** once in the Hub GUI. Batchmode builds fail with
   `No valid Unity Editor license found` if you skip this.
4. **Apple ID in Xcode** → Settings → Accounts → add the Apple ID, select the team.
   Note the **Team ID** (Manage Certificates / developer.apple.com → Membership).
   A free Apple ID works; builds then expire after 7 days.
5. **Clone the repo** and connect the iPad by USB, unlock it, tap **Trust**.
   ```bash
   git clone https://github.com/JeungChulPark/handmesh_etr.git
   cd handmesh_etr
   bash scripts/run/build_ios.sh devices    # should list the iPad
   ```

## Mac side — every build

```bash
git pull
TEAM_ID=<your team id> bash scripts/run/build_ios.sh
```

That runs three stages, and you can stop at any of them:

| Stage | What happens | Typical time |
|---|---|---|
| `unity` | `Unity -batchmode -executeMethod BuildiOS.Build` → `unity/DepthRefinement/build/Unity-iPhone.xcodeproj` | 10–25 min clean, 2–5 min with `APPEND=1` |
| `archive` | `xcodebuild archive` + `exportArchive` → `build-artifacts/export/*.ipa` | 3–8 min |
| `install` | `xcrun devicectl device install app` + launch on the iPad | seconds |

```bash
TEAM_ID=... APPEND=1 bash scripts/run/build_ios.sh          # fast iteration
TEAM_ID=... DEV_BUILD=1 bash scripts/run/build_ios.sh       # profiler + debugger
TEAM_ID=... bash scripts/run/build_ios.sh archive           # build an .ipa, don't install
DEVICE=<udid> bash scripts/run/build_ios.sh install         # pick one of several iPads
```

Drop `APPEND=1` whenever packages, player settings or native plugins changed — an
appended build reuses the old Xcode project and will not pick those up.

### With Claude on the Mac

The `unity-ios-deploy` agent (`.claude/agents/unity-ios-deploy.md`) wraps all of the
above: it knows the project's pinned versions, the bundle id, the model slots, and the
recurring signing/licensing failures. On the Mac:

> build the Unity app and put it on the iPad

It checks `uname -s`, refuses to pretend on a non-Mac, runs the script, and reads the
Unity/xcodebuild logs when something fails. Long IL2CPP builds run in the background.

---

## First-run gotchas

- **`No signing certificate "iOS Development" found`** — `TEAM_ID` alone is not enough
  on a fresh Mac. Open Xcode → Settings → Accounts once, then re-run.
- **`The developer is not trusted`** on the iPad — Settings → General → VPN & Device
  Management → trust. Free provisioning also expires after 7 days; reinstall.
- **App installs but never reaches the PC server** — that is the iOS Local Network
  permission. `Assets/Editor/IOSPlistPostprocessor.cs` injects
  `NSLocalNetworkUsageDescription` on every build so the prompt appears; if the
  postprocessor stops running, the traffic fails silently with no error.
- **Depth always zero** — the LiDAR path needs an iPad Pro; `environmentDepth` returns
  nothing on non-LiDAR devices.
- **IL2CPP killed mid-compile** — close the Unity Editor GUI, it holds several GB.

## Build outputs (never committed)

```
unity/DepthRefinement/build/               generated Xcode project
unity/DepthRefinement/build-artifacts/     .xcarchive, ExportOptions.plist, *.ipa
unity/DepthRefinement/Library/             Unity import cache
```
