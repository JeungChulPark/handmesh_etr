#!/usr/bin/env bash
# Unity -> Xcode -> iPad, in one command.  macOS only (Unity's iOS Build Support and
# Xcode do not exist on Linux).  Run from the repo root:
#
#   TEAM_ID=ABCDE12345 bash scripts/run/build_ios.sh              # full: build + install
#   TEAM_ID=... bash scripts/run/build_ios.sh unity               # stop after the Xcode project
#   TEAM_ID=... bash scripts/run/build_ios.sh archive             # ... + .xcarchive/.ipa
#   bash scripts/run/build_ios.sh install                         # install the last .ipa only
#   bash scripts/run/build_ios.sh devices                         # list attached iPads
#
# Environment:
#   TEAM_ID      Apple Developer Team ID (required for archive/install; Xcode >
#                Settings > Accounts shows it)
#   DEVICE       target device UDID (default: the single connected device)
#   UNITY        path to the Unity binary (default: the version in ProjectVersion.txt)
#   APPEND=1     reuse the existing Xcode project -> much faster incremental IL2CPP
#   DEV_BUILD=1  Unity development player (profiler + managed debugger)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="$ROOT/unity/DepthRefinement"
BUILD="$PROJECT/build"                 # generated Xcode project (gitignored)
ARTIFACTS="$PROJECT/build-artifacts"   # .xcarchive / .ipa (gitignored)
SCHEME="Unity-iPhone"
BUNDLE_ID="${BUNDLE_ID:-com.ETRI.DepthRefinement}"
STAGE="${1:-all}"

UNITY_VERSION="$(sed -n 's/^m_EditorVersion: //p' "$PROJECT/ProjectSettings/ProjectVersion.txt")"
UNITY="${UNITY:-/Applications/Unity/Hub/Editor/$UNITY_VERSION/Unity.app/Contents/MacOS/Unity}"

say() { printf '\n\033[1;36m[build_ios] %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[build_ios] %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "macOS required. Unity ships no iOS Build Support for Linux and Xcode is macOS-only."

# ---------------------------------------------------------------- devices
list_devices() { xcrun devicectl list devices; }

resolve_device() {
    if [ -n "${DEVICE:-}" ]; then echo "$DEVICE"; return; fi
    local ids
    ids="$(xcrun devicectl list devices --quiet --json-output /dev/stdout 2>/dev/null \
           | /usr/bin/python3 -c 'import json,sys
d=json.load(sys.stdin)["result"]["devices"]
for x in d:
    p=x.get("connectionProperties",{})
    if p.get("tunnelState") != "unavailable":
        print(x["identifier"])' || true)"
    [ -n "$ids" ] || die "no connected device. Unlock the iPad, plug it in, tap Trust, then: bash scripts/run/build_ios.sh devices"
    [ "$(echo "$ids" | wc -l)" -eq 1 ] || die "multiple devices connected; pick one with DEVICE=<udid>:
$ids"
    echo "$ids"
}

# ---------------------------------------------------------------- 1. Unity
unity_build() {
    [ -x "$UNITY" ] || die "Unity $UNITY_VERSION not found at $UNITY (install it, with the iOS Build Support module, from Unity Hub; or set UNITY=...)"
    local extra=()
    [ "${APPEND:-0}" = "1" ]    && extra+=(-append)
    [ "${DEV_BUILD:-0}" = "1" ] && extra+=(-development)
    [ -n "${TEAM_ID:-}" ]       && extra+=(-teamId "$TEAM_ID")

    say "Unity $UNITY_VERSION -> Xcode project ($BUILD)"
    "$UNITY" -quit -batchmode -nographics \
        -projectPath "$PROJECT" \
        -buildTarget iOS \
        -executeMethod HandMesh.DepthRefinement.Editor.BuildiOS.Build \
        -buildPath "$BUILD" \
        -bundleId "$BUNDLE_ID" \
        "${extra[@]}" \
        -logFile - | sed -n '/\[BuildiOS\]/p;/error CS/p;/Exception/p'
    [ -d "$BUILD/$SCHEME.xcodeproj" ] || die "Unity produced no Xcode project at $BUILD"
}

# ---------------------------------------------------------------- 2. Xcode
xcode_archive() {
    [ -n "${TEAM_ID:-}" ] || die "TEAM_ID is required to sign. Xcode > Settings > Accounts > Manage Certificates shows it."
    mkdir -p "$ARTIFACTS"
    local archive="$ARTIFACTS/$SCHEME.xcarchive"
    rm -rf "$archive"

    say "xcodebuild archive"
    xcodebuild -project "$BUILD/$SCHEME.xcodeproj" \
        -scheme "$SCHEME" \
        -configuration Release \
        -destination 'generic/platform=iOS' \
        -archivePath "$archive" \
        -allowProvisioningUpdates \
        DEVELOPMENT_TEAM="$TEAM_ID" \
        CODE_SIGN_STYLE=Automatic \
        archive

    # 'development' is the pre-Xcode-15.3 spelling of what is now called 'debugging';
    # Xcode still accepts it, and it keeps this working on older toolchains.
    cat > "$ARTIFACTS/ExportOptions.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>method</key><string>development</string>
  <key>teamID</key><string>$TEAM_ID</string>
  <key>signingStyle</key><string>automatic</string>
  <key>stripSwiftSymbols</key><true/>
  <key>compileBitcode</key><false/>
</dict>
</plist>
PLIST

    say "xcodebuild exportArchive -> .ipa"
    rm -rf "$ARTIFACTS/export"
    xcodebuild -exportArchive \
        -archivePath "$archive" \
        -exportOptionsPlist "$ARTIFACTS/ExportOptions.plist" \
        -exportPath "$ARTIFACTS/export" \
        -allowProvisioningUpdates
    ls -lh "$ARTIFACTS/export"/*.ipa
}

# ---------------------------------------------------------------- 3. iPad
install_app() {
    local device app ipa
    device="$(resolve_device)"
    # devicectl installs an .app bundle; take it from the archive (no unzip needed).
    app="$(ls -d "$ARTIFACTS/$SCHEME.xcarchive/Products/Applications/"*.app 2>/dev/null | head -1 || true)"
    if [ -z "$app" ]; then
        ipa="$(ls "$ARTIFACTS/export"/*.ipa 2>/dev/null | head -1 || true)"
        [ -n "$ipa" ] || die "nothing to install -- run the archive stage first"
        rm -rf "$ARTIFACTS/payload" && mkdir -p "$ARTIFACTS/payload"
        unzip -q "$ipa" -d "$ARTIFACTS/payload"
        app="$(ls -d "$ARTIFACTS/payload/Payload/"*.app | head -1)"
    fi

    say "installing $(basename "$app") -> $device"
    xcrun devicectl device install app --device "$device" "$app"
    say "launching"
    xcrun devicectl device process launch --device "$device" "$BUNDLE_ID" || \
        say "install OK; launch failed -- on a free Apple ID, trust the developer first: iPad > Settings > General > VPN & Device Management"
}

case "$STAGE" in
    devices) list_devices ;;
    unity)   unity_build ;;
    archive) unity_build; xcode_archive ;;
    install) install_app ;;
    all)     unity_build; xcode_archive; install_app ;;
    *)       die "usage: bash scripts/run/build_ios.sh [all|unity|archive|install|devices]" ;;
esac
say "done ($STAGE)"
