#!/usr/bin/env bash
# LINUX SIDE of the iOS pipeline: check that the ONNX in the Unity project is the one
# you meant to ship, then hand it to the Mac over git.  No Unity or Xcode needed here.
#
#   bash scripts/run/ship_model_to_unity.sh                      # verify what's there now
#   bash scripts/run/ship_model_to_unity.sh <exported.onnx> [slot]   # copy one in first
#
# The export scripts already write straight into Assets/Models:
#   python scripts/export/export_hybrid_fastvit_onnx.py --ckpt mobrecon_ckpt/<exp>/best.pt
#   python scripts/export/export_hybrid_b_onnx.py       --ckpt mobrecon_ckpt/<exp>/best.pt
# so the copy argument is only for an ONNX produced somewhere else.
#
# Filenames in Assets/Models must not change: the sibling .meta holds the GUID that
# DepthRefinement.unity references, so a rename silently unbinds the model from the
# scene.  This script therefore only ever overwrites an existing slot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS="$ROOT/unity/DepthRefinement/Assets/Models"
SRC="${1:-}"
SLOT="${2:-hybrid_fastvit}"

say() { printf '\n\033[1;36m[ship_model] %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[ship_model] %s\033[0m\n' "$*" >&2; exit 1; }

if [ -n "$SRC" ]; then
    [ -f "$SRC" ] || die "no such file: $SRC"
    DST="$MODELS/$SLOT.onnx"
    [ -f "$DST" ] || die "unknown slot '$SLOT'. Existing slots:
$(ls "$MODELS"/*.onnx | xargs -n1 basename | sed 's/\.onnx$//')"
    say "$(basename "$DST"): $(du -h "$DST" | cut -f1) -> $(du -h "$SRC" | cut -f1)"
    cp "$SRC" "$DST"
fi

say "models currently in the Unity project"
python - "$MODELS" <<'PY'
import pathlib, sys, onnx
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.onnx")):
    try:
        m = onnx.load(str(p))
        onnx.checker.check_model(m)
        shape = lambda t: [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
        print(f"  {p.name:<28} {p.stat().st_size/1e6:6.1f} MB  opset {m.opset_import[0].version}")
        for i in m.graph.input:
            print(f"      in   {i.name:<20} {shape(i)}")
        for o in m.graph.output:
            print(f"      out  {o.name:<20} {shape(o)}")
    except Exception as e:
        print(f"  {p.name:<28} INVALID: {e}")
        raise SystemExit(1)
PY

say "git state"
cd "$ROOT"
git status --porcelain -- unity/DepthRefinement/Assets/Models/ || true

if git diff --quiet -- unity/DepthRefinement/Assets/Models/ && \
   git diff --cached --quiet -- unity/DepthRefinement/Assets/Models/; then
    echo "  (no model change to ship -- Assets/Models matches HEAD)"
    exit 0
fi

cat <<EOF

Hand it to the Mac:

    git add unity/DepthRefinement/Assets/Models/
    git commit -m "chore(unity): update on-device ONNX"
    git push

then on the Mac, in the repo:

    git pull
    TEAM_ID=<team> APPEND=1 bash scripts/run/build_ios.sh

EOF
