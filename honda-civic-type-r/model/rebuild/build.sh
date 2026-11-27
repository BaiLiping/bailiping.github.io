#!/usr/bin/env bash
# run.sh plus a fine body mesh (mesh.py step 1, same field) passed to cutpanels.py as its 5th argument, so specs
# with "mesh": "fine" are cut from it, and the opened-state geometry (sides_openings.py: door / hood / hatch
# shut faces, inner frames, body jambs, seals, sill plates, B-pillars) passed to assemble.py as a parts file.
# When the rear worker's rear_specs.py / rear_parts.py / rear_badges.py are present they run too (as rear_run.sh). Usage: sides_run.sh <tag> [all|skip-field|parts-only]   (default all)
set -euo pipefail
cd "$(dirname "$0")"; export PATH="$HOME/.local/bin:$PATH"; PY=../../venv/bin/python; T=$1; MODE=${2:-all}
if [ "$MODE" = all ]; then $PY field.py env6.npz 0.006; fi
[ -f rear_specs.py ] && python3 rear_specs.py > /dev/null     # the rear worker's generated specs, when present
if [ "$MODE" != parts-only ]; then
  $PY build_body.py env6.npz body-$T.npz
  $PY mesh.py body-$T.npz body-$T.glb 1.0 2
  $PY mesh.py body-$T.npz body-$T-fine.glb 1.0 1
  $PY cutpanels.py body-$T-mesh.npz body-$T.npz cut_upper.json,cut_front.json,cut_mid.json,cut_rear.json,cut_low.json cut-$T.npz body-$T-fine-mesh.npz > cut-$T.log 2>&1 || { tail -5 cut-$T.log; exit 1; }
  tail -3 cut-$T.log
fi
$PY parts3v.py parts_front.json,parts_rear.json,parts_mid.json body-$T.npz parts-$T.npz > parts-$T.log 2>&1 || { tail -5 parts-$T.log; exit 1; }
cat parts-$T.log
EXTRA=""                                                        # the rear worker's lamp / vent / badge parts
if [ -f rear_parts.py ]; then $PY rear_parts.py body-$T.npz rearparts-$T.npz > rearparts-$T.log 2>&1 || { tail -5 rearparts-$T.log; exit 1; }; EXTRA="$EXTRA,rearparts-$T.npz"; fi
if [ -f rear_badges.py ]; then $PY rear_badges.py body-$T.npz badges-$T.npz > badges-$T.log 2>&1 || { tail -5 badges-$T.log; exit 1; }; EXTRA="$EXTRA,badges-$T.npz"; fi
$PY sides_openings.py cut-$T.npz body-$T.npz openings-$T.npz > openings-$T.log 2>&1 || { tail -5 openings-$T.log; exit 1; }
tail -1 openings-$T.log
blender -b --python assemble.py -- ../render-fix/final.glb cut-$T.npz $PWD/car-$T.glb --parts parts-$T.npz$EXTRA,openings-$T.npz > asm-$T.log 2>&1 || { tail -5 asm-$T.log; exit 1; }
grep -E "removed|wrote" asm-$T.log
node ../gltfx/compress_model.mjs car-$T.glb car-$T.draco.glb && cp car-$T.draco.glb ../../site/_preview/rb-$T.glb
ls -la car-$T.draco.glb | awk '{print "draco", $5}'
cd ../render-fix
POSES='[{"id":"fq","theta":-0.75,"phi":1.38,"distance":9},{"id":"rq","theta":2.4,"phi":1.35,"distance":9},{"id":"rear","theta":3.1416,"phi":1.45,"distance":9},{"id":"side","theta":-1.5708,"phi":1.45,"distance":10}]' \
  timeout 600 node look.cjs rb-$T 1280x800 "/_preview/rb-$T.glb" 2>&1 | tail -1
