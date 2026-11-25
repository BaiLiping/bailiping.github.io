#!/usr/bin/env python3
"""Stamp the engine-lab page's asset addresses with a version taken from the assets' contents.

Browsers may reuse a cached script, stylesheet or model for ten minutes after a deploy. If only some files are
reused, new code runs against an old page or an old model. Every local address therefore carries ?v=<stamp>,
and the stamp changes whenever any asset changes, so a deploy is fetched whole or not at all.

    python3 engine-lab/stamp.py           rewrite the stamps
    python3 engine-lab/stamp.py --check   exit 1 if they are out of date
"""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = [
    "comparison.css", "entry.js", "compare.js", "assembly.js", "flow.js",
    "engine930.json", "engine930.glb", "engineSC.json", "engineSC.glb", "preview-turbo.webp", "preview-sc.webp",
    "preview-turbo-exploded.webp", "preview-sc-exploded.webp",
]
CARRIERS = ["index.html", "entry.js", "compare.js"]          # the files that name those assets
STAMP = re.compile(rb"\?v=[0-9a-f]{10}")


def current_stamp():
    """Hash every asset with any existing stamp removed. Takes nothing. Returns ten hex characters."""
    digest = hashlib.sha1()
    for name in ASSETS:
        digest.update(STAMP.sub(b"?v=", (HERE / name).read_bytes()))
    return digest.hexdigest()[:10]


def stamped(name, stamp):
    """One carrier file's bytes with every stamp set. Takes the file name and the stamp. Returns bytes."""
    return STAMP.sub(b"?v=" + stamp.encode(), (HERE / name).read_bytes())


def main():
    """Rewrite the stamps, or with --check report whether they are current. Returns the exit status."""
    stamp = current_stamp()
    stale = [name for name in CARRIERS if stamped(name, stamp) != (HERE / name).read_bytes()]
    if "--check" in sys.argv:
        if stale:
            print("stale version stamps in: " + ", ".join(stale) + "; run python3 engine-lab/stamp.py")
        return 1 if stale else 0
    for name in stale:
        (HERE / name).write_bytes(stamped(name, stamp))
    print(f"stamp {stamp}: " + (", ".join(stale) + " updated" if stale else "already current"))
    return 0


sys.exit(main())
