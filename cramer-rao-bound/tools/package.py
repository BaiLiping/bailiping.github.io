#!/usr/bin/env python3
"""Package the standalone deck, guide, labs, sources, and published exports."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'Cramer-Rao-Bound-site-package.zip'
excluded = {'node_modules', 'exports', '__pycache__', '.DS_Store'}
files = [p for p in sorted(ROOT.rglob('*')) if p.is_file()
         and p != target and not (excluded & set(p.relative_to(ROOT).parts))]
with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
    for source in files:
        archive.write(source, Path(ROOT.name) / source.relative_to(ROOT))
with ZipFile(target) as archive:
    assert archive.testzip() is None
print(f'Packaged {len(files)} files: {target.name} ({target.stat().st_size:,} bytes)')
