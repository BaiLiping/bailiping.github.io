#!/usr/bin/env python3
"""Idempotent final polish after the checked frame-registration migration."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def replace_checked(text, old, new, count=1):
    if old not in text:
        if new not in text:
            raise ValueError('Missing expected text: ' + old[:80])
        return text
    if text.count(old) != count:
        raise ValueError('Unexpected match count: ' + old[:80])
    return text.replace(old, new)

article = ROOT / 'frame-registration/index.html'
s = article.read_text()
s = replace_checked(s, 'none — one ${2*D_MMD}-number residual',
                    'a ${2*D_MMD}-number residual; no pair assignments')
s = replace_checked(s, 'nothing — 15,458 weights',
                    '15,458 weights plus evolving coordinates and signatures')
s = replace_checked(s, "this page's solver code (repo)",
                    'historical companion code (may differ)')
s = replace_checked(s,
    '// cellKey(ix, iy): packs integer cell coordinates into one Map key. Takes two cell indices,\n// returns an integer unique for |ix|, |iy| < 1024.',
    '// cellKey(ix, iy): joins integer cell coordinates into a collision-free string Map key.', 2)
s = replace_checked(s, 'Frame registration · the whole story, live',
                    'Frame registration · models, objectives, and live examples')
s = replace_checked(s, 'Rung four · correspondences dissolved',
                    'Rung four · point-to-cell association')
if 'id="demo-implementation-scope"' not in s:
    body = '<div class="scope-note" id="demo-implementation-scope"><strong>Scope of the implementations.</strong> The controls run planar solvers on synthetic clouds. Nearest-neighbor search is brute force; the article’s NDT examples use hard Gaussian cells and damped Newton, whereas the separate slide lab uses translation-only direct search. The race’s soft-target and hypothesis-mixture panels are heuristics, not complete CPD or PMBM implementations. The final walkthrough separately demonstrates similarity CPD, Cartesian-grid filtering inspired by FilterReg, random-feature MMD optimization, and a small learned-flow analogy. These examples do not reproduce the original papers’ implementations, training, or evaluation datasets. Scene controls change noise, density, and outliers to illustrate model mismatch and optimization behavior. Compare pose errors under the same controls, rather than reading the browser race as a general ranking.</div>'
    s, n = re.subn(r'<div class="scope-note"><strong>Scope of the demos\.</strong>.*?</div>',
                   lambda _: body, s, flags=re.S)
    if n != 1: raise ValueError('Expected exactly one legacy implementation scope note')
if '// makePMBM(): soft assignment via global hypotheses' in s:
    comment = '// makePMBM(): legacy function name for a hypothesis-mixture teaching heuristic.\n// Randomized greedy sweeps propose one-to-one assignments, with a flat heuristic miss\n// score. Their weighted pair marginals drive the shared annealed rigid fitting step.\n// No Poisson or Bernoulli posterior, Murty ranking, or temporal propagation is implemented.\n// This is not a PMBM filter. Returns a {reset, step, T} closure object.\nfunction makePMBM(){'
    s, n = re.subn(r'// makePMBM\(\): soft assignment via global hypotheses.*?function makePMBM\(\)\{',
                   lambda _: comment, s, flags=re.S)
    if n != 1: raise ValueError('Expected exactly one legacy hypothesis-toy comment')
assert 'PMBM-flavored' not in s
assert "reproducing the paper's stress settings" not in s
article.write_text(s)

p = ROOT / 'frame-registration-slides/index.html'
raw = p.read_text()
m = re.search(r'(<script[^>]*id="bento-doc"[^>]*>)(.*?)(</script>)', raw, re.S)
if not m: raise ValueError('Missing Bento document')
deck = json.loads(m[2])
slide = next(s for s in deck['slides'] if s['id'] == 's-cpd')
geometry = {
    'eqbox': {'h':110},
    'eqimg-fr-cpd': {'x':200,'y':200,'w':880,'h':101},
    'eqcap': {'y':313},
    'colLhead': {'y':352}, 'colRhead': {'y':352},
    'colLbody': {'y':390,'h':284}, 'colRbody': {'y':390,'h':284},
}
for name, props in geometry.items():
    next(e for e in slide['elements'] if e['id'] == name).update(props)
p.write_text(raw[:m.start(2)] + json.dumps(deck, ensure_ascii=False, separators=(',', ':')) + raw[m.end(2):])

# Primary arXiv metadata and CVF proceedings identify FilterReg as CVPR 2019.
p = ROOT / 'frame-registration/REVIEW.md'
s = p.read_text()
s = replace_checked(s, 'Registration Using Gaussian Filter and Twist Parameterization*, ICRA, 2019.',
                    'Registration Using Gaussian Filter and Twist Parameterization*, CVPR, 2019.')
s = replace_checked(s, 'python scripts/frame-registration-audit.py\n',
                    'python scripts/frame-registration-audit.py\npython scripts/finalize-frame-registration.py\n') if 'python scripts/finalize-frame-registration.py' not in s else s
p.write_text(s)
print('Final captions, CPD equation layout, and FilterReg citation verified.')
