#!/usr/bin/env python3
"""Apply the reviewed 2026-09 frame-registration correction set.

Edits only the article, its Bento document (not its runtime), and the three
slide demos. Prefix matches are checked, and a version marker makes this
one-time migration idempotent. The deployed HTML remains directly editable.
"""
import base64
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = '<!-- frame-registration-review: 2026-09-07-v1 -->'


def swap(s, old, new, count=None):
    n = s.count(old)
    if n == 0 or (count is not None and n != count):
        raise ValueError(f'Expected {count or "at least one"} occurrence(s), got {n}: {old[:100]}')
    return s.replace(old, new)


def paragraph(s, prefix, body):
    pattern = r'(<p\b[^>]*>)\s*' + re.escape(prefix) + r'.*?</p>'
    s, n = re.subn(pattern, lambda m: m[1] + body + '</p>', s, flags=re.S)
    if n != 1:
        raise ValueError(f'Paragraph prefix matched {n} times: {prefix}')
    return s


ARTICLE = json.loads((ROOT / 'scripts/frame-registration-article.json').read_text())
SLIDE_CONTENT = json.loads((ROOT / 'scripts/frame-registration-slides.json').read_text())
SLIDE_EDITS = SLIDE_CONTENT['edits']
SLIDE_NOTES = SLIDE_CONTENT['notes']
TABLE_ROWS = SLIDE_CONTENT['table']


def equation_png(formula):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 1.2))
    fig.text(.01, .5, '$' + formula + '$', fontsize=24, va='center', color='#20232b')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=.08, transparent=True, dpi=130)
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def main():
    article = ROOT / 'frame-registration/index.html'
    if MARK in article.read_text():
        print('Frame-registration review already applied.'); return
    text = article.read_text()
    for prefix, body in ARTICLE:
        text = paragraph(text, prefix, body)
    text = swap(text, r'\arg\min_{R,\,t}', r'\arg\min_{R\in SO(d),\,t\in\mathbb R^d}', 1)
    text = swap(text, r'N \;=\; \frac{\log(1-p)}{\log\!\big(1-w^{s}\big)}', r'N \;=\; \left\lceil\frac{\log(1-p)}{\log\!\big(1-w^{s}\big)}\right\rceil', 1)
    text = swap(text, r'p(x)\;\propto\;', r'a_c(x)\;=\;', 1)
    text = swap(text, r'\tfrac{M}{N}}.', r'\tfrac{M}{V}}.', 1)
    text = swap(text, 'NDT: trade points for sufficient statistics', 'NDT: approximate local geometry with Gaussian cells', 1)
    text = swap(text, 'Why smoothness is the whole argument', 'Objective smoothness is not a convergence guarantee', 1)
    text = swap(text, 'The finale · one scene, three memories', 'The finale · one scene, four inference representations', 1)
    text = swap(text, 'Walk all three', 'Walk all four')
    text = swap(text, 'Soft assignment (CPD-style)', 'Annealed soft targets (toy)')
    text = swap(text, 'PMBM data association', 'Hypothesis mixture (toy)')
    # Clarify the comparison table without claiming that association removes all derivatives.
    text = swap(text, 'no — assignments flip discretely', 'piecewise smooth; assignment switches may be nonsmooth')
    text = swap(text, 'no — same flips, faster basin', 'smooth for fixed matches/normals; switching can be nonsmooth')
    text = swap(text, 'yes — EM on a mixture', 'smooth fixed-bandwidth Gaussian model; demo uses a schedule')
    text = swap(text, 'NN search — $O(m\\log n)$ with a k-d tree', 'typically $O(m\\log n)$ with an index; demo uses brute force')
    text = swap(text, 'all pairs, Gaussian-weighted, width from EM or a schedule + outlier bin', 'Gaussian soft targets with scheduled width and a heuristic miss score')
    last_table = r'''<table class="cmp"><thead><tr><th>Demo</th><th>Model and state</th><th>Update</th><th>Important limitation</th></tr></thead><tbody>
<tr><td>Similarity CPD</td><td>Moving Gaussian components; responsibilities retained for display</td><td>Weighted similarity fit and variance estimation</td><td>Rigid CPD instead fixes scale to 1; full matrix storage is optional</td></tr>
<tr><td>FilterReg-inspired</td><td>Fixed mixture; soft targets from a sparse Cartesian grid</td><td>One planar twist step and a bandwidth schedule</td><td>Not the paper's permutohedral implementation; analytic variance updates are possible in FilterReg</td></tr>
<tr><td>MMD-Reg-inspired</td><td>48 frequencies, 96 features; empirical mean residual</td><td>LM with a kernel schedule</td><td>Nonconvex; unweighted nonoverlap and clutter affect the optimum</td></tr>
<tr><td>Learned-flow toy</td><td>Embedded network, evolving coordinates, and feature signatures</td><td>Endpoint prediction, optional rigidity projection each step, final pose readout</td><td>Not RAP; the toy is not evidence about RAP's accuracy, scaling, or generalization</td></tr>
</tbody></table>'''
    text, n = re.subn(r'<table class="cmp">\s*<thead><tr><th>Design choice</th>.*?</table>', lambda m: last_table, text, flags=re.S)
    if n != 1: raise ValueError('Missing final comparison table')
    # Introductory notation and reading guidance, kept outside any existing demo container.
    note = r'''<div class="scope-note" id="registration-scope"><strong>Conventions and reading path.</strong> Source → target: $T(p)=Rp+t$. The demos use $SE(2)$ (one rotation, two translations); 3-D uses $SE(3)$ (three rotational and three translational degrees of freedom). A similarity additionally estimates scale. $c_i$ denotes an association, $h$ a grid-cell size, $\sigma$ a mixture standard deviation, $\ell$ a kernel length scale, and $D$ a frequency count (giving $2D$ features). The CPD subsection uses the original letters $y_m$ for source and $x_n$ for target; the MMD subsection explicitly reverses them. World units are arbitrary, not automatically meters. <strong>Keep separate: low objective, numerical stopping, and correct pose.</strong> Begin with Kabsch/ICP, then read the NDT and mixture details. Reviewed 7 September 2026.</div>'''
    pos = text.index('</p>', text.index('<p class="premise">')) + 4
    text = text[:pos] + '\n' + note + text[pos:]
    # UI diagnostics use stopped, never imply a successful registration from a tolerance alone.
    def script_labels(m):
        return m[0].replace('converged', 'stopped').replace("name: 'PMBM assoc', abbr: 'PMBM'", "name: 'hypothesis toy', abbr: 'hyp. toy'")
    text = re.sub(r'<script>.*?</script>', script_labels, text, flags=re.S)
    captions = {
        'CAP1': [
            'Two unpaired clouds with planted outliers: source orange must align with target blue.',
            'Moving points carry Gaussian components. A clutter component competes with them for each fixed observation.',
            'E-step: each fixed point splits probability across moving centroids and clutter. The web shows inlier responsibilities.',
            'M-step: compute responsibility-weighted centroids for both clouds.',
            'M-step: solve the proper rotation from weighted cross-covariance.',
            'M-step: estimate translation and similarity scale. Rigid CPD instead fixes s = 1.',
            'Variance: estimate sigma squared from the weighted residuals. It need not decrease every sweep.',
            'Result: the loop stops at its tolerance or budget; compare the pose, not just the stopping flag.',
        ],
        'CAP2': [
            'The fixed cloud carries the mixture; transformed moving points are its queries.',
            'Filter: accumulate a sparse Cartesian Gaussian grid, rebuilt when bandwidth changes. This is not a permutohedral lattice.',
            'Query: compute a soft target and confidence from filtered moments of the reversed mixture, not a row of the displayed CPD matrix.',
            'Twist: one local Gauss–Newton step per sweep with a bandwidth schedule. The FilterReg paper also permits optimized variance.',
        ],
        'CAP3': [
            'Encode each point using a fixed bank of 48 random frequencies.',
            'Average into 96 sine/cosine features per cloud. Outliers remain in the mean; no clutter component is present.',
            'Compare feature means. The squared residual approximates Gaussian-kernel MMD squared; finite features need not identify a distribution uniquely.',
            'Optimize the nonconvex residual with LM and a kernel schedule. A small step is not proof of correct alignment.',
        ],
        'CAP4': [
            'Embedded 15,458-parameter MLP and cloud signatures: a planar learned-flow analogy, not the RAP transformer or its evaluation.',
            'Flow: evolving coordinates follow predicted endpoints over kappa steps. This still performs inference and uses runtime state.',
            'Rigidify: project predicted coordinates onto a rigid fit. Optional per-step projection changes the trajectory.',
            'Read the pose using paired original and transported points. Correct algebra does not certify the learned endpoint.',
        ],
    }
    for name, values in captions.items():
        text, n = re.subn(r'const ' + name + r'=\[.*?\];', lambda m: 'const ' + name + '=' + json.dumps(values, ensure_ascii=False) + ';', text, flags=re.S)
        if n != 1: raise ValueError('Missing walkthrough captions: ' + name)
    text = swap(text, 'auto=[false,false,false];', 'auto=[false,false,false,false];')
    # A fixed-denominator capped cost is actually descended by the plotted gated ICP update.
    old = '''const ds = land.scene.source.map(p => nearest(applyT(T, p), land.scene.target).d)
    .filter(d => d < GATE);
  let s = 0;
  for(const d of ds) s += d * d;
  return ds.length ? s / ds.length : GATE * GATE;'''
    new = '''const ds = land.scene.source.map(p => nearest(applyT(T, p), land.scene.target).d);
  let s = 0;
  for(const d of ds) s += Math.min(d * d, GATE * GATE);
  return ds.length ? s / ds.length : Infinity;'''
    text = swap(text, old, new, 1).replace('gated NN cost', 'capped NN cost')
    text = swap(text, "'<td>' + runs.filter(r => r.deg < 5).length + ' / ' + runs.length + '</td>' +", "'<td>' + runs.filter(r => r.deg < 5).length + ' / ' + runs.length + '</td>' +\n        '<td>' + runs.filter(r => r.deg < 5 && r.dist < 0.2).length + ' / ' + runs.length + '</td>' +", 1)
    # Header is matched below after verifying its exact source wording.
    text = swap(text, '<th>success &lt;5°</th>', '<th>rotation &lt;5°</th><th>joint &lt;5° / &lt;0.2 u</th>', 1)
    text = swap(text, 'colspan="5"', 'colspan="6"', 1)
    # Keep grid keys collision-free outside the canvas too.
    text = swap(text, 'return (ix + 1024) * 2048 + (iy + 1024);', "return ix + ',' + iy;", 2)
    text = swap(text, "const ix = Math.floor(fld.hoverKey / 2048) - 1024, iy = fld.hoverKey % 2048 - 1024;", "const [ix, iy] = fld.hoverKey.split(',').map(Number);", 1)
    # Original snapshots of all 25 runs are no longer asserted in prose or slide graphics.
    text = text.replace('<!-- FRAME_REVIEW_MARK -->', '')
    outputs = [article, ROOT / 'frame-registration-slides/index.html',
               ROOT / 'frame-registration-slides/live/app.js', ROOT / 'frame-registration-slides/live/index.html']
    backups = {p: p.read_bytes() for p in outputs}
    try:
        patch_slides()
        patch_live()
        article.write_text(text.replace('</body>', MARK + '\n</body>'))
    except Exception:
        for p, content in backups.items(): p.write_bytes(content)
        raise
    print('Updated article, Bento slides, and live-demo descriptions/diagnostics.')


# Slide patches are kept separate below so the 20-slide sequence and runtime stay intact.
def patch_slides():
    path = ROOT / 'frame-registration-slides/index.html'
    raw = path.read_text()
    match = re.search(r'(<script[^>]*id="bento-doc"[^>]*>)(.*?)(</script>)', raw, re.S)
    if not match: raise ValueError('Missing Bento document')
    deck = json.loads(match[2]); byid = {s['id']: s for s in deck['slides']}
    def put(sid, eid, html, size=None):
        el = next(e for e in byid[sid]['elements'] if e['id'] == eid)
        el['html'] = html
        if size: el['fontSize'] = size
    def notes(sid, body): byid[sid]['notes'] = body
    # Exact IDs are checked against the stored document by the regression tests.
    edits = SLIDE_EDITS
    for sid, changes in edits.items():
        for eid, body in changes.items(): put(sid, eid, body)
    for sid, body in SLIDE_NOTES.items(): notes(sid, body)
    formulas = {
        'eqimg-fr-kabsch': r'H=U\Lambda V^\top,\quad \hat R=VCU^\top,\quad C=\mathrm{diag}(1,\ldots,1,\det(VU^\top))',
        'eqimg-fr-ransac': r'N=\left\lceil\frac{\log(1-p)}{\log(1-w^s)}\right\rceil',
        'eqimg-fr-cpd': r'P(m\mid x_n)=\frac{g_{mn}}{\sum_{k=1}^M g_{kn}+c},\quad g_{mn}=\exp\!\left(-\frac{\|x_n-T(y_m)\|^2}{2\sigma^2}\right)',
    }
    for slide in deck['slides']:
        for e in slide['elements']:
            if e['id'] in formulas:
                e['src'] = equation_png(formulas[e['id']])
    # Replace an unproven historical leaderboard with a reproducible evaluation guide.
    slide = byid['s-bench']
    slide['elements'] = [e for e in slide['elements'] if e['type'] != 'chart']
    template = dict(next(e for e in slide['elements'] if e['id'] == 'bench-take'))
    template.update(id='benchmark-guide', x=112, y=234, w=1056, h=350, fontSize=24,
        html='<b>Use the article’s “Benchmark 25 seeds” button.</b><br><br>Same clouds and local starts; RANSAC ignores initialization.<br>Seeds: 40 + 13k, k = 0,…,24. Budget: 70 steps per solver.<br><br><b>Report both errors:</b> RRE (rotation) and RTE (translation).<br>Joint success: RRE &lt; 5° <b>and</b> RTE &lt; 0.2 world units.<br><br>Compare noise, outliers, and starts one at a time.<br>These toy solvers are not official paper implementations.')
    slide['elements'].append(template)
    # Comparison table: CPD storage is an implementation choice, not a lower bound.
    table = next(e for e in byid['s-table']['elements'] if e['type']=='table')
    for row, vals in zip(table['rows'], TABLE_ROWS):
        for cell, val in zip(row['cells'], vals): cell['html'] = val
    deck['title'] = 'Frame Registration — Models, Objectives, and Guarantees'
    raw = raw[:match.start(2)] + json.dumps(deck, ensure_ascii=False, separators=(',', ':')) + raw[match.end(2):]
    raw = raw.replace('<title>bento/slides</title>', '<title>Frame registration — methods and limitations | Bai Liping</title>')
    path.write_text(raw.replace('</body>', MARK + '\n</body>'))


def patch_live():
    p = ROOT / 'frame-registration-slides/live/app.js'; s = p.read_text()
    replacements = {
        'Navigate a likelihood surface': 'Explore a point-to-cell score',
        'NDT replaces discrete target points with Gaussian cells. Registration becomes optimization over a smooth-ish score landscape.': 'Gaussian cells give a piecewise-smooth score. This demo uses translation-only direct search, not the article’s Newton solver.',
        'high-likelihood translations': 'high-score translations',
        'Choose a start on the likelihood map': 'Choose a start on the score map',
        'Converged inside the correct basin': 'Stopped: within the demo pose thresholds',
        'Settled in a local minimum — reset and reposition': 'Stopped without meeting pose thresholds — try another start',
        'Peak found near the correct translation': 'Stopped near the true translation; check rotation separately',
        'Local peak found — try another bright basin': 'Stopped away from truth — try another start',
        'return total / Math.max(1, scene.goodCount);': 'return total / Math.max(1, scene.source.length);',
    }
    for old,new in replacements.items(): s=swap(s,old,new)
    p.write_text(s)
    p = ROOT / 'frame-registration-slides/live/index.html'; s=p.read_text()
    s=swap(s,'Click or drag on the score map to choose a translation, then climb toward a local maximum.', 'Translation-only direct search on four shifted grids (cell size 1.3 u; covariance regularization 0.026 u²). Rotation stays fixed; stopping is not a maximum certificate.')
    s=swap(s,'Drag the blue scan to give ICP a start.', 'Gated point-to-point ICP also trims the worst 16% of surviving pairs. Drag the blue scan to set its start.')
    s=swap(s,'2-D teaching model · same rigid-transform mathematics as the 3-D case', '2-D teaching model · SE(2) has 3 pose parameters; SE(3) has 6')
    s=swap(s,'optimizes the scan’s likelihood landscape', 'explores a point-to-cell score by translation-only direct search')
    p.write_text(s)


if __name__ == '__main__': main()
