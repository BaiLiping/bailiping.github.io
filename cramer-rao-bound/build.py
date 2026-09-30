#!/usr/bin/env python3
"""Build the offline HTML from local sources. Python standard library only."""
from pathlib import Path
import json, re, html
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'src'
SOURCES={
'S1':'https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture15.pdf',
'S2':'https://arxiv.org/abs/1705.01064',
'S3':'https://arxiv.org/abs/1006.0888',
'S4':'https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture14.pdf'}
def main():
    deck=json.loads((SRC/'deck.json').read_text())
    formulas=json.loads((SRC/'formula-svg.json').read_text())
    assert len({s['id'] for s in deck})==len(deck),'Duplicate slide IDs'
    parts=[]
    for i,s in enumerate(deck):
        body=re.sub(r'(<div\b[^>]*data-formula="([^"]+)"[^>]*>)(</div>)',lambda m:m[1]+formulas[m[2]]+m[3],s['body'])
        assert 'data-formula' not in body or '<svg' in body
        title=html.escape(s['title']);sub=html.escape(s['subtitle'])
        if s['css']=='cover':
            heading=f'<header><p class="eyebrow">{html.escape(s["section"])}</p><p class="subtitle" style="font-size:18px;color:var(--ink)">Cramér–Rao Bound · Interactive slides</p></header>'
        else:
            heading=f'<header><p class="eyebrow">{html.escape(s["section"])}</p><h2>{title}</h2><p class="subtitle">{sub}</p></header>'
        refs=' · '.join(f'<a href="{SOURCES[k]}" aria-label="Source {k}">{k}</a>' for k in s['sources'])
        parts.append(f'<section class="slide {s["css"]} {"active" if i==0 else ""}" data-slide-id="{s["id"]}" aria-label="Slide {i+1}: {title}">{heading}<div class="body">{body}</div><footer class="slide-footer"><span class="site">BAI LIPING · RANDOM THOUGHTS</span><span>{html.escape(s["section"].split(" · ")[0].upper())}</span><span class="sources">{refs}</span><span class="page-no">{i+1:02d} / {len(deck)}</span></footer></section>')
    meta=json.dumps([dict(id=s['id'],title=s['title'],notes=s['notes']) for s in deck],ensure_ascii=False).replace('<','\\u003c')
    out='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Cramér–Rao Bound: Precision Has a Floor | Bai Liping</title><meta name="description" content="A 23-slide interactive guide to the Cramér–Rao Bound, Fisher information, estimator bias, and localization geometry, with three live, offline-capable labs."><link rel="canonical" href="https://bailiping.com/cramer-rao-bound/"><style>'''+(SRC/'style.css').read_text()+'''</style></head><body>
<div class="toolbar" role="navigation" aria-label="Presentation tools"><a class="brand" href="https://bailiping.com/">BLP</a><span class="title">CRAMÉR–RAO BOUND</span><button id="outline" title="Slide overview (O)">Overview</button><button id="notes-button" aria-expanded="false" aria-controls="notes" title="Presenter notes (N)">Notes</button><button id="read-mode">Read view</button><button id="print-deck">Print</button><button id="fullscreen" title="Fullscreen (F)">Fullscreen</button></div>
<main class="stage-wrap"><div class="stage" id="stage">'''+''.join(parts)+'''</div></main>
<nav class="nav" aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">←</button><span id="counter" aria-live="polite"></span><div id="position" aria-hidden="true"><div id="progress"></div></div><button id="next" aria-label="Next slide">→</button><span class="hint">← → navigate &nbsp; O overview &nbsp; N notes &nbsp; F fullscreen</span></nav>
<aside class="notes" id="notes" aria-label="Presenter notes"><button class="close small" id="notes-close" aria-label="Close notes">×</button><p class="kicker">Presenter notes</p><h3 id="notes-title"></h3><p id="notes-copy"></p></aside>
<dialog id="overview-dialog"><div class="toc-header"><h3>Slide overview</h3><button id="toc-close" aria-label="Close overview">×</button></div><div class="toc-grid" id="toc-grid"></div></dialog>
<noscript><p style="padding:20px">Enable JavaScript for navigation and interactive labs. The static PDF is an alternative.</p><style>.slide{display:flex}.stage-wrap,.stage{height:auto;display:block}.stage{transform:none}.toolbar,.nav{display:none}</style></noscript>
<script type="application/json" id="slide-meta">'''+meta+'''</script><script>'''+(SRC/'math.js').read_text()+'''</script><script>'''+(SRC/'app.js').read_text()+'''</script></body></html>'''
    (ROOT/'index.html').write_text(out)
    notes='# Cramér–Rao Bound — presenter notes\n\n23 slides · three live labs · approximately 30–40 minutes.\n\n'
    for i,s in enumerate(deck):
        notes+=f'## {i+1:02d}. {s["title"]}\n\n{s["subtitle"]}\n\n{s["notes"]}\n\n'
    (ROOT/'presenter-notes.md').write_text(notes)
    print(f'Built {ROOT/"index.html"}: {len(deck)} slides, {len(out):,} characters')
if __name__=='__main__':main()
