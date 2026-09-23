"""Render the manuscript's TikZ factor graph as a self-contained vector asset.

Requires pdflatex (with standalone/TikZ) and Poppler's pdftocairo.
Only n_p -> n^t and n_g -> n^g are changed to match Appendix I.
"""
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile

SOURCE = Path(__file__).resolve().parent
original = (SOURCE / "graph-drawing-original.tex").read_bytes()
assert sha256(original).hexdigest() == "307c379677b4c13f28ebf9a0c02bec7029b2bd44515fac97c6027a7a126f86aa"
graph = original.decode().replace("^{n_p}", "^{n^t}").replace("^{n_g}", "^{n^g}")
wrapper = r"""\documentclass[tikz,border=6pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\usetikzlibrary{arrows,arrows.meta,shapes.arrows,calc,backgrounds,intersections}
\newcommand{\V}[1]{\boldsymbol{#1}}
\definecolor{plotD}{HTML}{FF7F0E}
\begin{document}
\input{graph.tex}
\end{document}
"""
output = SOURCE.parent / "assets" / "fg-grbp.svg"
with tempfile.TemporaryDirectory(prefix="grbp-factor-graph-") as directory:
    work = Path(directory)
    (work / "graph.tex").write_text(graph)
    (work / "figure.tex").write_text(wrapper)
    run = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure.tex"],
        cwd=work, capture_output=True, text=True,
    )
    if run.returncode:
        raise RuntimeError(run.stdout)
    subprocess.run(["pdftocairo", "-svg", str(work / "figure.pdf"), str(output)], check=True)
svg = output.read_text().replace(
    "<defs>",
    "<title>Factor graph for GrBP</title>\n"
    "<desc>Legacy prior and likelihood chains connect through target association variables "
    "to the consistency factor psi. Group association variables connect psi to newborn "
    "likelihoods and states. Blue connections carry prediction between scans.</desc>\n<defs>",
    1,
)
output.write_text(svg)
print(f"Rendered {output.name} from the verified manuscript TikZ source.")
