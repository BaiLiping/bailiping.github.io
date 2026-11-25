# Factor graphs and the sum-product algorithm

A 14-slide Bento reading of Kschischang, Frey and Loeliger (2001), focused on Example 1, equations (2)–(6), and the five-step schedule in Figure 7 / Section II-D.

- Public entry: `/factor-graphs/`, linked under Random thoughts.
- `deck.mjs`: authored slide content, diagrams, notes and inline-live mapping.
- `build.mjs`: replaces the readable payload in the existing VI/EM Bento shell, preserves its compressed runtime, and produces `index.html` and `study.html`.
- `model.js`: pure finite-state factor/message arithmetic and independent exhaustive inference.
- `graph.js`: graph coordinates and renderer, shared by the static and live versions.
- `live/`: schedule stepper, message inspector and factor-sum workbench.

Run `node factor-graphs/build.mjs` and `node --test factor-graphs/test.cjs` from the site root. Serve the repository over HTTP for the live regions. No build dependencies are required. Math uses the existing pinned MathJax 3.2.2 integration.

The graph and message schedule match the paper. Binary alphabets and all numerical factor values are explicitly labeled teaching choices. The fixed matrix rows index x3; columns index x4 or x5. BP keeps unnormalized messages to match the paper. The schedule opens at its central exchange (step 3); Reset starts at the leaves. Selecting an arrow reveals its input vectors, contributing factors and term-by-term calculation for either output state. Selecting a variable explains its normalized incoming-message product. Intermediate beliefs are explicitly labeled partial, with the number of received inputs; they become exact once every input arrives. Every final marginal is checked against all 32 joint configurations.

Source: https://doi.org/10.1109/18.910572. The locally supplied licensed PDF is not redistributed.
