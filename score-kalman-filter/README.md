# The Score Kalman Filter

34 native Bento slides explaining [Iwasaki, Bloch, Lee and Ghaffari, arXiv:2605.16644v1](https://arxiv.org/html/2605.16644v1). The homepage lists the deck under Random Thoughts at `/score-kalman-filter/`.

## Edit and rebuild

Edit `bento-deck.mjs`, then run `node score-kalman-filter/build.mjs` from the repository root. The build reuses the existing native Bento renderer and local MathJax bundle. It generates `index.html`, `deck.json`, `live-demos.json` and `study.html`, and inserts the homepage link idempotently. Shared runtime assets remain unchanged.

## Lesson scope

The deck covers Eqs. (1)–(15), Algorithm 1, the information-form Kalman specialization, reported RMSE and runtime, and the limits of finite moment closure. Equation references link to version 1. The normalizability of cubic energies and the repeated-likelihood issue in optional refinement are explicitly labeled independent reading notes, not author corrections.

The three browser laboratories at `live/?lab=score`, `live/?lab=closure` and `live/?lab=update` implement scalar quartic score fitting, selected higher-moment Stein recurrences and one Gaussian likelihood update. They use numerical quadrature to supply synthetic reference moments and plots. That reference calculation is separate from the moment-only score fit and closure. These teaching examples do not reproduce the full SKF, truncated posterior recovery, or the paper's benchmark experiments.

## Attribution

`figures/se2-density-paper.png` reproduces Figure 1 without modification from the cited paper, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source image: <https://arxiv.org/html/2605.16644v1/fig_se2_prediction.png>. The slide and reading notes include author and source attribution. Selected Table A1 values are transcribed with their benchmark context. Native Bento and bundled dependency license notices are preserved in the generated page.
