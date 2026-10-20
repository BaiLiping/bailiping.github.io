# SLAM Handbook interactive companion

Original learning material that accompanies the *SLAM Handbook: From Localization and Mapping to Spatial Intelligence* (compiled 21 April 2026; editors L. Carlone, A. Kim, T. Barfoot, D. Cremers and F. Dellaert). The handbook is linked to, not copied or redistributed.

The reader covers **all 18 numbered chapters** with a three-idea primer, **three computation-backed parameter experiments** (54 total), two or more worked self-checks, explicit section and printed-page references, a warning about simplified model assumptions, a private browser notebook and a chapter progress tracker.

No package manager or third-party JavaScript library is required. Load `index.html` from a web server at `/slam-handbook/`. Numerical models are in `model-01.js` through `model-18.js`; lessons are in `chapters-1.js`, `chapters-2.js`, `chapters-3.js`; the reader is `app.js`.

Run `node slam-handbook/test-models.cjs` from the repository root to validate chapters, default/boundary/midpoint parameter scenarios, curves and metrics. That automated test exercises 216 parameter combinations but does not replace visual browser and device testing.

After one successful online visit, the service worker should cache the learning guide and experiments for offline use. The linked original book PDF is **not** cached. Notes and completion state remain in local browser storage; `Export notebook` downloads a Markdown backup. No notes leave the browser.

See `PROGRESS.md` for integration status, verification evidence and publication instructions. Preserve other site content when linking this guide under Random thoughts.
