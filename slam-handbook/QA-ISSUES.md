# Outstanding release QA findings — 20 October 2026

## Mathematical issue: chapter 6 observability demo
The `crlb` numerical model currently uses
`information = count / sigma² + strength`. When the "Relative geometry strength"
slider is zero, the model still produces a finite variance bound. This contradicts
the chapter's observability explanation and worked check, which state that an
unobservable direction has zero Fisher information and no ordinary inverse CRLB.

Correct the example to use measurement sensitivity in the Fisher information, e.g.
`information = count * strength / sigma²`. For zero strength, show the bound
as "undefined" (not an infinite numeric plot point) and plot finite Fisher
information rather than an invented finite variance bound. Re-run all 216 cases.

## Browser verification
The original project has passed a scripted DOM harness but not a real browser.
A minimal Playwright smoke test is included at `qa-browser.cjs`. On a machine
with Node.js and Chromium Playwright installed, from the repository root run:
`python3 -m http.server 8765`, then
`node slam-handbook/qa-browser.cjs`.
Also manually verify each slider and reset button, mobile notebook export,
offline reload after an online visit, keyboard access and horizontal overflow.

## Mobile notebook export
`style-responsive.css` hides `#notesExport` at widths <= 720px, and the
reader does not provide a second export button in the chapter notebook. Add a
visible notebook-level export affordance for phone readers.

Do not publish the guide to the public homepage until these issues are
addressed and QA evidence is recorded in `PROGRESS.md`.
