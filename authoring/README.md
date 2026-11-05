# Visual authoring for bailiping.com

Open **Bailiping Editor.command** on the Mac desktop, or run
`scripts/start-authoring.command` from this checkout. The launcher opens
`http://127.0.0.1:4317/__authoring/`. Keep its Terminal window running.

Choose a presentation, double-click text to edit, drag objects or their resize
handles, and use Bento's properties panel for typography and colours. The slide
thumbnails support reordering. Keep an introductory slide immediately before
its live demonstration.

**Save changes** (or Command-S) saves to this checkout. **Preview saved** opens
that saved presentation. Bento's **Slideshow** previews current unsaved edits
and runs the live demonstrations. The editing canvas shows each lab's fallback
image, so it can be positioned without an iframe taking the mouse input.

Equations render normally and reveal their LaTeX while editing their text box.
Use **Export draft** for a portable JSON backup; **Import draft** merges its
changes into the open deck and reports conflicting edits. Bento's own Save-as
menu exports an HTML copy; the authoring bar's Save changes is the website save.

Add comments with Bento's Comment tool (`C`), then **Copy feedback** and paste it
into the Codex chat. The copied text identifies this checkout, slide IDs,
selected objects, and unresolved comments. Ask Codex to review and publish the
saved edits. Saving locally does not push to GitHub or update the public site.
The public decks also include object and slide context for browser annotations
in supported ChatGPT desktop browsers.

## How edits survive rebuilding

The deck generator remains the source for scientific content and lab code.
`<deck>/authoring.json` is a second, version-controlled source: the changes made
in the visual editor, addressed by stable slide and object IDs. A save updates
both this file and the local `index.html`. Previous files are backed up under
`.authoring-backups/<deck>/`, which is excluded from Git.

Every Bento builder passes its final HTML through
`withAuthoring(html, import.meta.url)` from `scripts/authoring-build.mjs`. The
helper reapplies user edits to fresh generator output and updates live-slide
positions, mount bounds, and route aliases. Independent source changes merge;
an overlapping change stops the build with the exact property that needs
reconciliation. The saved edits are never discarded automatically.

The two formerly browser-generated decks, `likelihood-vs-density` and
`multidensity-fusion`, now have `build.mjs` entry points that package the same
shared Bento engine and their existing authoring modules at build time.

When editing a deck in code, read `authoring.json` first. Preserve IDs. Run the
deck's normal builder, check the resulting layout and lab navigation, and
commit its source, saved edits, and generated HTML together. Resolve reported
conflicts by reconciling the intended source and saved value. Do not bypass the
helper or delete edits just to get a green build.

The same helper links `assets/deck-theme.css`, the shared deck typography and
presentation frame: self-hosted Source Serif 4 replaces Georgia, IBM Plex Mono
replaces every monospace stack (most readers lack `SFMono-Regular`, so
eyebrows and footers used to fall back to Courier), rounded panels get a soft
shadow, and the slide sits on a warm surround instead of black letterboxing.
Change the look of every deck there rather than in individual generators.

New Bento builders need the same helper call. Run
`node scripts/install-authoring.mjs` to add annotation context and the deck
theme to an existing static deck without rebuilding its content. Non-Bento articles and the
homepage remain code-authored; browser feedback still works on those pages.

## Validation and recovery

Run `node --test tests/authoring.test.mjs` for merge, rebuild, session, and save
tests. Browser checks use `tests/authoring-browser.mjs` (Playwright supplied by
the caller) and operate in an isolated temporary checkout.

The Bento structural audit reports a same-origin sandbox warning for the two
module-authored decks. Their labs require this permission to import their
local ES modules; it is retained deliberately.

The server binds only to `127.0.0.1`, checks the request origin and session, and
rejects stale saves after a file changes on disk or in another tab. A rejected
save keeps the draft open: export it before reloading. Open the current deck
again and import that draft; if its properties overlap, ask Codex to reconcile
them using the exported file and current source.

Recover an earlier save from `.authoring-backups/` by restoring both
`index.html` and `authoring.json` from the same snapshot. A snapshot made before
the first visual save has no `authoring.json`; remove the newer sidecar when
restoring that snapshot. Rebuild and preview before publishing.
