# Multitarget tracking on nuScenes

Public project page and research presentation for Bai Liping’s `RFS_Filters` work.

- `index.html`: responsive research narrative, results table, source links, and categorized archive.
- `presentation.html`: 20 regular Bento slides. The original-sequence viewer immediately follows its explanatory slide; it has a complete static fallback for printing.
- `live/`: GIF viewer with automatic playback on selection at 2× speed, infinite looping, a stop button, deterministic initial state, and the existing Bento keyboard/lifecycle bridge. No tracking algorithm runs in the browser.
- `assets/figures.json`: all 73 image entries from the supplied Notion page, preserving source block IDs and original captions. Some images are repeated in the source.
- `assets/validation-results.txt`: the final metrics from the attached `0.761.log`, with old local machine paths omitted.

Build from this directory:

```sh
node build-page.mjs
node build-deck.mjs
```

The deck builder reuses the checked-in, licensed Bento runtime from `../et-handover/index.html` without modifying its compressed payloads. No new dependencies are required. Keep authoring changes in `deck.mjs`, `build-page.mjs`, and the local styles/scripts rather than editing generated HTML.

Sources:

- User-supplied Notion: https://www.notion.so/2270d664735d8183bd8bc130601992e2
- Code: https://github.com/BaiLiping/RFS_Filters (reviewed at `7d7b9bce1846a67c7fe48d9e10236a8b30d7a9c5`)
- Metric definitions: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/tracking/README.md

The featured log identifies `v1.0-trainval`, `val`, 6,019 samples, and an October 12, 2022 run. The score is an archived result, not a new reproduction or test-set result. The detector checkpoint and complete tracker settings cannot be established from the log. The earlier 0.707 → 0.69 observation comes from the narrative notes and is not the same experiment. Method labels on paired MHT / TO-PMB images are retained from the source; no overall performance ranking is claimed.

Original figure files remain unchanged in `assets/`. The builders create `assets/playback/*-2x.gif` from those originals by changing only frame delays and infinite-loop metadata; palettes, pixels, frame ordering, and disposal methods remain intact. The project-page hero and presentation cover use the user-selected crossing GIF. All displayed and full-size GIF links use these faster looping variants. Expiring media URLs and raw Notion record metadata are not published. The external Google Drive collection and large JSON/ZIP attachments remain linked through the original notes.
