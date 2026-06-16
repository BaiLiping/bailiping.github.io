# bailiping.com

GitHub Pages site for `bailiping.com`.

## Structure

- `/` is the main website entry point.
- `/sales/` is the static English second-hand sale catalog.
- `/handover/` is the target-handover project page with paper, repository, and result animation links.

## Edit sale items

Update `sales/data/items.js`. The data is grouped by seller:

- `status`: `available`, `reserved`, or `sold`
- `images`: one or more image URLs
- `price`: display text, so currencies can be written exactly as needed

The Feishu link provided by the user is stored in `sourceDocument`, but it was not readable without Feishu login from this environment.
