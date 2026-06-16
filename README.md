# Rotary Second-Hand Catalog

Static English catalog modeled after `moving-sale-items.netlify.app`.

## Edit items

Update `data/items.js`. The data is grouped by seller:

- `status`: `available`, `reserved`, or `sold`
- `images`: one or more image URLs
- `price`: display text, so currencies can be written exactly as needed

The Feishu link provided by the user is stored in `sourceDocument`, but it was not readable without Feishu login from this environment.

## Deploy on Netlify

Use this folder as the publish directory. No build command is required.
