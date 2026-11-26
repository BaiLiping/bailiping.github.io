#!/usr/bin/env python3
"""Build the shareable Cars page and its list-only preview from the homepage."""
from hashlib import sha256
from html import escape, unescape
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'cars'
URL = 'https://bailiping.com/cars/'


def preview_image(names):
    image = Image.new('RGB', (1200, 630), 'white')
    draw = ImageDraw.Draw(image)
    font_file = ROOT / 'assets/fonts/source-serif-4-400-700-latin.woff2'
    row = min(80, 510 // len(names))
    font = ImageFont.truetype(str(font_file), round(row * .70))
    width = max(draw.textlength(name, font=font) for name in names)
    if width > 1020:
        font = font.font_variant(size=int(font.size * 1020 / width))
        width = max(draw.textlength(name, font=font) for name in names)
    x = (image.width - width) / 2
    y = (image.height - (len(names) - 1) * row) / 2
    for name in names:
        draw.text((x, y), name, font=font, fill='#1a2332', anchor='lm')
        y += row
    image.save(DEST / 'preview.png', optimize=True)


def main():
    home = (ROOT / 'index.html').read_text()
    header = re.search(r'<header class="site-header">.*?</header>', home, re.S)[0]
    header = header.replace('id="tab-cars"', 'id="tab-cars" aria-current="page"')
    section = re.search(r'<section class="group" id="cars".*?</section>', home, re.S)[0]
    names = [re.sub(r'^\d{4} | \([^()]+\)$', '', unescape(name))
             for name in re.findall(r'<strong>([^<]+)</strong>', section)]
    if not names:
        raise ValueError('The Cars section has no model names')
    DEST.mkdir(exist_ok=True)
    preview_image(names)
    image_version = sha256((DEST / 'preview.png').read_bytes()).hexdigest()[:10]
    image_url = f'{URL}preview.png?v={image_version}'
    listing = escape(' · '.join(names), quote=True)
    footer = re.search(r'<footer class="site-footer">.*?</footer>', home, re.S)[0]
    nav_script = re.search(r'<script src="/assets/home\.js[^"]*"></script>', home)[0]
    page = f'''<!doctype html>
<!-- Generated from the homepage by scripts/build-cars-preview.py. -->
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Cars | Bai Liping</title>
    <meta name="description" content="{listing}">
    <link rel="canonical" href="{URL}">
    <meta property="og:type" content="website">
    <meta property="og:title" content="Cars">
    <meta property="og:description" content="{listing}">
    <meta property="og:url" content="{URL}">
    <meta property="og:image" content="{image_url}">
    <meta property="og:image:type" content="image/png">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:image:alt" content="{listing}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="Cars">
    <meta name="twitter:description" content="{listing}">
    <meta name="twitter:image" content="{image_url}">
    <meta name="twitter:image:alt" content="{listing}">
    <link rel="icon" href="/favicon.ico">
    <link rel="stylesheet" href="/assets/home.css?v=cars-route-1">
  </head>
  <body>
    {header}
    <main>
      {section}
      {footer}
    </main>
    {nav_script}
  </body>
</html>
'''
    (DEST / 'index.html').write_text(page)


if __name__ == '__main__':
    main()
