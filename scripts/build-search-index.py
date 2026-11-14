#!/usr/bin/env python3
"""Build search discovery files from the site's tracked, public HTML pages."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, urljoin


ROOT = Path(__file__).resolve().parent.parent
SITE = "https://bailiping.com"
DIRECTORY = Path("site-map/index.html")
SUPPORT_DIRS = {"assets", "vendor", "node_modules", "slides-assets", "scripts", "tests", "test-results"}


class Head(HTMLParser):
    """Read only head metadata; SVG titles in the body are not page titles."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_head = False
        self.in_title = False
        self.title = ""
        self.description = ""
        self.canonical = ""
        self.redirect = False
        self.noindex = False

    def handle_starttag(self, tag, attrs):
        if tag == "head":
            self.in_head = True
        if not self.in_head:
            return
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "link" and "canonical" in attrs.get("rel", "").lower().split():
            self.canonical = attrs.get("href", "")
        elif tag == "meta":
            name = attrs.get("name", "").lower()
            content = attrs.get("content", "")
            if name == "description":
                self.description = content
            if name in {"robots", "googlebot"}:
                self.noindex |= bool({"noindex", "none"} & set(re.split(r"[\s,]+", content.lower())))
            if attrs.get("http-equiv", "").lower() == "refresh":
                self.redirect = True

    def handle_endtag(self, tag):
        if tag == "head":
            self.in_head = False
        elif tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_head and self.in_title:
            self.title += data


@dataclass(frozen=True)
class Page:
    path: Path
    url: str
    title: str
    description: str

    @property
    def label(self):
        return re.sub(r"\s*[|·—–-]\s*Bai Liping$", "", self.title)


def discover():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.html"], cwd=ROOT
    ).decode().split("\0")
    pages, skipped = [], []
    for name in sorted(filter(None, tracked)):
        path = Path(name)
        if path == DIRECTORY:
            continue
        if any(part.startswith((".", "_")) or part in SUPPORT_DIRS for part in path.parts):
            skipped.append((name, "support file"))
            continue
        if not (ROOT / path).is_file():
            continue
        head = Head()
        head.feed((ROOT / path).read_text(encoding="utf-8"))
        title = " ".join(head.title.split())
        if not title or head.redirect or head.noindex:
            reason = "no page title" if not title else "redirect" if head.redirect else "noindex"
            skipped.append((name, reason))
            continue
        route = "/" + path.as_posix()
        if path.name == "index.html":
            route = route[:-len("index.html")]
        url = SITE + quote(route, safe="/")
        if head.canonical and urljoin(url, head.canonical) != url:
            skipped.append((name, "canonical points elsewhere"))
            continue
        pages.append(Page(path, url, title, " ".join(head.description.split())))
    if not pages or not any(page.url == SITE + "/" for page in pages):
        raise ValueError("The public homepage must be present in the search index")
    return pages, skipped


def sitemap(pages):
    urls = sorted({page.url for page in pages} | {SITE + "/site-map/"})
    entries = "\n".join(f"  <url>\n    <loc>{escape(url)}</loc>\n  </url>" for url in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + entries + "\n</urlset>\n")


def directory(pages):
    groups = defaultdict(list)
    for page in pages:
        if page.path.as_posix() != "index.html":
            groups[page.path.parts[0]].append(page)
    sections = []
    for topic, entries in groups.items():
        primary = next((page for page in entries if page.path == Path(topic) / "index.html"), None)
        label = primary.label if primary else topic.replace("-", " ").title()
        heading = escape(label)
        if primary:
            heading = f'<a href="{escape(primary.url.removeprefix(SITE))}">{heading}</a>'
        parts = [f'      <section aria-labelledby="topic-{escape(topic)}">',
                 f'        <h2 id="topic-{escape(topic)}">{heading}</h2>']
        if primary and primary.description:
            parts.append(f"        <p>{escape(primary.description)}</p>")
        companions = sorted((page for page in entries if page != primary), key=lambda page: page.label.casefold())
        if companions:
            parts.append("        <ul>")
            for page in companions:
                parts.append(f'          <li><a href="{escape(page.url.removeprefix(SITE))}">{escape(page.label)}</a></li>')
            parts.append("        </ul>")
        parts.append("      </section>")
        sections.append((label.casefold(), "\n".join(parts)))
    content = "\n".join(section for _, section in sorted(sections))
    return '''<!doctype html>
<!-- Generated by scripts/build-search-index.py. -->
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>All pages | Bai Liping</title>
    <meta name="description" content="Browse Bai Liping's research presentations, articles, interactive demonstrations, study notes, and personal collections.">
    <link rel="canonical" href="https://bailiping.com/site-map/">
    <link rel="icon" href="/favicon.ico">
    <meta property="og:type" content="website">
    <meta property="og:site_name" content="Bai Liping">
    <meta property="og:title" content="All pages | Bai Liping">
    <meta property="og:description" content="Browse Bai Liping's research presentations, articles, interactive demonstrations, study notes, and personal collections.">
    <meta property="og:url" content="https://bailiping.com/site-map/">
    <meta property="og:image" content="https://bailiping.com/assets/home/social-card.jpg">
    <meta name="twitter:card" content="summary_large_image">
    <style>
      /* Same voice as the homepage (assets/home.css): serif headings, one green accent. */
      @font-face { font-family: 'Source Serif 4'; font-style: normal; font-weight: 400 700; font-display: swap; src: url(/assets/fonts/source-serif-4-400-700-latin.woff2) format('woff2'); }
      :root { color-scheme: light; --ink: #1a2332; --body: #374151; --muted: #636b78; --accent: #2f6b4f; --accent-hover: #1e5039; --line: #e2e4e7; }
      * { box-sizing: border-box; }
      body { margin: 0; background: #fff; color: var(--body); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; font-size: 15px; line-height: 1.6; -webkit-font-smoothing: antialiased; }
      body::before { content: ''; display: block; height: 3px; background: var(--accent); }
      nav, main { width: min(980px, calc(100% - 40px)); margin-inline: auto; }
      nav { padding-block: 26px 0; font-size: 14px; }
      main { padding-block: 30px 64px; }
      a { color: var(--accent); text-decoration: none; overflow-wrap: anywhere; }
      a:hover { color: var(--accent-hover); text-decoration: underline; text-underline-offset: .18em; }
      a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 2px; }
      h1 { margin: 0; font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif; font-size: clamp(2rem, 5vw, 2.5rem); line-height: 1.15; font-weight: 650; letter-spacing: -.012em; color: var(--ink); }
      .intro { margin: 12px 0 30px; max-width: 68ch; color: var(--muted); }
      .topics { columns: 2; column-gap: 44px; border-top: 1px solid var(--line); }
      section { min-width: 0; padding: 16px 0 17px; border-bottom: 1px solid var(--line); break-inside: avoid; }
      h2 { margin: 0; font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif; font-size: 1.18rem; line-height: 1.35; font-weight: 600; color: var(--ink); }
      h2 a { color: var(--ink); }
      h2 a:hover { color: var(--accent); }
      section p { margin: 5px 0 0; font-size: 14px; line-height: 1.5; color: var(--muted); }
      ul { margin: 8px 0 0; padding: 0; list-style: none; font-size: 14px; }
      li { padding: 3px 0; }
      li::before { content: '→'; margin-right: 8px; color: var(--muted); }
      @media (max-width: 820px) { .topics { columns: 1; } }
      @media (max-width: 580px) { nav, main { width: calc(100% - 32px); } li { padding: 6px 0; } }
    </style>
  </head>
  <body>
    <header><nav aria-label="Site"><a href="/">← Bai Liping</a></nav></header>
    <main>
      <h1>All pages</h1>
      <p class="intro">Browse research presentations, articles, interactive demonstrations, study notes, and personal collections. Related demos and notes are listed beneath each topic.</p>
      <div class="topics">
''' + content + '''
      </div>
    </main>
  </body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the checked-in discovery files are stale")
    args = parser.parse_args()
    pages, skipped = discover()
    outputs = {Path("sitemap.xml"): sitemap(pages), DIRECTORY: directory(pages)}
    stale = []
    for relative, content in outputs.items():
        target = ROOT / relative
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != content:
                stale.append(str(relative))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    print(f"Search discovery: {len(pages) + 1} public page URLs, {len(skipped)} excluded files")
    for name, reason in skipped:
        print(f"  Excluded {name}: {reason}")
    if stale:
        parser.exit(1, "Stale files: " + ", ".join(stale) + "\nRun python3 scripts/build-search-index.py\n")


if __name__ == "__main__":
    main()
