import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const defaults = ['et-handover/index.html'];
const requested = process.argv.slice(2);
const files = (requested.length ? requested : defaults).map(file => path.resolve(root, file));
let failures = 0;

function fail(file, message) {
  failures += 1;
  process.stderr.write(path.relative(root, file) + ': ' + message + '\n');
}

function parseBlock(html, pattern, file, label) {
  const match = html.match(pattern);
  if (!match) {
    fail(file, 'missing ' + label);
    return null;
  }
  try {
    return { raw: match[1], value: JSON.parse(match[1]) };
  } catch (error) {
    fail(file, 'invalid ' + label + ': ' + error.message);
    return null;
  }
}

function duplicates(values) {
  const seen = new Set();
  return values.filter(value => {
    if (seen.has(value)) return true;
    seen.add(value);
    return false;
  });
}

for (const file of files) {
  if (!fs.existsSync(file)) {
    fail(file, 'file does not exist');
    continue;
  }

  const html = fs.readFileSync(file, 'utf8');
  if (!html.includes('content="bento-slides"')) fail(file, 'missing Bento generator metadata');

  const parsed = parseBlock(
    html,
    /<script type="application\/bento\+json" id="bento-doc">\s*([\s\S]*?)\s*<\/script>/,
    file,
    '#bento-doc JSON'
  );
  if (!parsed) continue;
  const doc = parsed.value;

  if (parsed.raw.includes('<')) fail(file, '#bento-doc contains an unescaped "<"');
  if (doc.format !== 'bento/slides' || doc.version !== 1) fail(file, 'unexpected Bento format/version');
  if (doc.collab) fail(file, 'collaboration data is present');
  if (!doc.theme?.fontFamily || !doc.size?.width || !doc.size?.height) fail(file, 'missing required size/theme fields');
  if (!Array.isArray(doc.slides) || !doc.slides.length) fail(file, 'has no slides');

  const slideIds = (doc.slides || []).map(slide => slide.id);
  for (const id of duplicates(slideIds)) fail(file, 'duplicate slide id ' + id);

  for (const slide of doc.slides || []) {
    if (!slide.notes?.trim()) fail(file, slide.id + ' has no speaker notes');
    if (!Array.isArray(slide.elements)) {
      fail(file, slide.id + ' has no element list');
      continue;
    }
    const ids = slide.elements.map(element => element.id);
    for (const id of duplicates(ids)) fail(file, slide.id + ' has duplicate element id ' + id);
    const morphIds = slide.elements.map(element => element.morphId).filter(Boolean);
    for (const id of duplicates(morphIds)) fail(file, slide.id + ' has duplicate morph key ' + id);

    for (const element of slide.elements) {
      for (const key of ['id', 'type', 'x', 'y', 'w', 'h', 'rotation', 'opacity']) {
        if (element[key] === undefined) fail(file, slide.id + '/' + (element.id || '?') + ' missing ' + key);
      }
      if (element.x < 0 || element.y < 0 || element.x + element.w > doc.size.width || element.y + element.h > doc.size.height) {
        fail(file, slide.id + '/' + element.id + ' is outside the canvas');
      }
    }
  }

  const demoMatch = html.match(
    /<script type="application\/json" id="companion-demo-map">\s*([\s\S]*?)\s*<\/script>/
  );
  let demos = [];
  if (demoMatch) {
    try {
      demos = JSON.parse(demoMatch[1]);
    } catch (error) {
      fail(file, 'invalid companion demo map: ' + error.message);
    }
  }
  if (demos.length && !html.includes('../slides-assets/bento-demo-bridge.js') && !html.includes('./inline-live.js')) {
    fail(file, 'demo map exists without the bridge script');
  }

  for (const demo of demos) {
    if (!slideIds.includes(demo.slide)) fail(file, 'demo references missing slide ' + demo.slide);
    let targetUrl;
    try {
      const base = new URL('http://repo/' + path.relative(root, file).replaceAll(path.sep, '/'));
      targetUrl = new URL(demo.src, base);
    } catch (error) {
      fail(file, demo.slide + ' has invalid demo URL: ' + error.message);
      continue;
    }
    const selector = targetUrl.searchParams.get('slide-embed');
    if (!selector?.startsWith('#')) {
      fail(file, demo.slide + ' has no hash slide-embed selector');
      continue;
    }
    const sourceFile = path.join(root, targetUrl.pathname.replace(/^\/+/, ''), 'index.html');
    if (!fs.existsSync(sourceFile)) {
      fail(file, demo.slide + ' source page is missing: ' + path.relative(root, sourceFile));
      continue;
    }
    const sourceHtml = fs.readFileSync(sourceFile, 'utf8');
    const id = selector.slice(1).replace(/[.*+?^$(){}|[\]\\]/g, '\\$&');
    if (!new RegExp("id=[\"']" + id + "[\"']").test(sourceHtml)) {
      fail(file, demo.slide + ' target ' + selector + ' is missing from ' + path.relative(root, sourceFile));
    }
  }

  process.stdout.write(
    path.relative(root, file) + ': OK — ' + doc.slides.length + ' slides, ' + demos.length + ' live demos\n'
  );
}

if (failures) {
  process.stderr.write(String(failures) + ' validation failure(s)\n');
  process.exit(1);
}
