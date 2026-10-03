import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createManifest} from '../assets/authoring-model.mjs';
import {docPattern, readDocument, readManifest, renderDocument, safeJSON} from './authoring-build.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mime = {'.html':'text/html', '.js':'text/javascript', '.mjs':'text/javascript', '.css':'text/css', '.json':'application/json', '.svg':'image/svg+xml', '.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.webp':'image/webp', '.pdf':'application/pdf', '.woff2':'font/woff2', '.mp4':'video/mp4', '.txt':'text/plain'};
const digest = (...values) => crypto.createHash('sha256').update(values.join('\n')).digest('hex');
const read = file => fs.readFileSync(file, 'utf8');
const revision = directory => digest(read(path.join(directory, 'index.html')), fs.existsSync(path.join(directory, 'authoring.json')) ? read(path.join(directory, 'authoring.json')) : '');

function decksIn(root) {
  return fs.readdirSync(root, {withFileTypes:true}).filter(entry => entry.isDirectory() && !entry.name.startsWith('.')).flatMap(entry => {
    const file = path.join(root, entry.name, 'index.html');
    if (!fs.existsSync(file)) return [];
    try {
      const doc = readDocument(read(file));
      return doc.slides?.length ? [{slug:entry.name, title:doc.title, slides:doc.slides.length, edited:fs.existsSync(path.join(root, entry.name, 'authoring.json'))}] : [];
    } catch { return []; }
  }).sort((a,b) => a.title.localeCompare(b.title));
}

function saveFiles(root, directory, html, manifest) {
  const deck = path.basename(directory);
  const backup = path.join(root, '.authoring-backups', deck, new Date().toISOString().replaceAll(':','-') + '-' + crypto.randomBytes(3).toString('hex'));
  fs.mkdirSync(backup, {recursive:true});
  const targets = [['index.html', html], ['authoring.json', safeJSON(manifest) + '\n']];
  const old = new Map(targets.map(([name]) => [name, fs.existsSync(path.join(directory,name)) ? fs.readFileSync(path.join(directory,name)) : null]));
  for (const [name, content] of old) if (content) fs.writeFileSync(path.join(backup,name), content);
  try {
    for (const [name, content] of targets) fs.writeFileSync(path.join(directory,name + '.authoring-tmp'), content);
    for (const [name] of targets) fs.renameSync(path.join(directory,name + '.authoring-tmp'), path.join(directory,name));
  } catch (error) {
    for (const [name, content] of old) {
      if (content) fs.writeFileSync(path.join(directory,name),content);
      else fs.rmSync(path.join(directory,name), {force:true});
      fs.rmSync(path.join(directory,name + '.authoring-tmp'), {force:true});
    }
    throw error;
  }
}

async function bodyJSON(request) {
  let size = 0; const chunks = [];
  for await (const chunk of request) {
    size += chunk.length;
    if (size > 25 * 1024 * 1024) throw new Error('This draft is larger than 25 MB. Use linked media for large videos.');
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

export function createAuthoringServer({root = projectRoot} = {}) {
  root = fs.realpathSync(root);
  const sessions = new Map();
  const server = http.createServer(async (request,response) => {
    const send = (code, value, type = 'application/json') => {
      response.writeHead(code, {'Content-Type':type + (type.startsWith('text/') || type === 'application/json' ? '; charset=utf-8' : ''), 'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff', 'Referrer-Policy':'no-referrer'});
      response.end(typeof value === 'string' || Buffer.isBuffer(value) ? value : JSON.stringify(value));
    };
    try {
      const origin = `http://127.0.0.1:${server.address().port}`;
      if (request.headers.host !== new URL(origin).host) return send(403, {error:'Open the editor using its 127.0.0.1 address.'});
      const url = new URL(request.url, origin);
      if (request.method === 'POST') {
        if (request.headers.origin !== origin || request.headers['content-type'] !== 'application/json') return send(403,{error:'Save requests must come from this editor.'});
        if (url.pathname !== '/__authoring/save') return send(404,{error:'Unknown action.'});
        const input = await bodyJSON(request), session = sessions.get(input.session);
        if (!session || session.expires < Date.now()) return send(403,{error:'This editing session expired. Export your draft, reopen the editor, and import the draft.'});
        const directory = path.join(root,session.deck);
        if (revision(directory) !== session.revision || input.revision !== session.revision) return send(409,{error:'This presentation changed on disk or in another tab. Export your draft before reopening it; Codex can merge both versions.'});
        const original = read(path.join(directory,'index.html')), current = readDocument(original);
        const {manifest,doc} = createManifest(current,readManifest(directory),input.baseline,input.edited);
        const html = renderDocument(original,doc,current);
        saveFiles(root,directory,html,manifest);
        session.revision = revision(directory);
        return send(200,{revision:session.revision, changes:manifest.changes.length, message:'Saved on this Mac. Ready for Codex to review and publish.'});
      }
      if (request.method !== 'GET' && request.method !== 'HEAD') return send(405,{error:'Unsupported request.'});
      if (url.pathname === '/__authoring/decks') return send(200,{decks:decksIn(root), root});
      if (url.pathname === '/__authoring/health') return send(200,{app:'bailiping-authoring', root});
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === '/__authoring' || pathname === '/__authoring/') pathname = '/authoring/index.html';
      if (pathname.split('/').some(part => part.startsWith('.') && part !== '')) return send(403,{error:'Not available.'});
      let file = path.resolve(root, '.' + pathname);
      if (file !== root && !file.startsWith(root + path.sep)) return send(403,{error:'Not available.'});
      if (!fs.existsSync(file)) return send(404,{error:'File not found.'});
      if (fs.statSync(file).isDirectory()) file = path.join(file,'index.html');
      if (!fs.existsSync(file)) return send(404,{error:'File not found.'});
      const real = fs.realpathSync(file);
      if (!real.startsWith(root + path.sep)) return send(403,{error:'Not available.'});
      if (url.searchParams.get('edit') === '1' && path.basename(file) === 'index.html') {
        const deck = path.relative(root,path.dirname(file));
        if (!decksIn(root).some(item => item.slug === deck)) return send(400,{error:'This page is not a supported Bento deck.'});
        let html = read(file); const doc = readDocument(html);
        const session = crypto.randomBytes(24).toString('hex'), rev = revision(path.dirname(file));
        for (const [id, record] of sessions) if (record.expires < Date.now()) sessions.delete(id);
        sessions.set(session,{deck, revision:rev, expires:Date.now()+24*60*60*1000});
        doc.readonly = false; delete doc.collab;
        // Separate the editor's recovery snapshots from the public presentation.
        doc.docId += '-local-authoring';
        html = html.replace(docPattern,(_,a,b,c) => a + safeJSON(doc) + c);
        html = html.replace(/<script\b[^>]*\bsrc=["'][^"']*mathjax-dynamic\.js[^"']*["'][^>]*><\/script>/g,'');
        // The shared adapter prioritizes the presentation copy over thumbnails.
        html = html.replace(/src="\.\/inline-live\.js(?:\?[^"']*)?"/g,'src="../assets/bento-inline-live.js"');
        html = html.replace('</head>', '<link rel="stylesheet" href="/authoring/editor.css"></head>');
        const config = {session, revision:rev, deck, root, title:doc.title};
        html = html.replace('<body>', '<body data-bailiping-editor><script type="application/json" id="authoring-config">' + safeJSON(config) + '</script><script type="module" src="/authoring/editor.mjs"></script>');
        return send(200,html,'text/html');
      }
      return send(200,fs.readFileSync(file),mime[path.extname(file)] || 'application/octet-stream');
    } catch (error) { return send(400,{error:error.message}); }
  });
  return server;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.BAILIPING_EDITOR_PORT || 4317);
  const server = createAuthoringServer();
  server.on('error',error => {console.error(error.code === 'EADDRINUSE' ? 'The editor is already running, or port ' + port + ' is in use.' : error.message); process.exitCode=1;});
  server.listen(port,'127.0.0.1',() => console.log(`Bailiping editor: http://127.0.0.1:${server.address().port}/__authoring/\nSource: ${projectRoot}\nKeep this process running while editing. Save changes writes local files; publishing is a separate Codex step.`));
}
