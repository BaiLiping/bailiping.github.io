import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {withAuthoring, docPattern, safeJSON} from './authoring-build.mjs';

export async function buildModuleDeck(builderURL) {
  const directory=path.dirname(fileURLToPath(builderURL));
  const {deck,inlineLiveMap}=await import(new URL('./bento-deck.mjs',builderURL));
  const file=path.join(directory,'index.html');
  let html=fs.readFileSync(file,'utf8');
  html=html.replace(docPattern,(_,a,b,c)=>a+safeJSON(deck)+c)
    .replace(/(<script[^>]*id="bento-inline-live-map"[^>]*>)[\s\S]*?(<\/script>)/,(_,a,b)=>a+safeJSON(inlineLiveMap)+b)
    .replace(/<script type="module" src="boot.mjs"><\/script>/,'');
  if(!html.includes('id="bento-rt"')){
    const template=fs.readFileSync(new URL('../kalman-filter-derivations/index.html',builderURL),'utf8');
    const blocks=template.match(/<script id="bento-rt(?:-css)?"[^>]*>[\s\S]*?<\/script>/g);
    if(blocks?.length!==2)throw Error('Shared Bento engine missing');
    html=html.replace('</body>',blocks.join('\n')+'\n<script type="module" src="../assets/bento-bootstrap.mjs"></script>\n<script src="../assets/bento-inline-live.js"></script>\n<script type="module" src="../assets/deck-extensions.js"></script>\n</body>');
  }
  fs.writeFileSync(file,withAuthoring(html,builderURL));
  console.log(path.basename(directory)+': '+deck.slides.length+' slides ready for presentation and visual editing.');
}
