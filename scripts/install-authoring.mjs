// Add feedback context to checked-in decks without regenerating their content.
import fs from 'node:fs';
import {readDocument} from './authoring-build.mjs';
const root=new URL('../',import.meta.url);
for(const entry of fs.readdirSync(root,{withFileTypes:true})){
  if(!entry.isDirectory()||entry.name.startsWith('.'))continue;
  const file=new URL(entry.name+'/index.html',root);if(!fs.existsSync(file))continue;
  const html=fs.readFileSync(file,'utf8');let doc;try{doc=readDocument(html)}catch{continue}
  if(!doc.slides?.length)continue;
  // Existing edited output already contains its patches; only install the tag.
  if(!html.includes('src="../assets/slide-annotations.js"'))fs.writeFileSync(file,html.replace('</body>','<script src="../assets/slide-annotations.js"></script>\n</body>'));
  console.log(entry.name);
}
