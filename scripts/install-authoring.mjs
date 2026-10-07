// Add feedback context and the shared deck theme to checked-in decks without regenerating their content.
import fs from 'node:fs';
import {readDocument,withSiteAssets} from './authoring-build.mjs';
const root=new URL('../',import.meta.url);
for(const entry of fs.readdirSync(root,{withFileTypes:true})){
  if(!entry.isDirectory()||entry.name.startsWith('.'))continue;
  const file=new URL(entry.name+'/index.html',root);if(!fs.existsSync(file))continue;
  const html=fs.readFileSync(file,'utf8');let doc;try{doc=readDocument(html)}catch{continue}
  if(!doc.slides?.length)continue;
  // Existing edited output already contains its patches; only install the tag.
  const next=withSiteAssets(html);if(next!==html)fs.writeFileSync(file,next);
  console.log(entry.name);
}
