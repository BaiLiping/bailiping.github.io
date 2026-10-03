import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {changesBetween,applyChanges,createManifest,updateLiveMap,validateDocument} from '../assets/authoring-model.mjs';
import {readDocument,withAuthoring,renderDocument} from '../scripts/authoring-build.mjs';
import {createAuthoringServer} from '../scripts/authoring-server.mjs';

const slide=id=>({id,notes:'Speaker notes',elements:[{id:'title',type:'text',x:72,y:70,w:900,h:58,fontSize:38,html:'Original '+id}]});
const base=()=>({format:'bento/slides',version:1,docId:'test',readonly:true,title:'Test',size:{width:1280,height:720},slides:[slide('intro'),slide('live'),slide('end')]});
const copy=v=>structuredClone(v);
const shell=doc=>`<!doctype html><head><script>(()=>{const routes={"intro":0,"live":1,"end":2};})();</script></head><body><script type="application/bento+json" id="bento-doc">${JSON.stringify(doc)}</script><script type="application/json" id="bento-inline-live-map">[]</script></body>`;

test('visual edits survive an independent upstream content change',()=>{
  const source=base(),edited=copy(source);edited.slides[0].elements[0].x=140;edited.slides[1].elements[0].html='User wording';
  const next=copy(source);next.slides[2].notes='Updated science';next.slides[0].elements.push({id:'new',type:'text',html:'New source object'});
  const result=applyChanges(next,changesBetween(source,edited));
  assert.equal(result.slides[0].elements[0].x,140);assert.equal(result.slides[1].elements[0].html,'User wording');assert.equal(result.slides[2].notes,'Updated science');assert.equal(result.slides[0].elements[1].id,'new');
});
test('same-property source changes fail without silently dropping either version',()=>{
  const source=base(),edited=copy(source),next=copy(source);edited.slides[0].elements[0].html='User';next.slides[0].elements[0].html='Agent';
  assert.throws(()=>applyChanges(next,changesBetween(source,edited)),/overlap source changes.*intro.*html/);
  assert.equal(next.slides[0].elements[0].html,'Agent');
});
test('reorder and new upstream slides merge; competing reorders conflict',()=>{
  const source=base(),edited=copy(source);edited.slides=[edited.slides[2],...edited.slides.slice(0,2)];
  const next=copy(source);next.slides.splice(1,0,slide('new'));
  assert.deepEqual(applyChanges(next,changesBetween(source,edited)).slides.map(s=>s.id),['end','intro','new','live']);
  next.slides=[source.slides[1],source.slides[0],source.slides[2]];
  assert.throws(()=>applyChanges(next,changesBetween(source,edited)),/overlap/);
});
test('successive saves compose edits and undoing a saved edit removes its override',()=>{
  const source=base(),one=copy(source);one.slides[0].elements[0].x=100;
  const first=createManifest(source,null,source,one);
  const two=copy(first.doc);two.slides[2].notes='My notes';
  const second=createManifest(first.doc,first.manifest,first.doc,two);
  assert.equal(applyChanges(source,second.manifest.changes).slides[0].elements[0].x,100);
  const undone=copy(second.doc);undone.slides[0].elements[0].x=72;
  const third=createManifest(second.doc,second.manifest,second.doc,undone);
  assert.equal(third.manifest.changes.length,1);assert.equal(third.manifest.changes[0].path.at(-1),'notes');
});
test('source defaults and editor session identity do not become spurious overrides',()=>{
  const source=base(),normalized=copy(source);normalized.readonly=false;normalized.docId+='-local-authoring';normalized.collab={secret:'not stored'};normalized.slides[0].elements[0].opacity=1;
  const edited=copy(normalized);edited.slides[0].elements[0].fontSize=44;
  const saved=createManifest(source,null,normalized,edited);
  assert.equal(saved.manifest.changes.length,1);assert.equal(saved.doc.readonly,true);assert.equal(saved.doc.docId,'test');assert.equal(saved.doc.collab,undefined);assert.equal(saved.doc.slides[0].elements[0].opacity,undefined);
});
test('deleted and added objects survive another save',()=>{
  const source=base(),edited=copy(source);edited.slides[0].elements=[{id:'replacement',type:'text',html:'Replacement'}];
  const one=createManifest(source,null,source,edited),two=copy(one.doc);two.slides[0].elements[0].html='Refined replacement';
  const final=createManifest(one.doc,one.manifest,one.doc,two);
  assert.deepEqual(applyChanges(source,final.manifest.changes).slides[0].elements,two.slides[0].elements);
});
test('rebuild replays patches, updates aliases, and leaves original source intact',()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'authoring-build-'));
  try{
    const source=base(),edited=copy(source);edited.slides=[edited.slides[2],...edited.slides.slice(0,2)];edited.slides[0].elements[0].html='</script> stays text';
    fs.writeFileSync(path.join(root,'authoring.json'),JSON.stringify(createManifest(source,null,source,edited).manifest));
    const first=withAuthoring(shell(source),pathToFileURL(path.join(root,'build.mjs')).href);
    assert.equal(readDocument(first).slides[0].elements[0].html,'</script> stays text');assert.match(first,/"intro":1,"live":2,"end":0/);
    assert.equal(first,withAuthoring(shell(source),pathToFileURL(path.join(root,'build.mjs')).href));
    assert.equal(source.slides[0].id,'intro');
  }finally{fs.rmSync(root,{recursive:true,force:true})}
});
test('live mappings follow new positions and enforce adjacent introductions',()=>{
  const source=base();source.slides[1].elements.push({id:'live-demo-mount',type:'shape',x:72,y:180,w:1100,h:475});
  const map=[{slide:'live',introSlide:'intro',slideIndex:1,bounds:{}}];
  const moved=copy(source);moved.slides=[moved.slides[2],...moved.slides.slice(0,2)];
  assert.equal(updateLiveMap(map,moved)[0].slideIndex,2);assert.equal(updateLiveMap(map,moved)[0].bounds.width,1100);
  moved.slides=[source.slides[0],source.slides[2],source.slides[1]];assert.throws(()=>updateLiveMap(map,moved),/Keep the introduction/);
});
test('rendered math and dangling links cannot be saved',()=>{
  const source=base(),edited=copy(source);edited.slides[0].elements[0].html='<mjx-container>bad</mjx-container>';assert.throws(()=>validateDocument(edited,source),/LaTeX/);
  const linked=base();linked.slides[0].elements[0].link='end';const removed=copy(linked);removed.slides.pop();assert.throws(()=>validateDocument(removed,linked),/still links/);
});
test('moving a lab fallback also updates its marker and live frame bounds',()=>{
  const source=base();source.slides[1].elements.push({id:'fallback',type:'image',x:72,y:180,w:1100,h:475},{id:'live-demo-mount',type:'shape',x:72,y:180,w:1100,h:475});
  const edited=copy(source);edited.slides[1].elements.find(e=>e.id==='fallback').x=90;
  const html=shell(source).replace('id="bento-inline-live-map">[]','id="bento-inline-live-map">'+JSON.stringify([{slide:'live',introSlide:'intro',slideIndex:1,bounds:{x:72,y:180,width:1100,height:475}}]));
  const result=renderDocument(html,edited,source);
  assert.equal(readDocument(result).slides[1].elements.find(e=>e.id==='live-demo-mount').x,90);
  assert.equal(JSON.parse(result.match(/id="bento-inline-live-map">([\s\S]*?)<\/script>/)[1])[0].bounds.x,90);
});
test('existing multi-lab sequences remain editable and can move as a group',()=>{
  const source=base();source.slides.splice(2,0,slide('second-live'));
  const map=[{slide:'second-live',introSlide:'intro',slideIndex:2}];
  const edited=copy(source);edited.slides[0].elements[0].html='Updated introduction';
  assert.equal(updateLiveMap(map,edited,source)[0].slideIndex,2);
  edited.slides=[edited.slides.at(-1),...edited.slides.slice(0,-1)];
  assert.equal(updateLiveMap(map,edited,source)[0].slideIndex,3);
  const broken=copy(source);broken.slides=[source.slides[0],source.slides[3],source.slides[1],source.slides[2]];
  assert.throws(()=>updateLiveMap(map,broken,source),/Keep the introduction/);
});
test('local save persists, preserves readonly publication, backs up, and rejects stale tabs',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'authoring-server-'));fs.mkdirSync(path.join(root,'test'));fs.writeFileSync(path.join(root,'test/index.html'),shell(base()));
  const server=createAuthoringServer({root});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const origin='http://127.0.0.1:'+server.address().port;
  try{
    const open=async()=>{const html=await(await fetch(origin+'/test/?edit=1')).text();const config=JSON.parse(html.match(/id="authoring-config">([\s\S]*?)<\/script>/)[1]);return{config,doc:readDocument(html)}};
    const a=await open(),b=await open();assert.equal(a.doc.readonly,false);
    const edited=copy(a.doc);edited.slides[0].elements[0].html='Saved from editor';
    const post=(session,doc,originHeader=origin)=>fetch(origin+'/__authoring/save',{method:'POST',headers:{'Content-Type':'application/json',Origin:originHeader},body:JSON.stringify({session:session.config.session,revision:session.config.revision,baseline:session.doc,edited:doc})});
    assert.equal((await post(a,edited,'https://example.com')).status,403);
    assert.equal((await post(a,edited)).status,200);
    const published=readDocument(fs.readFileSync(path.join(root,'test/index.html'),'utf8'));
    assert.equal(published.readonly,true);assert.equal(published.slides[0].elements[0].html,'Saved from editor');
    assert.ok(fs.existsSync(path.join(root,'test/authoring.json')));assert.equal(fs.readdirSync(path.join(root,'.authoring-backups/test')).length,1);
    assert.equal((await post(b,b.doc)).status,409);
    assert.equal((await fetch(origin+'/.authoring-backups/test/')).status,403);
    assert.equal((await fetch(origin+'/.git/config')).status,403);
    const reloaded=await open();assert.equal(reloaded.doc.slides[0].elements[0].html,'Saved from editor');
  }finally{await new Promise(resolve=>server.close(resolve));fs.rmSync(root,{recursive:true,force:true})}
});
