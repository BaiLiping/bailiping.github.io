import fs from 'node:fs';
export const sources={notion:'https://www.notion.so/2270d664735d8183bd8bc130601992e2',repo:'https://github.com/BaiLiping/RFS_Filters',evaluation:'https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/tracking/README.md'};
export const figures=JSON.parse(fs.readFileSync(new URL('./assets/figures.json',import.meta.url),'utf8'));
const log=fs.readFileSync(new URL('./assets/validation-results.txt',import.meta.url),'utf8');
const classNames={bicycle:'Bicycle',bus:'Bus',car:'Car',motorcy:'Motorcycle',pedestr:'Pedestrian',trailer:'Trailer',truck:'Truck'};
export const results=log.split('\n').filter(l=>/^(bicycle|bus|car|motorcy|pedestr|trailer|truck)\s/.test(l)).map(l=>{const [label,...v]=l.trim().split(/\s+/);return {name:classNames[label],amota:Number(v[0]),amotp:Number(v[1]),recall:Number(v[2]),ids:Number(v[13]),fragmentations:Number(v[14])}});
export const metrics=Object.fromEntries(log.split('Aggregated results:')[1].trim().split('\n').filter(l=>/^\w+\s+[\d.]+$/.test(l)).map(l=>{const [k,v]=l.trim().split(/\s+/);return [k,Number(v)]}));
export const groups=[
 {id:'occlusion',title:'Occlusion and missed detections',indices:[1,2]},
 {id:'crossing',title:'Crossings and overlapping tracks',indices:[3,4,5,6,31]},
 {id:'classification',title:'Classification changes',indices:[7,8,19,20,21,22,23,25,28,32,35,40]},
 {id:'birth',title:'Birth-rate tuning and track restarts',indices:[9,11,12,13,14,15,16,17,18,26,34,38,39]},
 {id:'turning',title:'Turning motion',indices:[29,30,33,41]},
 {id:'comparison',title:'TO-PMB and MHT comparisons',indices:[43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,64,65]},
 {id:'coalescence',title:'Coalescence and long associations',indices:[66,67,68,69,70,71,72,73]},
 {id:'additional',title:'Additional source figures and sequence',indices:[10,24,27,36,37,42,62,63]}
];
export function figure(index){return figures.find(f=>f.index===index)}
export const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
export const figureUrl=index=>'./assets/'+figure(index).file;
export const noteUrl=index=>sources.notion+'#'+figure(index).blockId.replaceAll('-','');
