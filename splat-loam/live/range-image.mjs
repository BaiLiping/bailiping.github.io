/** Display mapping only; colours are not attributes of the reconstructed scene. */
const BACKGROUND = [247,246,237];
const STOPS = [[24,78,90],[32,137,148],[210,194,128],[198,107,61]];
const clamp01 = value => Math.max(0,Math.min(1,value));

/** A fixed scale: range is 0–12 metres, opacity is 0–1. No opacity tint on range. */
export function rangeColor(value,mode='expectedDepth',colorMode='gray') {
  if(mode==='opacity') return BACKGROUND.map((a,k)=>Math.round(a+([22,125,141][k]-a)*clamp01(value)));
  const t=clamp01(value/12);
  if(colorMode==='gray') return Array(3).fill(Math.round(40+185*t));
  const x=t*3,i=Math.min(2,Math.floor(x)),f=x-i;
  return STOPS[i].map((a,k)=>Math.round(a+(STOPS[i+1][k]-a)*f));
}

/** Undefined expected ranges remain undefined in the model and neutral here. */
export function rangeImageRgba(image,mode,colorMode='gray') {
  const values=image[mode],pixels=new Uint8ClampedArray(image.opacity.length*4);
  for(let i=0;i<image.opacity.length;i++) {
    const o=image.opacity[i],value=values[i];
    const missing=!Number.isFinite(o)||!Number.isFinite(value)||o<=0||(mode!=='opacity'&&o<=1e-6);
    pixels.set([...(missing?BACKGROUND:rangeColor(value,mode,colorMode)),255],i*4);
  }
  return pixels;
}

