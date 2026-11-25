import test from 'node:test';
import assert from 'node:assert/strict';
import {makeRenderScene,rangeImage} from '../live/model.mjs';
import {rangeImageRgba} from '../live/range-image.mjs';

const background=[247,246,237,255];
const pixel=(rgba,index)=>Array.from(rgba.slice(index*4,index*4+4));

test('yaw-shifted empty rays render as background without losing valid neighbouring pixels',()=>{
  for(const yaw of [6,35]) {
    const image=rangeImage(makeRenderScene(),{yaw});
    const missing=image.expectedDepth.findIndex(value=>Number.isNaN(value));
    const valid=image.expectedDepth.findIndex(value=>Number.isFinite(value));
    assert.ok(missing>=0,'this real control setting must include undefined expected range');
    assert.ok(valid>=0,'the image must also contain valid returns');
    if(yaw===6)assert.ok(image.opacity[missing]>0,'near-empty positive-opacity rays must also be masked');
    const rgba=rangeImageRgba(image,'expectedDepth');
    assert.deepEqual(pixel(rgba,missing),background);
    assert.notDeepEqual(pixel(rgba,valid),background);
    assert.ok(Number.isNaN(image.expectedDepth[missing]),'display mapping must not turn missing range into zero');
  }
});

test('range defaults to grayscale with an explicitly selectable fixed false-colour scale',()=>{
  const image={depth:Float32Array.from([0,4,8,12,24]),opacity:Float32Array.from([1,1,1,1,1])};
  assert.deepEqual(Array.from(rangeImageRgba(image,'depth','color')),[24,78,90,255,32,137,148,255,210,194,128,255,198,107,61,255,198,107,61,255]);
  assert.deepEqual(pixel(rangeImageRgba(image,'depth'),0),[40,40,40,255]);
  assert.deepEqual(pixel(rangeImageRgba(image,'depth'),3),[225,225,225,255]);
  assert.deepEqual(pixel(rangeImageRgba(image,'opacity'),0),[22,125,141,255]);
  assert.deepEqual(pixel(rangeImageRgba({depth:[0],expectedDepth:[NaN],opacity:[0]},'expectedDepth'),0),background);
  assert.deepEqual(pixel(rangeImageRgba({depth:[0],opacity:[0]},'opacity'),0),background);
});

test('equal range values have equal shades regardless of accumulated opacity',()=>{
  const image={expectedDepth:[6,6,6],opacity:[1,0.5,0.01]};
  for(const palette of ['gray','color']) {
    const rgba=rangeImageRgba(image,'expectedDepth',palette);
    assert.deepEqual(pixel(rgba,0),pixel(rgba,1));
    assert.deepEqual(pixel(rgba,1),pixel(rgba,2));
  }
});

test('all image modes handle supported control extremes, including undefined expected ranges',()=>{
  let missingCount=0;
  for(const yaw of [-35,35])for(const tilt of [-60,60])for(const scale of [.45,2.1])for(const opacity of [0,1]) {
    const image=rangeImage(makeRenderScene({tilt,scale,opacity}),{yaw});
    for(const mode of ['expectedDepth','depth','opacity']) {
      const rgba=rangeImageRgba(image,mode);
      assert.equal(rgba.length,image.width*image.height*4);
      for(let i=0;i<image.opacity.length;i++) {
        assert.equal(rgba[i*4+3],255);
        if(mode==='expectedDepth'&&!Number.isFinite(image.expectedDepth[i])) {
          missingCount++;
          assert.deepEqual(pixel(rgba,i),background,`yaw=${yaw}, tilt=${tilt}, scale=${scale}, opacity=${opacity}`);
        }
      }
    }
  }
  assert.ok(missingCount>0,'the regression must exercise missing expected-depth samples');
});

