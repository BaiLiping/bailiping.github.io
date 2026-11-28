import * as T from '/honda-civic-type-r/vendor/three-r170.js';

// Studio Small 08 by Poly Haven, CC0. The upper half of the lossless image
// contains the RGB mantissa, the lower half an 8-bit shared exponent.
export async function studioEnvironment(renderer, url) {
  const packed = await new T.TextureLoader().loadAsync(url);
  const width = packed.image.width, height = packed.image.height / 2;
  const halfFloat = 1016; // THREE.HalfFloatType in r170
  const target = new T.WebGLRenderTarget(width,height,{type:halfFloat,depthBuffer:false});
  const material = new T.ShaderMaterial({
    uniforms:{packedMap:{value:packed}},depthTest:false,depthWrite:false,
    vertexShader:'varying vec2 st; void main(){st=uv; gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader:`varying vec2 st; uniform sampler2D packedMap;
      void main(){
        vec3 rgb=texture2D(packedMap,vec2(st.x,.5+.5*st.y)).rgb;
        float exponent=floor(texture2D(packedMap,vec2(st.x,.5*st.y)).r*255.+.5)-128.;
        gl_FragColor=vec4(rgb*exp2(exponent),1.);
      }`
  });
  const plane=new T.PlaneGeometry(2,2);
  const pass=new T.Scene();pass.add(new T.Mesh(plane,material));
  const previous=renderer.getRenderTarget();renderer.setRenderTarget(target);
  renderer.render(pass,new T.OrthographicCamera(-1,1,1,-1,.1,2));renderer.setRenderTarget(previous);
  target.texture.mapping=T.EquirectangularReflectionMapping;
  const pmrem=new T.PMREMGenerator(renderer), env=pmrem.fromEquirectangular(target.texture);
  target.dispose();packed.dispose();material.dispose();plane.dispose();pmrem.dispose();
  return env.texture;
}
