import {Raycaster, Vector2} from '/honda-civic-type-r/vendor/three-r170.js';

export function attachDrawing({parts, canvas, camera, focus, hold}) {
  const strip = document.getElementById('assembly-strip');
  const system = document.getElementById('assembly-system');
  const select = document.getElementById('assembly-part');
  const tooltip = document.getElementById('assembly-name');
  const byObject = new Map(parts.map(part => [part.object, part]));
  const detail = document.getElementById('assembly-detail');
  const component = document.getElementById('assembly-component');
  const componentLabel = document.getElementById('component-label');
  const componentNames = {
    Tire:'Tire', Alloy:'Alloy wheel and lugs', Caliper:'Brake caliper', Rotor:'Rotor and hub',
    Motor:'Motor housing', Reduction:'Reduction gearcase', Inverter:'Inverter',
    Cooler:'Heat exchanger', ShaftLeft:'Left halfshaft', ShaftRight:'Right halfshaft'
  };
  const byComponent = new Map();
  const arrangements = new Map();
  for(const part of parts) {
    if(!part.detail)continue;
    const children=Object.entries(part.detail).map(([suffix,delta])=>{
      const object=part.object.getObjectByName(part.object.name+suffix);
      if(!object)throw Error('Missing component '+part.object.name+suffix);
      const child={object,delta,home:object.position.clone(),suffix,label:componentNames[suffix]};
      byComponent.set(object,child);
      return child;
    });
    arrangements.set(part.object.name,children);
  }
  let separated=false;
  function restoreDetails(){
    for(const children of arrangements.values())for(const child of children){
      child.object.position.copy(child.home);
      child.object.visible=true;
    }
    separated=false;
    componentLabel.hidden=true;
    component.replaceChildren();
    detail.hidden=true;
    detail.textContent='Separate components';
    detail.setAttribute('aria-pressed','false');
  }
  const ray = new Raycaster();
  let active = false, selected = false, down = null, dragged = false;
  const systems = [...new Set(parts.map(part => part.system))];
  for (const name of systems) system.add(new Option(name, name));

  function fillParts() {
    select.replaceChildren(new Option('All parts', ''));
    for (const name of systems) {
      if (system.value && name !== system.value) continue;
      const options = document.createElement('optgroup');
      options.label = name;
      for (const part of parts.filter(p => p.system === name)) {
        options.append(new Option(part.label, part.object.name));
      }
      select.append(options);
    }
  }
  function showChoice() {
    hold();restoreDetails();
    selected = Boolean(system.value || select.value);
    for (const part of parts) {
      part.object.visible = (!system.value || part.system === system.value) &&
        (!select.value || part.object.name === select.value);
    }
    const name = select.value;
    const side = name.endsWith('Right') ? -1 : 1;
    let direction;
    if (/^(Cover|Wheel|Brake)/.test(name)) direction = [2 * side, .32, .65];
    else if (name.startsWith('Headlamp')) direction = [.6 * side, .67, 1.3];
    else if (name.startsWith('TailLamp')) direction = [.8 * side, .48, -1.3];
    else if (/^(Lid|Pack|Module|Ancillary)/.test(name)) direction = [.6, 1.55, .9];
    else if (name === 'Cabin') direction = [.18, .32, -1];
    else if (name === 'RearFascia' || name === 'Liftgate') direction = [.8, .56, -1.35];
    if(arrangements.has(name))detail.hidden=false;
    focus(selected, direction);
    tooltip.hidden = true;
  }
  function focusComponents() {
    const right=select.value.endsWith('Right') ? -1 : 1;
    focus(true,select.value.includes('Drive') ? [.84,.78,1.18] : [right*1.8,.41,.93]);
  }
  component.onchange=()=>{
    hold();
    for(const child of arrangements.get(select.value))child.object.visible=!component.value || child.suffix===component.value;
    tooltip.hidden=true;
    focusComponents();
  };
  detail.onclick=()=>{
    hold();
    separated=!separated;
    const children=arrangements.get(select.value);
    component.replaceChildren(new Option('All components',''));
    for(const child of children){
      child.object.position.copy(child.home);
      child.object.visible=true;
      if(separated) {
        child.object.position.add({x:child.delta[0],y:child.delta[1],z:child.delta[2]});
        component.add(new Option(child.label,child.suffix));
      }
    }
    componentLabel.hidden=!separated;
    detail.textContent=separated ? 'Reassemble components' : 'Separate components';
    detail.setAttribute('aria-pressed',String(separated));
    focusComponents();
  };
  function clear() {
    restoreDetails();
    for (const part of parts) part.object.visible = true;
    selected = false; system.value = ''; fillParts(); tooltip.hidden = true;
  }
  function pick(event) {
    const rect = canvas.getBoundingClientRect();
    const pointer = new Vector2(2*(event.clientX-rect.left)/rect.width-1, 1-2*(event.clientY-rect.top)/rect.height);
    ray.setFromCamera(pointer, camera);
    const roots = parts.filter(part => part.object.visible).map(part => part.object);
    const hit = ray.intersectObjects(roots, true).find(({object}) => {
      if(!object.isMesh)return false;
      for(let node=object;node;node=node.parent)if(!node.visible)return false;
      return true;
    });
    let child;
    for (let node = hit?.object; node; node = node.parent) {
      if (byComponent.has(node)) child=byComponent.get(node);
      if (byObject.has(node)) return {part:byObject.get(node),child};
    }
  }
  system.onchange = () => { fillParts(); showChoice(); };
  select.onchange = showChoice;
  // Selecting a name or touching the drawing keeps the exploded pose on screen.
  strip.addEventListener('focusin', hold);
  strip.addEventListener('pointerdown', hold);
  canvas.addEventListener('pointerdown', event => { down=[event.clientX,event.clientY]; dragged = !event.isPrimary; });
  canvas.addEventListener('pointermove', event => {
    if (!active) return;
    if (down && Math.hypot(event.clientX-down[0], event.clientY-down[1]) > 5) dragged = true;
    if (down || event.pointerType !== 'mouse' || (selected && !separated)) { tooltip.hidden=true; return; }
    const hit = pick(event);
    tooltip.hidden = !hit;
    if (hit) {
      const rect = canvas.getBoundingClientRect();
      tooltip.textContent = separated && hit.child ? hit.child.label : hit.part.label;
      tooltip.style.left = Math.max(4, Math.min(event.clientX-rect.left+10, rect.width-tooltip.offsetWidth-5))+'px';
      tooltip.style.top = Math.max(3, Math.min(event.clientY-rect.top+12, rect.height-32))+'px';
    }
  });
  canvas.addEventListener('pointerup', event => {
    if (active && down && !dragged && (!select.value || separated)) {
      const hit = pick(event);
      if (separated && hit?.child) {
        component.value=hit.child.suffix;component.onchange();
      } else if (hit && !select.value) {
        system.value=hit.part.system;fillParts();select.value=hit.part.object.name;showChoice();
      }
    }
    down=null;
  });
  canvas.addEventListener('pointercancel', () => { down=null; tooltip.hidden=true; });
  canvas.addEventListener('pointerleave', () => { tooltip.hidden=true; });
  fillParts();
  return {clear, get selected(){return selected;}, get separated(){return separated;}, setActive(on){
    if (active === on) return;
    active=on; strip.hidden=!on; tooltip.hidden=true;
  }};
}
