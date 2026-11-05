/**
 * Joint pose–splat fitting from two fixed RGB observations.
 *
 * A deliberately small 2D orthographic teaching model: four anisotropic
 * Gaussians with known shape, opacity, color and layer order; unknown centers;
 * one anchored view and one unknown SE(2) camera transform. The observation
 * generator is used once, then discarded. Neither objective nor optimizer can
 * access its generating centers or pose. This is not a full 3D SLAM system.
 *
 * The renderer, local L1 + SSIM objective and their analytic derivatives are
 * shared with the fixed-camera lab. The pose Jacobian includes rotation of
 * both each projected center AND its anisotropic covariance.
 */
import {BACKGROUND, encodeSplats, evaluateTraining, renderSplats} from './model.mjs';

const FIXED = Object.freeze([
  {sx: .115, sy: .055, angle: -.45, opacity: .90, color: [.94, .39, .23]},
  {sx: .065, sy: .105, angle: .65, opacity: .86, color: [.17, .76, .69]},
  {sx: .120, sy: .055, angle: .30, opacity: .91, color: [.97, .76, .27]},
  {sx: .060, sy: .100, angle: -.20, opacity: .86, color: [.39, .50, .94]},
].map(s => Object.freeze({...s, color: Object.freeze(s.color)})));
const ANCHOR = Object.freeze([0, 0, 0]);
const MODES = new Set(['joint', 'pose', 'splats']);
const SCHEDULES = new Set(['simultaneous', 'alternating']);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

function worldSplats(centers) {
  return FIXED.map((s, i) => ({...s, color: [...s.color], x: centers[2*i], y: centers[2*i+1]}));
}

/** world → image, about image center: m = 0.5 + R(theta)(mu-0.5) + t. */
export function projectJointSplats(centers, pose) {
  const [tx, ty, angle] = pose, c = Math.cos(angle), s = Math.sin(angle);
  return worldSplats(centers).map(g => {
    const x = g.x - .5, y = g.y - .5;
    return {...g, x: .5 + c*x - s*y + tx, y: .5 + s*x + c*y + ty, angle: g.angle + angle};
  });
}

function observations(size) {
  // Synthetic acquisition only. These parameters are not retained in state.
  const centers = [.29, .31, .68, .30, .58, .64, .27, .70];
  const pose = [.060, -.040, .180];
  return [ANCHOR, pose].map(p => renderSplats(projectJointSplats(centers, p), {size}));
}

function validateOptions(mode, schedule) {
  if (!MODES.has(mode)) throw new RangeError('Mode must be joint, pose, or splats.');
  if (!SCHEDULES.has(schedule)) throw new RangeError('Schedule must be simultaneous or alternating.');
}

function historyPoint(r) {
  return {iteration:r.iteration, loss:r.loss, l1:r.l1, ssim:r.ssim, dssim:r.dssim,
    lambda:r.lambda, mode:r.mode, schedule:r.schedule};
}

export function createJoint({size=24, lambda=.2, mode='joint', schedule='simultaneous'}={}) {
  validateOptions(mode, schedule);
  const targets = observations(size);
  const centers = Float64Array.from([.35,.26,.61,.36,.64,.69,.22,.63]);
  const pose = Float64Array.from([-.025,.035,-.100]);
  const state = {
    size, centers, pose, targets, background:[...BACKGROUND],
    initialCenters:Array.from(centers), initialPose:Array.from(pose),
    lambda:clamp(lambda,0,1), mode, schedule, iteration:0, phase:0, history:[],
    accepted:true, backtracks:0,
  };
  const r = evaluateJoint(state);
  state.initialLoss = r.loss;
  state.history.push(historyPoint(r));
  return state;
}

export function evaluateJoint(state) {
  const centersGradient = new Float64Array(state.centers.length);
  const poseGradient = new Float64Array(3);
  const cameras = [ANCHOR, state.pose].map((pose, view) => {
    const projectedSplats = projectJointSplats(state.centers, pose);
    const r = evaluateTraining({size:state.size, params:encodeSplats(projectedSplats),
      target:state.targets[view], background:state.background, lambda:state.lambda,
      iteration:state.iteration, history:[]});
    const c = Math.cos(pose[2]), s = Math.sin(pose[2]);
    for (let i=0; i<FIXED.length; i++) {
      const gx=r.gradients[9*i]/2, gy=r.gradients[9*i+1]/2;
      centersGradient[2*i] += c*gx+s*gy;
      centersGradient[2*i+1] += -s*gx+c*gy;
      if (view===1) {
        const x=state.centers[2*i]-.5, y=state.centers[2*i+1]-.5;
        poseGradient[0] += gx;
        poseGradient[1] += gy;
        // d(R(theta)q)/dtheta plus d(projected angle)/dtheta = 1.
        poseGradient[2] += gx*(-s*x-c*y)+gy*(c*x-s*y)+r.gradients[9*i+4]/2;
      }
    }
    return {id:view===0?'anchor':'moving', label:view===0?'View 1 · anchored':'View 2 · estimated pose',
      anchored:view===0, pose:Array.from(pose), projectedSplats,
      image:r.image, target:state.targets[view], residual:r.residual,
      pixelGradient:r.pixelGradient, l1:r.l1, ssim:r.ssim, dssim:r.dssim, loss:r.loss};
  });
  const mean = key => (cameras[0][key]+cameras[1][key])/2;
  return {
    size:state.size, splats:worldSplats(state.centers), centers:Array.from(state.centers), pose:Array.from(state.pose),
    cameras, gradients:{centers:centersGradient, pose:poseGradient},
    loss:mean('loss'), l1:mean('l1'), ssim:mean('ssim'), dssim:mean('dssim'),
    lambda:state.lambda, mode:state.mode, schedule:state.schedule, iteration:state.iteration,
    history:state.history, initialLoss:state.initialLoss, phase:state.phase,
    accepted:state.accepted, backtracks:state.backtracks,
  };
}

/** Backtracking gradient descent. Every candidate is judged on RGB only. */
function descend(state, r, moveCenters, movePose) {
  const beforeCenters=Float64Array.from(state.centers), beforePose=Float64Array.from(state.pose);
  // Parameter scaling is fixed in advance, not derived from hidden truth.
  const gc=r.gradients.centers, gp=r.gradients.pose;
  const poseRates=[.5,.5,2];
  let slope=0;
  if(moveCenters) for(const g of gc) slope+=g*g;
  if(movePose) for(let i=0;i<3;i++) slope+=gp[i]*gp[i]*poseRates[i];
  if(slope<1e-24) return {result:r, accepted:false, backtracks:0};
  let step=.20;
  for(let attempt=0;attempt<18;attempt++,step*=.5) {
    if(moveCenters) for(let i=0;i<gc.length;i++) state.centers[i]=beforeCenters[i]-step*gc[i];
    if(movePose) for(let i=0;i<3;i++) state.pose[i]=beforePose[i]-step*poseRates[i]*gp[i];
    const candidate=evaluateJoint(state);
    if(Number.isFinite(candidate.loss)&&candidate.loss <= r.loss-1e-4*step*slope) {
      return {result:candidate, accepted:true, backtracks:attempt};
    }
  }
  state.centers.set(beforeCenters); state.pose.set(beforePose);
  return {result:r, accepted:false, backtracks:18};
}

/**
 * A simultaneous iteration computes both blocks at the same old state.
 * An alternating iteration updates pose, recomputes residuals, then centers.
 * Changing controls starts a fresh loss-history phase without resetting fit.
 */
export function stepJoint(state,{steps=1,mode=state.mode,schedule=state.schedule,lambda=state.lambda}={}) {
  validateOptions(mode,schedule);
  lambda=clamp(lambda,0,1);
  const changed=mode!==state.mode||schedule!==state.schedule||lambda!==state.lambda;
  state.mode=mode; state.schedule=schedule; state.lambda=lambda;
  let r=evaluateJoint(state);
  if(changed) {
    state.phase++;
    state.initialLoss=r.loss;
    state.history.length=0;
    state.history.push(historyPoint(r));
  }
  for(let step=0;step<Math.max(0,Math.floor(steps));step++) {
    let update;
    if(mode==='joint'&&schedule==='alternating') {
      const first=descend(state,r,false,true);
      const second=descend(state,first.result,true,false);
      update={result:second.result, accepted:first.accepted||second.accepted,
        backtracks:first.backtracks+second.backtracks};
    } else update=descend(state,r,mode!=='pose',mode!=='splats');
    state.iteration++;
    state.accepted=update.accepted; state.backtracks=update.backtracks;
    r=evaluateJoint(state);
    state.history.push(historyPoint(r));
  }
  return evaluateJoint(state);
}
