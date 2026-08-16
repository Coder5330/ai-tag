// How long *could* Albert last?
//
// Every physics knob we swept produced the same ~3s survival, which suggests
// the limit is not in the physics. This measures the ceiling directly: train
// the pair as usual, then replace Albert with a hand-written evader that flees
// perfectly (straight away from Kai, bending off the walls) and see how long
// that lasts against the same trained Kai.
//
//   scripted >> learned  -> Albert's learning is the problem
//   scripted ~= learned  -> the arena is unwinnable and the fix is elsewhere
//
//   node test/oracle.mjs [updates] [room]

import { RNG } from '../src/rng.js';
import {
  TagEnv,
  ARENA_W,
  ARENA_H,
  OBS_DIM,
  BRANCHES,
  RUNNER,
  TAGGER,
} from '../src/env.js';
import { SelfPlayTrainer } from '../src/trainer.js';
import { sampleRow } from '../src/nn.js';

const UPDATES = Number(process.argv[2] || 240);
const ROOM = Number(process.argv[3] || 0);

const trainer = new SelfPlayTrainer({ roomIndex: ROOM });
for (let u = 0; u < UPDATES; u++) trainer.runUpdate();

const obs = new Float32Array(OBS_DIM);
const rng = new RNG(31337);

/** Steer `a` toward world heading `want`. */
function turnToward(a, want) {
  let diff = want - a.th;
  while (diff > Math.PI) diff -= 2 * Math.PI;
  while (diff < -Math.PI) diff += 2 * Math.PI;
  return diff > 0.05 ? 1 : diff < -0.05 ? 2 : 0;
}

/** Hand-written ideal evader: run directly away, curving off the walls. */
function fleeAction(env) {
  const a = env.agents[RUNNER];
  const k = env.agents[TAGGER];
  let dx = a.x - k.x;
  let dy = a.y - k.y;
  const d = Math.hypot(dx, dy) || 1;
  dx /= d;
  dy /= d;

  // Repulsion from each wall, strongest when close, so the evader arcs along
  // the perimeter instead of trapping itself in a corner.
  const m = 10;
  let wx = 0;
  let wy = 0;
  if (a.x < m) wx += (m - a.x) / m;
  if (a.x > ARENA_W - m) wx -= (a.x - (ARENA_W - m)) / m;
  if (a.y < m) wy += (m - a.y) / m;
  if (a.y > ARENA_H - m) wy -= (a.y - (ARENA_H - m)) / m;

  const want = Math.atan2(dy + wy * 2.0, dx + wx * 2.0);
  return [1, turnToward(a, want), 0, 0];
}

/** Sample the tagger's trained policy. */
function taggerAction(env) {
  env.observe(TAGGER, obs, 0);
  const b = trainer.brains[TAGGER];
  b.forward(obs, 1);
  const out = [];
  for (let k = 0; k < BRANCHES.length; k++) {
    out.push(sampleRow(b.probs[k], 0, BRANCHES[k], rng));
  }
  return out;
}

function runnerAction(env) {
  env.observe(RUNNER, obs, 0);
  const b = trainer.brains[RUNNER];
  b.forward(obs, 1);
  const out = [];
  for (let k = 0; k < BRANCHES.length; k++) {
    out.push(sampleRow(b.probs[k], 0, BRANCHES[k], rng));
  }
  return out;
}

function evaluate(runnerPolicy, episodes = 300) {
  const env = new TagEnv(new RNG(555), ROOM);
  let tags = 0;
  let frames = 0;
  for (let e = 0; e < episodes; e++) {
    env.reset();
    let res;
    do {
      res = env.step([runnerPolicy(env), taggerAction(env)]);
    } while (!res.done);
    if (res.tagged) tags++;
    frames += env.frame;
  }
  return { tagRate: tags / episodes, survival: frames / episodes / 60 };
}

const learned = evaluate(runnerAction);
const scripted = evaluate(fleeAction);

console.log(`\n=== ceiling check · room ${ROOM} · ${UPDATES} updates ===`);
console.log(
  `  learned Albert    survives ${learned.survival.toFixed(2)}s   tagged ${(
    learned.tagRate * 100
  ).toFixed(1)}%`,
);
console.log(
  `  scripted flee     survives ${scripted.survival.toFixed(2)}s   tagged ${(
    scripted.tagRate * 100
  ).toFixed(1)}%`,
);
const gain = scripted.survival - learned.survival;
console.log(
  `\n  -> ${
    gain > 1.5
      ? 'the arena IS survivable; Albert is failing to learn it'
      : 'even perfect evasion loses — the arena itself is the problem'
  } (gap ${gain.toFixed(2)}s)`,
);
