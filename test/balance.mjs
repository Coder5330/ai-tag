// Pure-physics balance check — no neural networks involved.
//
// Ten trained configurations all landed near 3s survival, which made it
// impossible to tell a physics problem from a learning problem. This removes
// learning entirely: a scripted evader (flee, curve off walls) against a
// scripted pursuer (lead the target), so whatever survival time comes out is
// a property of the arena alone.
//
// Runs in seconds, so the speed/size/clock trade-off can be swept properly
// before spending another six minutes of training on a guess.
//
//   node test/balance.mjs

import { RNG } from '../src/rng.js';
import {
  TagEnv,
  SPECS,
  TAG,
  ARENA_W,
  ARENA_H,
  RUNNER,
  TAGGER,
  EPISODE_FRAMES,
  setRoundFrames,
} from '../src/env.js';

function turnToward(a, want) {
  let d = want - a.th;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d > 0.05 ? 1 : d < -0.05 ? 2 : 0;
}

/** Flee the tagger, curving away from walls so as not to corner yourself. */
function evade(env) {
  const a = env.agents[RUNNER];
  const k = env.agents[TAGGER];
  let dx = a.x - k.x;
  let dy = a.y - k.y;
  const d = Math.hypot(dx, dy) || 1;
  dx /= d;
  dy /= d;
  const m = 10;
  let wx = 0;
  let wy = 0;
  if (a.x < m) wx += (m - a.x) / m;
  if (a.x > ARENA_W - m) wx -= (a.x - (ARENA_W - m)) / m;
  if (a.y < m) wy += (m - a.y) / m;
  if (a.y > ARENA_H - m) wy -= (a.y - (ARENA_H - m)) / m;
  return [1, turnToward(a, Math.atan2(dy + wy * 2, dx + wx * 2)), 0, 0];
}

/** Pursue with a lead: aim where the runner is going, not where it is. */
function pursue(env) {
  const a = env.agents[TAGGER];
  const r = env.agents[RUNNER];
  const dist = Math.hypot(r.x - a.x, r.y - a.y);
  const lead = dist / Math.max(1, a.spec.maxSpeed);
  return [1, turnToward(a, Math.atan2(r.y + r.vy * lead - a.y, r.x + r.vx * lead - a.x)), 0, 0];
}

function measure(episodes = 400) {
  const env = new TagEnv(new RNG(2024), 0);
  let tags = 0;
  let frames = 0;
  for (let e = 0; e < episodes; e++) {
    env.reset();
    let res;
    do {
      res = env.step([evade(env), pursue(env)]);
    } while (!res.done);
    if (res.tagged) tags++;
    frames += env.frame;
  }
  return { tagRate: tags / episodes, survival: frames / episodes / 60 };
}

const base = { rs: 12.5, ts: 13.6, rt: 5.8, tt: 3.8, tag: 3.2 };
function apply(c) {
  SPECS[RUNNER].maxSpeed = c.rs;
  SPECS[TAGGER].maxSpeed = c.ts;
  SPECS[RUNNER].turnRate = c.rt;
  SPECS[TAGGER].turnRate = c.tt;
  TAG.dist = c.tag;
}

const configs = [
  ['current', {}],
  ['both 25% slower', { rs: 9.4, ts: 10.2 }],
  ['both 40% slower', { rs: 7.5, ts: 8.2 }],
  ['tagger edge 3%', { ts: 12.9 }],
  ['tagger edge 0%', { ts: 12.5 }],
  ['runner faster', { rs: 13.0, ts: 12.5 }],
  ['tag radius 2.2', { tag: 2.2 }],
  ['wide-turning tagger', { tt: 2.4 }],
  ['slower + narrow tag', { rs: 9.4, ts: 10.2, tag: 2.2 }],
  ['slower + even speed', { rs: 10.2, ts: 10.2 }],
  ['slower + runner faster', { rs: 10.4, ts: 10.2, tag: 2.4 }],
];

const roundS = EPISODE_FRAMES / 60;
console.log(`\nscripted evader vs scripted pursuer · ${roundS}s rounds · arena ${ARENA_W}\n`);
console.log('  config                    survival   escapes');
console.log('  ------------------------  --------   -------');

for (const [name, over] of configs) {
  apply({ ...base, ...over });
  const r = measure();
  const short = measureShort(EPISODE_FRAMES / 60);
  console.log(
    `  ${name.padEnd(24)}  ${r.survival.toFixed(2).padStart(5)}s   ` +
      `${((1 - r.tagRate) * 100).toFixed(0).padStart(4)}%`,
  );
}
apply(base);

/** Share of rounds where the evader was still alive at `seconds`. */
function measureShort(seconds) {
  const env = new TagEnv(new RNG(2024), 0);
  const limit = seconds * 60;
  let alive = 0;
  const n = 400;
  for (let e = 0; e < n; e++) {
    env.reset();
    let res;
    do {
      res = env.step([evade(env), pursue(env)]);
    } while (!res.done && env.frame < limit);
    if (!res.tagged) alive++;
  }
  return (100 * alive) / n;
}
