// Physics contract checks for the 3D arena. Run: node test/physics.mjs
//
// These pin the rules the agents are supposed to be able to exploit. If a
// constant drifts (jump height vs body height, say) the game silently loses a
// strategy and only the training curves would ever hint at it.

import assert from 'node:assert/strict';
import { RNG } from '../src/rng.js';
import {
  TagEnv,
  RUNNER,
  TAGGER,
  AGENT_H,
  BOX,
  WALL_TOP,
  JUMP_V,
  GRAVITY,
  FRAME_SKIP,
  SPECS,
} from '../src/env.js';

let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log(`  ok  ${name}`);
}

const NOOP = [0, 0, 0, 0];
const FWD = [1, 0, 0, 0];
const JUMP = [0, 0, 1, 0];
const FWD_JUMP = [1, 0, 1, 0];
const GRAB = [0, 0, 0, 1];
const FWD_GRAB = [1, 0, 0, 1];

function env(room = 1) {
  const e = new TagEnv(new RNG(1), room);
  e.reset();
  return e;
}

/** Place the two agents deterministically, overriding the random reset. */
function place(e, rx, ry, rth, tx, ty, tth) {
  const [r, t] = e.agents;
  r.x = rx;
  r.y = ry;
  r.th = rth;
  r.z = 0;
  r.vx = r.vy = r.vz = 0;
  r.grounded = true;
  t.x = tx;
  t.y = ty;
  t.th = tth;
  t.z = 0;
  t.vx = t.vy = t.vz = 0;
  t.grounded = true;
  e.tagged = false;
}

console.log('physics');

check('a jump clears a rival\'s head', () => {
  const apex = (JUMP_V * JUMP_V) / (2 * GRAVITY);
  assert.ok(
    apex > AGENT_H,
    `jump apex ${apex.toFixed(2)} must exceed body height ${AGENT_H} or jumping over is impossible`,
  );
});

check('a jump does not reach a fixed block from the floor', () => {
  const apex = (JUMP_V * JUMP_V) / (2 * GRAVITY);
  assert.ok(apex < WALL_TOP, 'blocks should not be climbable without a crate');
  assert.ok(BOX + apex > WALL_TOP, 'a crate should make a block climbable');
});

check('jumping actually lifts the body past head height', () => {
  const e = env(0);
  place(e, 22, 22, 0, 40, 40, 0);
  let peak = 0;
  e.step([JUMP, NOOP]);
  for (let i = 0; i < 20; i++) {
    e.step([NOOP, NOOP]);
    peak = Math.max(peak, e.agents[RUNNER].z);
  }
  assert.ok(peak > AGENT_H, `peak ${peak.toFixed(2)} should exceed ${AGENT_H}`);
  assert.equal(e.agents[RUNNER].z, 0, 'and must come back down');
  assert.equal(e.agents[RUNNER].grounded, true);
});

check('a tag needs vertical overlap, not just horizontal', () => {
  const e = env(0);
  place(e, 22, 22, 0, 22, 22, 0); // exactly on top of each other in xy
  const [r, t] = e.agents;

  r.z = 0;
  assert.equal(e._touching(r, t), true, 'same height -> tagged');

  r.z = AGENT_H + 0.2; // fully above the tagger's head
  assert.equal(e._touching(r, t), false, 'cleanly above -> not tagged');

  r.z = AGENT_H - 0.2; // still overlapping by a sliver
  assert.equal(e._touching(r, t), true, 'partial overlap -> still tagged');
});

check('a runner can jump clean over an incoming tagger', () => {
  const e = env(0);
  // Tagger charges along +x at the stationary runner; the runner jumps early
  // enough that it is airborne while they cross.
  // The window is genuinely tight: the body is above head height for ~0.51s
  // out of a 1.03s hop, so the jump has to be timed, not spammed.
  place(e, 30, 22, 0, 14, 22, 0);
  e.agents[TAGGER].vx = e.agents[TAGGER].spec.maxSpeed;
  let crossed = false;
  for (let i = 0; i < 40 && !e.tagged; i++) {
    const r = e.agents[RUNNER];
    const d = Math.abs(e.agents[TAGGER].x - r.x);
    const res = e.step([d < 7 && r.grounded ? JUMP : NOOP, FWD]);
    if (Math.abs(e.agents[TAGGER].x - r.x) < 1) crossed = true;
    if (res.done) break;
  }
  assert.ok(crossed, 'the tagger should have run underneath the runner');
  assert.equal(e.tagged, false, 'passing underneath must not count as a tag');
});

check('an agent can land on a crate and stand on it', () => {
  const e = env(1);
  const crate = e.boxes[0];
  crate.x = 26;
  crate.y = 22;
  crate.z = 0;
  crate.vx = crate.vy = crate.vz = 0;
  place(e, 21.5, 22, 0, 40, 40, 0);
  for (let i = 0; i < 30; i++) {
    e.step([e.agents[RUNNER].grounded ? FWD_JUMP : FWD, NOOP]);
    if (e.agents[RUNNER].grounded && e.agents[RUNNER].z > 1) break;
  }
  const r = e.agents[RUNNER];
  assert.ok(r.z > 1, `runner should be standing on the crate, z=${r.z.toFixed(2)}`);
  assert.equal(r.grounded, true);
});

check('crates fall and settle on the floor', () => {
  const e = env(1);
  const crate = e.boxes[0];
  crate.z = 12;
  crate.vx = crate.vy = crate.vz = 0;
  for (let i = 0; i < 40; i++) e.step([NOOP, NOOP]);
  assert.equal(crate.z, 0, 'a dropped crate ends up on the floor');
});

check('a crate in front can be grabbed, carried and released', () => {
  const e = env(1);
  const crate = e.boxes[0];
  crate.x = 25;
  crate.y = 22;
  crate.z = 0;
  crate.vx = crate.vy = crate.vz = 0;
  place(e, 22, 22, 0, 40, 40, 0); // facing +x, crate 3 units ahead
  e.step([GRAB, NOOP]);

  const r = e.agents[RUNNER];
  assert.equal(r.held, 0, 'the crate in front should be picked up');
  assert.equal(crate.heldBy, RUNNER);

  const before = crate.x;
  for (let i = 0; i < 8; i++) e.step([FWD_GRAB, NOOP]);
  assert.ok(crate.x > before + 1, 'the crate should be carried along');
  assert.ok(
    Math.hypot(crate.x - r.x, crate.y - r.y) < BOX + 2,
    'and should stay in front of the carrier',
  );

  e.step([NOOP, NOOP]); // let go
  assert.equal(r.held, -1);
  assert.equal(crate.heldBy, -1);
});

check('picking up a crate pays the runner exactly once per round', () => {
  const e = env(1);
  const crate = e.boxes[0];
  crate.x = 25;
  crate.y = 22;
  crate.z = 0;
  place(e, 22, 22, 0, 40, 40, 0);
  const first = e.step([GRAB, NOOP]).rewards[RUNNER];
  const second = e.step([GRAB, NOOP]).rewards[RUNNER];
  assert.ok(first > 0.5, `first grab should pay the bonus, got ${first}`);
  assert.ok(second < 0.5, `holding on should not pay again, got ${second}`);
});

check('agents stay inside the room', () => {
  const e = env(2);
  place(e, 22, 22, 0, 22, 30, Math.PI);
  for (let i = 0; i < 200; i++) {
    e.step([FWD_JUMP, FWD_JUMP]);
    for (const a of e.agents) {
      assert.ok(a.x >= 0 && a.x <= 44 && a.y >= 0 && a.y <= 44, 'agent escaped the room');
      assert.ok(a.z >= 0 && a.z < 40, `agent left the world vertically, z=${a.z}`);
      assert.ok(Number.isFinite(a.x + a.y + a.z), 'agent state went non-finite');
    }
    for (const c of e.boxes) {
      assert.ok(Number.isFinite(c.x + c.y + c.z), 'crate state went non-finite');
      assert.ok(c.z >= 0, 'crate fell through the floor');
    }
    if (e.frame >= 600) e.reset();
  }
});

check('observations survive a role with jumping disabled', () => {
  // Regression: vertical speed used to be normalised by the per-role jump
  // impulse, so jumpV = 0 produced 0/0 and silently NaN-poisoned the network.
  const saved = SPECS.map((s) => s.jumpV);
  SPECS[RUNNER].jumpV = 0;
  SPECS[TAGGER].jumpV = 0;
  try {
    const e = env(0);
    const obs = new Float32Array(1024);
    for (let i = 0; i < 40; i++) {
      e.step([FWD_JUMP, FWD_JUMP]);
      for (const role of [RUNNER, TAGGER]) {
        const n = e.observe(role, obs, 0);
        for (let j = 0; j < n; j++) {
          assert.ok(Number.isFinite(obs[j]), `obs[${j}] not finite with jumpV=0`);
        }
      }
    }
  } finally {
    SPECS[RUNNER].jumpV = saved[RUNNER];
    SPECS[TAGGER].jumpV = saved[TAGGER];
  }
});

check('observations are finite and bounded in every room', () => {
  for (let room = 0; room < 3; room++) {
    const e = env(room);
    const obs = new Float32Array(1024);
    const rng = new RNG(9);
    for (let i = 0; i < 150; i++) {
      const rnd = () => [rng.int(3), rng.int(3), rng.int(2), rng.int(2)];
      const res = e.step([rnd(), rnd()]);
      for (const role of [RUNNER, TAGGER]) {
        const n = e.observe(role, obs, 0);
        for (let j = 0; j < n; j++) {
          assert.ok(Number.isFinite(obs[j]), `room ${room} obs[${j}] not finite`);
          assert.ok(Math.abs(obs[j]) < 10, `room ${room} obs[${j}] = ${obs[j]} out of range`);
        }
      }
      if (res.done) e.reset();
    }
  }
});

console.log(`\n${passed} checks passed`);
