// The tag arena — a real 3D room.
//
// Role 0 = RUNNER (orange, "Albert")  -- rewarded for staying alive
// Role 1 = TAGGER (blue,  "Kai")      -- rewarded for tagging, punished for time
//
// Agents move on the floor, jump under gravity, stand on top of crates, and
// can grab and carry (or throw) the loose crates. Height matters: a tag only
// lands if the two bodies overlap vertically as well as horizontally, so
// jumping over an incoming tagger is a legal escape.
//
// Each agent emits four numbers, as in the reference:
//   [move, turn, jump, grab]  ->  e.g. [1, 2, 0, 1]

export const RUNNER = 0;
export const TAGGER = 1;

export const ARENA_W = 44;
export const ARENA_H = 44;

export const N_RAYS = 13;
export const RAY_FOV = (220 * Math.PI) / 180;
export const RAY_MAX = 32;
export const RAY_KINDS = 3; // wall / crate / opponent

export const BOX = 3.2; // crate edge length
export const BOX_HALF = BOX / 2;
export const MAX_BOXES = 2; // observation slots reserved for crates

export const AGENT_R = 1.6; // collision radius in xy
export const AGENT_H = 3.4; // body height

export const GRAVITY = 34;
// Per-role jump/air handling lives in SPECS below. JUMP_V stays a fixed
// reference scale for normalising vertical speed in observations: tying that
// to a per-role tunable makes the input mean different things for the two
// agents, and divides by zero outright if a role has jumping disabled.
export const JUMP_V = 17.5;
export const VZ_SCALE = JUMP_V;

export const GRAB_REACH = 5.0;
export const GRAB_ARC = Math.cos(1.0); // ±~57° in front

export const DIRECT_OBS = 23; // self (10) + opponent (12) + time (1)
export const PREV_ACT_OBS = 10; // 3 + 3 + 2 + 2
export const BOX_OBS = MAX_BOXES * 7;
export const OBS_DIM = DIRECT_OBS + PREV_ACT_OBS + BOX_OBS + N_RAYS * (1 + RAY_KINDS); // 99
export const BRANCHES = [3, 3, 2, 2]; // move, turn, jump, grab

export const DT = 1 / 60;
export const FRAME_SKIP = 4; // one decision every 4 physics frames
// Round length is the single most important balance number in the game. In a
// bounded arena a pursuer corners an evader eventually regardless of speeds —
// a scripted evader escapes 0% of 10s rounds even when it is FASTER than the
// pursuer — so the clock, not the physics, decides whether evasion can pay.
export let EPISODE_FRAMES = 360; // 6 seconds
export function setRoundFrames(n) {
  EPISODE_FRAMES = n;
}

// The tagger is faster in a straight line; the runner turns harder.
// That asymmetry is what makes juking a winning strategy instead of a tie.
export const SPECS = [
  // jumpV: apex = jumpV^2 / (2*GRAVITY). air: how much steering survives a jump.
  // jumpCd: seconds before another jump is allowed, which stops bunny-hopping
  // from being a free action.
  // Turn radius = maxSpeed / turnRate, and that is the number that decides
  // whether juking works: the evader can only cut inside the pursuer if the
  // pursuer's turn radius is meaningfully wider than the tag radius.
  { maxSpeed: 12.5, turnRate: 5.8, accel: 9.0, jumpV: 17.5, air: 0.35, jumpCd: 0 }, // runner
  { maxSpeed: 13.6, turnRate: 2.4, accel: 9.0, jumpV: 17.5, air: 0.35, jumpCd: 0 }, // tagger
];

// Mutable so experiments can vary it; `dist` is how close the two bodies must
// be for a tag to land.
export const TAG = { dist: AGENT_R * 2 };

export const ALIVE_BONUS = 0.001; // per physics frame, per the video's reward fn
export const FIRST_GRAB_BONUS = 1; // one-off, so the runner discovers the crates

// Low enough that a crate-assisted jump reaches the top (3.2 + 4.5 > 6),
// but too high to reach from the floor alone.
export const WALL_TOP = 6;

export const ROOMS = [
  { name: 'Room 1 — Open', obstacles: [], boxes: [] },
  {
    name: 'Room 2 — Crates',
    obstacles: [
      { x0: 10, y0: 10, x1: 16, y1: 16 },
      { x0: 28, y0: 28, x1: 34, y1: 34 },
    ],
    boxes: [
      { x: 30, y: 13 },
      { x: 14, y: 31 },
    ],
  },
  {
    name: 'Room 3 — Maze',
    obstacles: [
      { x0: 12, y0: 0, x1: 15, y1: 17 },
      { x0: 12, y0: 27, x1: 15, y1: 44 },
      { x0: 29, y0: 9, x1: 32, y1: 35 },
      { x0: 17, y0: 20, x1: 27, y1: 23 },
    ],
    boxes: [
      { x: 22, y: 8 },
      { x: 22, y: 38 },
    ],
  },
];

const MOVE_INPUT = [0, 1, -1];
const TURN_INPUT = [0, 1, -1];

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Ray vs axis-aligned box in xy, gated on the ray's height. Returns t or -1. */
function rayBox(px, py, dx, dy, b, h, zLo, zHi) {
  if (h < zLo || h > zHi) return -1;
  const invx = dx !== 0 ? 1 / dx : Infinity;
  const invy = dy !== 0 ? 1 / dy : Infinity;
  let t0 = (b.x0 - px) * invx;
  let t1 = (b.x1 - px) * invx;
  if (t0 > t1) [t0, t1] = [t1, t0];
  let s0 = (b.y0 - py) * invy;
  let s1 = (b.y1 - py) * invy;
  if (s0 > s1) [s0, s1] = [s1, s0];
  const tEnter = Math.max(t0, s0);
  const tExit = Math.min(t1, s1);
  if (tExit < 0 || tEnter > tExit) return -1;
  return tEnter >= 0 ? tEnter : 0;
}

function rayArenaExit(px, py, dx, dy) {
  let t = Infinity;
  if (dx > 1e-9) t = Math.min(t, (ARENA_W - px) / dx);
  else if (dx < -1e-9) t = Math.min(t, -px / dx);
  if (dy > 1e-9) t = Math.min(t, (ARENA_H - py) / dy);
  else if (dy < -1e-9) t = Math.min(t, -py / dy);
  return t;
}

function rayCircle(px, py, dx, dy, cx, cy, r) {
  const ox = px - cx;
  const oy = py - cy;
  const b = ox * dx + oy * dy;
  const c = ox * ox + oy * oy - r * r;
  if (c > 0 && b > 0) return -1;
  const disc = b * b - c;
  if (disc < 0) return -1;
  const t = -b - Math.sqrt(disc);
  return t >= 0 ? t : 0;
}

export class Agent {
  constructor(role) {
    this.role = role;
    this.spec = SPECS[role];
    this.x = 0;
    this.y = 0;
    this.z = 0; // height of the body's underside
    this.th = 0;
    this.vx = 0;
    this.vy = 0;
    this.vz = 0;
    this.grounded = true;
    this.jumpTimer = 0; // seconds left before another jump is allowed
    this.held = -1; // index of the crate being carried, or -1
    this.act = [0, 0, 0, 0];
    this.rays = new Float32Array(N_RAYS * 2); // [dist, kind] per ray, for drawing
  }
  get top() {
    return this.z + AGENT_H;
  }
}

export class Crate {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.z = 0;
    this.vx = 0;
    this.vy = 0;
    this.vz = 0;
    this.heldBy = -1;
  }
  get top() {
    return this.z + BOX;
  }
  aabb() {
    return { x0: this.x - BOX_HALF, y0: this.y - BOX_HALF, x1: this.x + BOX_HALF, y1: this.y + BOX_HALF };
  }
}

export class TagEnv {
  constructor(rng, roomIndex = 0, opts = {}) {
    this.rng = rng;
    this.agents = [new Agent(RUNNER), new Agent(TAGGER)];
    this.boxes = [];
    this.setRoom(roomIndex);
    this.frame = 0;
    this.tagged = false;
    this.opponentId = 0;
    this.shaping = opts.shaping || 0;
    // Scale applied to the runner's half of the shaping term. Chasing pure
    // distance is good advice for the tagger but bad for the runner: in a
    // closed room the farthest point from your pursuer is usually a corner.
    this.shapingRunner = opts.shapingRunner ?? 1;
    this.trail = [[], []];
    this.spawn = [null, null];
    this.grabbedThisEpisode = false;
  }

  setRoom(i) {
    this.roomIndex = i;
    this.obstacles = ROOMS[i].obstacles;
    this.boxSpawns = ROOMS[i].boxes;
    this.boxes = this.boxSpawns.map((b) => new Crate(b.x, b.y));
  }

  _blockedAt(x, y, pad) {
    for (const b of this.obstacles) {
      if (x > b.x0 - pad && x < b.x1 + pad && y > b.y0 - pad && y < b.y1 + pad) return true;
    }
    return false;
  }

  _placeAgent(a, minDistFrom) {
    for (let attempt = 0; attempt < 200; attempt++) {
      const x = this.rng.range(AGENT_R + 1, ARENA_W - AGENT_R - 1);
      const y = this.rng.range(AGENT_R + 1, ARENA_H - AGENT_R - 1);
      if (this._blockedAt(x, y, AGENT_R + 0.5)) continue;
      if (minDistFrom && Math.hypot(x - minDistFrom.x, y - minDistFrom.y) < minDistFrom.d) continue;
      a.x = x;
      a.y = y;
      return;
    }
    a.x = ARENA_W * 0.5;
    a.y = ARENA_H * 0.5;
  }

  reset() {
    this.frame = 0;
    this.tagged = false;
    this.grabbedThisEpisode = false;
    const [runner, tagger] = this.agents;
    this._placeAgent(runner, null);
    this._placeAgent(tagger, { x: runner.x, y: runner.y, d: 14 });
    for (const a of this.agents) {
      a.th = this.rng.range(-Math.PI, Math.PI);
      a.vx = a.vy = a.vz = 0;
      a.z = 0;
      a.grounded = true;
      a.jumpTimer = 0;
      a.held = -1;
      a.act = [0, 0, 0, 0];
    }
    this.boxes = this.boxSpawns.map((b, i) => {
      const c = new Crate(b.x, b.y);
      // jitter so the crates aren't a memorised landmark
      c.x = clamp(b.x + this.rng.range(-3, 3), BOX_HALF + 0.5, ARENA_W - BOX_HALF - 0.5);
      c.y = clamp(b.y + this.rng.range(-3, 3), BOX_HALF + 0.5, ARENA_H - BOX_HALF - 0.5);
      return c;
    });
    this.spawn = [
      { x: runner.x, y: runner.y },
      { x: tagger.x, y: tagger.y },
    ];
    this.trail = [[], []];
    this._castAll();
    return this;
  }

  /**
   * Advance one decision (FRAME_SKIP physics frames).
   * actions: [[move,turn,jump,grab] for runner, ... for tagger]
   */
  step(actions) {
    const [runner, tagger] = this.agents;
    for (let i = 0; i < 2; i++) this.agents[i].act = actions[i];

    let rRunner = 0;
    let rTagger = 0;
    const d0 = Math.hypot(tagger.x - runner.x, tagger.y - runner.y);

    for (let f = 0; f < FRAME_SKIP && !this.tagged; f++) {
      for (let i = 0; i < 2; i++) this._doGrab(this.agents[i], actions[i][3]);
      for (let i = 0; i < 2; i++) this._integrate(this.agents[i], actions[i]);
      this._stepBoxes();
      this.frame++;

      rRunner += ALIVE_BONUS;
      rTagger -= ALIVE_BONUS;

      if (this._touching(runner, tagger)) this.tagged = true;
    }

    if (!this.grabbedThisEpisode && runner.held >= 0) {
      // One-off nudge so the runner ever tries the crates at all; without it
      // the reward is far too infrequent to discover them.
      this.grabbedThisEpisode = true;
      rRunner += FIRST_GRAB_BONUS;
    }

    if (this.tagged) {
      rRunner -= 1;
      rTagger += 1;
    }

    if (this.shaping > 0) {
      // Potential-based shaping on Φ = -distance. Because it telescopes it
      // cannot change which policy is optimal (Ng et al., 1999) — it only
      // gives the tagger a gradient before it has ever scored, which is what
      // lets this converge in minutes instead of days.
      const d1 = Math.hypot(tagger.x - runner.x, tagger.y - runner.y);
      const s = this.shaping * (d0 - d1);
      rTagger += s;
      rRunner -= s * this.shapingRunner;
    }

    this._castAll();
    this._pushTrail();

    const done = this.tagged || this.frame >= EPISODE_FRAMES;
    return { done, tagged: this.tagged, rewards: [rRunner, rTagger] };
  }

  /** Tag requires overlap in xy AND in height — you can jump over a tagger. */
  _touching(a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    if (dx * dx + dy * dy > TAG.dist * TAG.dist) return false;
    return a.z < b.top && b.z < a.top;
  }

  // ------------------------------------------------------------- grabbing
  _doGrab(a, want) {
    const idx = this.agents.indexOf(a);
    if (!want) {
      if (a.held >= 0) {
        this.boxes[a.held].heldBy = -1;
        a.held = -1;
      }
      return;
    }
    if (a.held >= 0) return;

    const c = Math.cos(a.th);
    const s = Math.sin(a.th);
    let best = -1;
    let bestD = GRAB_REACH;
    for (let i = 0; i < this.boxes.length; i++) {
      const b = this.boxes[i];
      if (b.heldBy >= 0) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.hypot(dx, dy);
      if (d > bestD || d < 1e-3) continue;
      if ((dx * c + dy * s) / d < GRAB_ARC) continue; // must be in front
      if (b.top < a.z || b.z > a.top) continue; // must be at a reachable height
      best = i;
      bestD = d;
    }
    if (best >= 0) {
      a.held = best;
      this.boxes[best].heldBy = idx;
    }
  }

  // ------------------------------------------------------------- movement
  _integrate(a, act) {
    const spec = a.spec;
    a.th += TURN_INPUT[act[1]] * spec.turnRate * DT;
    if (a.th > Math.PI) a.th -= 2 * Math.PI;
    else if (a.th < -Math.PI) a.th += 2 * Math.PI;

    const drive = MOVE_INPUT[act[0]];
    const tx = Math.cos(a.th) * drive * spec.maxSpeed;
    const ty = Math.sin(a.th) * drive * spec.maxSpeed;
    const k = Math.min(1, spec.accel * DT * (a.grounded ? 1 : spec.air));
    a.vx += (tx - a.vx) * k;
    a.vy += (ty - a.vy) * k;

    if (a.jumpTimer > 0) a.jumpTimer -= DT;
    if (act[2] === 1 && a.grounded && a.jumpTimer <= 0) {
      a.vz = spec.jumpV;
      a.grounded = false;
      a.jumpTimer = spec.jumpCd;
    }
    a.vz -= GRAVITY * DT;

    a.x += a.vx * DT;
    a.y += a.vy * DT;
    this._collideXY(a);

    a.z += a.vz * DT;
    this._collideZ(a);
  }

  /** Highest surface directly under `a` that it could be standing on. */
  _supportUnder(a) {
    let s = 0;
    for (const b of this.obstacles) {
      if (this._overlapsAABB(a.x, a.y, AGENT_R, b) && a.z >= WALL_TOP - 0.35) s = Math.max(s, WALL_TOP);
    }
    for (const c of this.boxes) {
      if (c.heldBy >= 0) continue;
      if (this._overlapsAABB(a.x, a.y, AGENT_R, c.aabb()) && a.z >= c.top - 0.35) {
        s = Math.max(s, c.top);
      }
    }
    return s;
  }

  _overlapsAABB(x, y, r, b) {
    const cx = clamp(x, b.x0, b.x1);
    const cy = clamp(y, b.y0, b.y1);
    const dx = x - cx;
    const dy = y - cy;
    return dx * dx + dy * dy < r * r;
  }

  _collideZ(a) {
    const support = this._supportUnder(a);
    if (a.z <= support) {
      a.z = support;
      if (a.vz < 0) a.vz = 0;
      a.grounded = true;
    } else {
      a.grounded = false;
    }
  }

  _collideXY(a) {
    if (a.x < AGENT_R) {
      a.x = AGENT_R;
      if (a.vx < 0) a.vx = 0;
    } else if (a.x > ARENA_W - AGENT_R) {
      a.x = ARENA_W - AGENT_R;
      if (a.vx > 0) a.vx = 0;
    }
    if (a.y < AGENT_R) {
      a.y = AGENT_R;
      if (a.vy < 0) a.vy = 0;
    } else if (a.y > ARENA_H - AGENT_R) {
      a.y = ARENA_H - AGENT_R;
      if (a.vy > 0) a.vy = 0;
    }

    // Fixed blocks: only if the body actually overlaps them vertically,
    // otherwise the agent is standing on top and must not be shoved.
    for (const b of this.obstacles) {
      if (a.z >= WALL_TOP - 0.05) continue;
      this._pushCircleOutOfBox(a, b, 1);
    }

    // Loose crates: the agent shoves them, and takes part of the push back.
    for (let i = 0; i < this.boxes.length; i++) {
      const c = this.boxes[i];
      if (c.heldBy >= 0) continue;
      if (a.z >= c.top - 0.05 || c.z >= a.top - 0.05) continue;
      const before = { x: a.x, y: a.y };
      this._pushCircleOutOfBox(a, c.aabb(), 0.42);
      const mx = a.x - before.x;
      const my = a.y - before.y;
      if (mx || my) {
        // The share the agent did NOT absorb is applied to the crate.
        c.x -= (mx / 0.42) * 0.58;
        c.y -= (my / 0.42) * 0.58;
        c.vx = -(mx / 0.42) * 0.58 / DT * 0.55;
        c.vy = -(my / 0.42) * 0.58 / DT * 0.55;
      }
    }
  }

  /** Separate a circle from an AABB; `w` is the share of the fix the circle takes. */
  _pushCircleOutOfBox(a, b, w) {
    const cx = clamp(a.x, b.x0, b.x1);
    const cy = clamp(a.y, b.y0, b.y1);
    let dx = a.x - cx;
    let dy = a.y - cy;
    let d2 = dx * dx + dy * dy;
    if (d2 >= AGENT_R * AGENT_R) return;

    if (d2 > 1e-12) {
      const d = Math.sqrt(d2);
      const push = (AGENT_R - d) * w;
      const nx = dx / d;
      const ny = dy / d;
      a.x += nx * push;
      a.y += ny * push;
      const vn = a.vx * nx + a.vy * ny;
      if (vn < 0) {
        a.vx -= vn * nx;
        a.vy -= vn * ny;
      }
    } else {
      const left = a.x - b.x0;
      const right = b.x1 - a.x;
      const down = a.y - b.y0;
      const up = b.y1 - a.y;
      const m = Math.min(left, right, down, up);
      if (m === left) {
        a.x = b.x0 - AGENT_R;
        a.vx = Math.min(a.vx, 0);
      } else if (m === right) {
        a.x = b.x1 + AGENT_R;
        a.vx = Math.max(a.vx, 0);
      } else if (m === down) {
        a.y = b.y0 - AGENT_R;
        a.vy = Math.min(a.vy, 0);
      } else {
        a.y = b.y1 + AGENT_R;
        a.vy = Math.max(a.vy, 0);
      }
    }
  }

  // ---------------------------------------------------------------- crates
  _stepBoxes() {
    for (const c of this.boxes) {
      if (c.heldBy >= 0) {
        const a = this.agents[c.heldBy];
        const tx = a.x + Math.cos(a.th) * (AGENT_R + BOX_HALF + 0.35);
        const ty = a.y + Math.sin(a.th) * (AGENT_R + BOX_HALF + 0.35);
        const tz = a.z + 0.9;
        const px = c.x;
        const py = c.y;
        c.x += (tx - c.x) * 0.5;
        c.y += (ty - c.y) * 0.5;
        c.z += (tz - c.z) * 0.5;
        c.vx = (c.x - px) / DT;
        c.vy = (c.y - py) / DT;
        c.vz = 0;
        // Detach if the carry point is jammed into geometry, the same way the
        // reference drops a crate when the holding force gets too high.
        const jam =
          c.x < BOX_HALF ||
          c.x > ARENA_W - BOX_HALF ||
          c.y < BOX_HALF ||
          c.y > ARENA_H - BOX_HALF ||
          Math.hypot(c.x - tx, c.y - ty) > BOX;
        c.x = clamp(c.x, BOX_HALF, ARENA_W - BOX_HALF);
        c.y = clamp(c.y, BOX_HALF, ARENA_H - BOX_HALF);
        if (jam) {
          this.agents[c.heldBy].held = -1;
          c.heldBy = -1;
        }
        continue;
      }

      c.vz -= GRAVITY * DT;
      c.x += c.vx * DT;
      c.y += c.vy * DT;
      c.z += c.vz * DT;

      // ground friction
      c.vx *= 0.86;
      c.vy *= 0.86;

      c.x = clamp(c.x, BOX_HALF, ARENA_W - BOX_HALF);
      c.y = clamp(c.y, BOX_HALF, ARENA_H - BOX_HALF);

      // rest on the floor or on top of a fixed block
      let support = 0;
      for (const b of this.obstacles) {
        const overlap = c.x + BOX_HALF > b.x0 && c.x - BOX_HALF < b.x1 && c.y + BOX_HALF > b.y0 && c.y - BOX_HALF < b.y1;
        if (!overlap) continue;
        if (c.z >= WALL_TOP - 0.35) support = Math.max(support, WALL_TOP);
        else {
          // pushed into the side of a block: slide it back out
          const left = c.x + BOX_HALF - b.x0;
          const right = b.x1 - (c.x - BOX_HALF);
          const down = c.y + BOX_HALF - b.y0;
          const up = b.y1 - (c.y - BOX_HALF);
          const m = Math.min(left, right, down, up);
          if (m === left) c.x = b.x0 - BOX_HALF;
          else if (m === right) c.x = b.x1 + BOX_HALF;
          else if (m === down) c.y = b.y0 - BOX_HALF;
          else c.y = b.y1 + BOX_HALF;
          c.vx = c.vy = 0;
        }
      }
      if (c.z <= support) {
        c.z = support;
        if (c.vz < 0) c.vz = 0;
      }
    }

    // keep crates from occupying the same space
    for (let i = 0; i < this.boxes.length; i++) {
      for (let j = i + 1; j < this.boxes.length; j++) {
        const a = this.boxes[i];
        const b = this.boxes[j];
        if (a.heldBy >= 0 && b.heldBy >= 0) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        if (Math.abs(dx) >= BOX || Math.abs(dy) >= BOX) continue;
        if (a.top <= b.z || b.top <= a.z) continue;
        const ox = BOX - Math.abs(dx);
        const oy = BOX - Math.abs(dy);
        const sx = dx >= 0 ? 1 : -1;
        const sy = dy >= 0 ? 1 : -1;
        if (ox < oy) {
          if (a.heldBy < 0) a.x -= (sx * ox) / 2;
          if (b.heldBy < 0) b.x += (sx * ox) / 2;
        } else {
          if (a.heldBy < 0) a.y -= (sy * oy) / 2;
          if (b.heldBy < 0) b.y += (sy * oy) / 2;
        }
      }
    }
  }

  // ---------------------------------------------------------------- vision
  _castAll() {
    this._cast(this.agents[0], this.agents[1]);
    this._cast(this.agents[1], this.agents[0]);
  }

  _cast(a, other) {
    const step = N_RAYS > 1 ? RAY_FOV / (N_RAYS - 1) : 0;
    const start = a.th - RAY_FOV / 2;
    const h = a.z + AGENT_H * 0.65; // eye height — a jumping agent sees over crates
    for (let i = 0; i < N_RAYS; i++) {
      const ang = start + step * i;
      const dx = Math.cos(ang);
      const dy = Math.sin(ang);
      let best = Math.min(RAY_MAX, rayArenaExit(a.x, a.y, dx, dy));
      let kind = 0; // wall

      for (const b of this.obstacles) {
        const t = rayBox(a.x, a.y, dx, dy, b, h, 0, WALL_TOP);
        if (t >= 0 && t < best) {
          best = t;
          kind = 0;
        }
      }
      for (const c of this.boxes) {
        const t = rayBox(a.x, a.y, dx, dy, c.aabb(), h, c.z, c.top);
        if (t >= 0 && t < best) {
          best = t;
          kind = 1; // crate
        }
      }
      if (h >= other.z && h <= other.top) {
        const tc = rayCircle(a.x, a.y, dx, dy, other.x, other.y, AGENT_R);
        if (tc >= 0 && tc < best) {
          best = tc;
          kind = 2; // opponent
        }
      }
      a.rays[i * 2] = best;
      a.rays[i * 2 + 1] = kind;
    }
  }

  _pushTrail() {
    for (let i = 0; i < 2; i++) {
      const t = this.trail[i];
      t.push(this.agents[i].x, this.agents[i].y);
      if (t.length > 120) t.splice(0, t.length - 120);
    }
  }

  // ----------------------------------------------------------- observation
  observe(role, out, offset = 0) {
    const a = this.agents[role];
    const b = this.agents[1 - role];
    const ms = a.spec.maxSpeed;
    const c = Math.cos(a.th);
    const s = Math.sin(a.th);
    let o = offset;

    // --- self (10)
    out[o++] = (a.x - ARENA_W / 2) / (ARENA_W / 2);
    out[o++] = (a.y - ARENA_H / 2) / (ARENA_H / 2);
    out[o++] = a.z / WALL_TOP;
    out[o++] = c;
    out[o++] = s;
    out[o++] = (a.vx * c + a.vy * s) / ms;
    out[o++] = (-a.vx * s + a.vy * c) / ms;
    out[o++] = a.vz / VZ_SCALE;
    out[o++] = a.grounded && a.jumpTimer <= 0 ? 1 : 0; // jump available
    out[o++] = a.held >= 0 ? 1 : 0;

    // --- opponent (12)
    const rx = b.x - a.x;
    const ry = b.y - a.y;
    out[o++] = (rx * c + ry * s) / ARENA_W;
    out[o++] = (-rx * s + ry * c) / ARENA_H;
    out[o++] = (b.z - a.z) / WALL_TOP;
    const dist = Math.hypot(rx, ry);
    out[o++] = dist / ARENA_W;
    out[o++] = Math.exp(-dist / 8);
    const dth = b.th - a.th;
    out[o++] = Math.cos(dth);
    out[o++] = Math.sin(dth);
    out[o++] = (b.vx * c + b.vy * s) / b.spec.maxSpeed;
    out[o++] = (-b.vx * s + b.vy * c) / b.spec.maxSpeed;
    out[o++] = b.vz / VZ_SCALE;
    out[o++] = b.grounded ? 1 : 0;
    out[o++] = b.held >= 0 ? 1 : 0;

    // --- clock (1)
    out[o++] = 1 - this.frame / EPISODE_FRAMES;

    // --- previous action (10)
    for (let k = 0; k < BRANCHES.length; k++) {
      for (let i = 0; i < BRANCHES[k]; i++) out[o++] = a.act[k] === i ? 1 : 0;
    }

    // --- crates (7 each)
    for (let i = 0; i < MAX_BOXES; i++) {
      const cr = this.boxes[i];
      if (!cr) {
        for (let j = 0; j < 7; j++) out[o++] = 0;
        continue;
      }
      const bx = cr.x - a.x;
      const by = cr.y - a.y;
      const bd = Math.hypot(bx, by);
      out[o++] = 1; // this slot holds a real crate
      out[o++] = (bx * c + by * s) / ARENA_W;
      out[o++] = (-bx * s + by * c) / ARENA_H;
      out[o++] = (cr.z - a.z) / WALL_TOP;
      out[o++] = Math.exp(-bd / 8);
      out[o++] = cr.heldBy === role ? 1 : 0;
      out[o++] = cr.heldBy === 1 - role ? 1 : 0;
    }

    // --- raycasts (4 each)
    for (let i = 0; i < N_RAYS; i++) {
      const d = a.rays[i * 2];
      const kind = a.rays[i * 2 + 1];
      out[o++] = 1 - Math.min(d, RAY_MAX) / RAY_MAX;
      out[o++] = kind === 0 ? 1 : 0;
      out[o++] = kind === 1 ? 1 : 0;
      out[o++] = kind === 2 ? 1 : 0;
    }
    return o;
  }
}
