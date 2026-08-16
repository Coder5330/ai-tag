// Flat-shaded 3D renderer on Canvas2D.
//
// The whole scene is axis-aligned boxes plus a floor, viewed from a fixed
// camera outside the room, so a perspective projection with a painter's-order
// draw gives exactly the look we want without a WebGL dependency:
//   walls -> floor & floor decals -> depth-sorted solids.
// That order is always correct here because the camera sits outside a convex
// room looking in: nothing inside can ever be occluded by a wall.

import {
  ARENA_W,
  ARENA_H,
  N_RAYS,
  RAY_FOV,
  RUNNER,
  TAGGER,
  EPISODE_FRAMES,
  AGENT_R,
  AGENT_H,
  BOX,
  BOX_HALF,
  WALL_TOP,
} from './env.js';

const WALL_H = 16;
const TILE = 4; // floor/wall grid spacing, in world units

const C = {
  wallRed: '#a83b36',
  wallLine: 'rgba(255,255,255,0.72)',
  floor: '#474d56',
  floorLine: 'rgba(240,244,250,0.8)',
  block: '#e6e9ee',
  crate: '#c98f4e',
  crateHeld: '#f0c07a',
  runner: '#e86a1c',
  tagger: '#2e86e0',
  panel: '#0d0f13',
  panelEdge: '#2a2f38',
  runnerInk: '#ff9a4d',
  taggerInk: '#6cb4ff',
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function norm(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

function shade(hex, k) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, Math.round(((n >> 16) & 255) * k));
  const g = Math.min(255, Math.round(((n >> 8) & 255) * k));
  const b = Math.min(255, Math.round((n & 255) * k));
  return `rgb(${r},${g},${b})`;
}

function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

export class ArenaRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = 1;
    this.eye = [ARENA_W / 2, -34, 22];
    this.target = [ARENA_W / 2, 20, 5.5];
    this.fovY = (46 * Math.PI) / 180;
    this.resize();
  }

  resize() {
    const c = this.canvas;
    const rect = c.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round((rect.height || rect.width * 0.62) * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    this.dpr = dpr;
    this.w = w;
    this.h = h;

    const zA = norm(sub(this.eye, this.target)); // points back toward the eye
    const xA = norm(cross([0, 0, 1], zA));
    const yA = cross(zA, xA);
    this.basis = { xA, yA, zA };
    this.f = h / 2 / Math.tan(this.fovY / 2);
  }

  /** World point -> { x, y, depth }. depth <= 0 means behind the camera. */
  project(p) {
    const { xA, yA, zA } = this.basis;
    const d = sub(p, this.eye);
    const vz = -dot(d, zA);
    if (vz <= 0.2) return { x: 0, y: 0, depth: vz };
    return {
      x: this.w / 2 + (this.f * dot(d, xA)) / vz,
      y: this.h / 2 - (this.f * dot(d, yA)) / vz,
      depth: vz,
    };
  }

  _poly(pts, fill, stroke, lw) {
    const { ctx } = this;
    const ps = [];
    for (const p of pts) {
      const q = this.project(p);
      if (q.depth <= 0) return false; // fully clip anything crossing the eye plane
      ps.push(q);
    }
    ctx.beginPath();
    ctx.moveTo(ps[0].x, ps[0].y);
    for (let i = 1; i < ps.length; i++) ctx.lineTo(ps[i].x, ps[i].y);
    ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = (lw || 1) * this.dpr;
      ctx.stroke();
    }
    return true;
  }

  _line(a, b, color, lw, alpha = 1) {
    const { ctx } = this;
    const p = this.project(a);
    const q = this.project(b);
    if (p.depth <= 0 || q.depth <= 0) return;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw * this.dpr;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(q.x, q.y);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  _text(world, text, color, px, weight = '700', align = 'center') {
    const { ctx } = this;
    const p = this.project(world);
    if (p.depth <= 0) return;
    // Scale the label with distance so wall text sits in the perspective.
    const size = (px * this.f) / p.depth;
    ctx.fillStyle = color;
    ctx.font = `${weight} ${size}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.fillText(text, p.x, p.y);
  }

  draw(env, prev, alpha, opts = {}) {
    this.resize();
    const { ctx, w, h } = this;

    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, '#0a0c11');
    g.addColorStop(1, '#14171e');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);

    this._walls(env, opts);
    this._floor(env);
    this._trails(env);
    if (opts.showRays) {
      this._cone(env.agents[RUNNER], this._pose(env, prev, alpha, RUNNER), C.runner);
      this._cone(env.agents[TAGGER], this._pose(env, prev, alpha, TAGGER), C.tagger);
    }

    // Drop shadows go on the floor before anything solid — without them you
    // cannot tell a jumping agent from one that is simply further away.
    for (const c of env.boxes) this._shadow(c.x, c.y, BOX_HALF * 1.05, c.z);
    for (const role of [RUNNER, TAGGER]) {
      const p = this._pose(env, prev, alpha, role);
      this._shadow(p.x, p.y, AGENT_R * 1.15, p.z);
    }

    // --- depth-sorted solids
    const faces = [];
    for (const b of env.obstacles) {
      this._boxFaces(faces, b.x0, b.y0, b.x1, b.y1, 0, WALL_TOP, C.block);
    }
    for (const c of env.boxes) {
      this._boxFaces(
        faces,
        c.x - BOX_HALF,
        c.y - BOX_HALF,
        c.x + BOX_HALF,
        c.y + BOX_HALF,
        c.z,
        c.z + BOX,
        c.heldBy >= 0 ? C.crateHeld : C.crate,
        true,
      );
    }
    for (const role of [RUNNER, TAGGER]) {
      this._agentFaces(faces, env.agents[role], this._pose(env, prev, alpha, role), role);
    }
    faces.sort((a, b) => b.depth - a.depth);
    for (const f of faces) {
      if (this._poly(f.pts, f.fill, f.stroke, f.lw) && f.after) f.after();
    }

    for (const role of [RUNNER, TAGGER]) {
      this._nameTag(env.agents[role], this._pose(env, prev, alpha, role), role);
    }

    if (opts.tagFlash > 0) this._flash(opts.tagFlash);
  }

  _pose(env, prev, alpha, role) {
    const a = env.agents[role];
    const p = prev && prev[role];
    if (!p) return { x: a.x, y: a.y, z: a.z, th: a.th };
    return {
      x: p.x + (a.x - p.x) * alpha,
      y: p.y + (a.y - p.y) * alpha,
      z: p.z + (a.z - p.z) * alpha,
      th: lerpAngle(p.th, a.th, alpha),
    };
  }

  // ------------------------------------------------------------------ room
  _walls(env, opts) {
    const W = ARENA_W;
    const H = ARENA_H;

    // back wall (y = H), then the two side walls
    this._poly(
      [
        [0, H, 0],
        [W, H, 0],
        [W, H, WALL_H],
        [0, H, WALL_H],
      ],
      shade(C.wallRed, 1),
    );
    for (let x = 0; x <= W; x += TILE)
      this._line([x, H, 0], [x, H, WALL_H], C.wallLine, 1.2, 0.5);
    for (let z = 0; z <= WALL_H; z += TILE)
      this._line([0, H, z], [W, H, z], C.wallLine, 1.2, 0.5);

    for (const [x, k] of [
      [0, 0.72],
      [W, 0.72],
    ]) {
      this._poly(
        [
          [x, 0, 0],
          [x, H, 0],
          [x, H, WALL_H],
          [x, 0, WALL_H],
        ],
        shade(C.wallRed, k),
      );
      for (let y = 0; y <= H; y += TILE)
        this._line([x, y, 0], [x, y, WALL_H], C.wallLine, 1.2, 0.32);
      for (let z = 0; z <= WALL_H; z += TILE)
        this._line([x, 0, z], [x, H, z], C.wallLine, 1.2, 0.32);
    }

    this._displays(env, opts);
  }

  /** Scoreboards mounted on the back wall, as in the reference. */
  _displays(env, opts) {
    const H = ARENA_H;
    const panel = (x0, x1, z0, z1) => {
      this._poly(
        [
          [x0, H - 0.35, z0],
          [x1, H - 0.35, z0],
          [x1, H - 0.35, z1],
          [x0, H - 0.35, z1],
        ],
        C.panel,
        C.panelEdge,
        1.5,
      );
    };

    // left: Albert's escapes · right: Kai's tags
    panel(3.5, 14.5, 9.4, 13.2);
    this._text([9, H - 0.4, 11.3], String(opts.escapes ?? 0), C.runnerInk, 2.6);
    panel(29.5, 40.5, 9.4, 13.2);
    this._text([35, H - 0.4, 11.3], String(opts.tags ?? 0), C.taggerInk, 2.6);

    // centre: seconds left in the round, and the round number below it
    panel(17.5, 26.5, 10.2, 14.2);
    const left = Math.max(0, (EPISODE_FRAMES - env.frame) / 60);
    this._text([22, H - 0.4, 12.2], left.toFixed(1), '#f2f5f9', 2.8);
    panel(18.5, 25.5, 6.6, 9.4);
    this._text([22, H - 0.4, 8], `#${opts.round ?? 1}`, '#8d97a8', 1.9);
  }

  _floor(env) {
    const W = ARENA_W;
    const H = ARENA_H;
    this._poly(
      [
        [0, 0, 0],
        [W, 0, 0],
        [W, H, 0],
        [0, H, 0],
      ],
      C.floor,
    );
    for (let x = 0; x <= W; x += TILE) this._line([x, 0, 0], [x, H, 0], C.floorLine, 1.2, 0.5);
    for (let y = 0; y <= H; y += TILE) this._line([0, y, 0], [W, y, 0], C.floorLine, 1.2, 0.5);

    if (env.spawn) {
      this._stripes(env.spawn[RUNNER], C.runner);
      this._stripes(env.spawn[TAGGER], C.tagger);
    }
  }

  /** Hatched patch marking where an agent started this round. */
  _stripes(p, color) {
    if (!p) return;
    const r = 3.2;
    const x0 = Math.max(0, p.x - r);
    const x1 = Math.min(ARENA_W, p.x + r);
    const y0 = Math.max(0, p.y - r);
    const y1 = Math.min(ARENA_H, p.y + r);
    if (x1 <= x0 || y1 <= y0) return;

    this._poly(
      [
        [x0, y0, 0.01],
        [x1, y0, 0.01],
        [x1, y1, 0.01],
        [x0, y1, 0.01],
      ],
      'rgba(0,0,0,0.22)',
    );
    // Diagonals y = x - c, each clipped to the patch.
    for (let c = x0 - y1; c <= x1 - y0; c += 1.5) {
      const ax = Math.max(x0, y0 + c);
      const bx = Math.min(x1, y1 + c);
      if (bx <= ax) continue;
      this._line([ax, ax - c, 0.02], [bx, bx - c, 0.02], color, 2.4, 0.5);
    }
  }

  _trails(env) {
    for (const [role, color] of [
      [RUNNER, C.runner],
      [TAGGER, C.tagger],
    ]) {
      const pts = env.trail[role];
      const n = pts.length / 2;
      for (let i = 1; i < n; i++) {
        const t = i / n;
        this._line(
          [pts[(i - 1) * 2], pts[(i - 1) * 2 + 1], 0.05],
          [pts[i * 2], pts[i * 2 + 1], 0.05],
          color,
          1 + 2.2 * t,
          0.03 + 0.22 * t * t,
        );
      }
    }
  }

  /** Translucent view volume built from the agent's actual raycast hits. */
  _cone(agent, pose, color) {
    const { ctx } = this;
    const step = N_RAYS > 1 ? RAY_FOV / (N_RAYS - 1) : 0;
    const start = pose.th - RAY_FOV / 2;
    const zBot = pose.z + 0.06;
    const zTop = pose.z + AGENT_H * 0.9;

    const rim = [];
    for (let i = 0; i < N_RAYS; i++) {
      const ang = start + step * i;
      const d = agent.rays[i * 2];
      rim.push([pose.x + Math.cos(ang) * d, pose.y + Math.sin(ang) * d, agent.rays[i * 2 + 1]]);
    }

    ctx.save();
    // floor footprint
    for (let i = 0; i < rim.length - 1; i++) {
      ctx.globalAlpha = 0.16;
      this._poly(
        [
          [pose.x, pose.y, zBot],
          [rim[i][0], rim[i][1], zBot],
          [rim[i + 1][0], rim[i + 1][1], zBot],
        ],
        color,
      );
    }
    // the slab of air the agent can see through
    for (let i = 0; i < rim.length - 1; i++) {
      ctx.globalAlpha = rim[i][2] === 2 || rim[i + 1][2] === 2 ? 0.32 : 0.09;
      this._poly(
        [
          [rim[i][0], rim[i][1], zBot],
          [rim[i + 1][0], rim[i + 1][1], zBot],
          [rim[i + 1][0], rim[i + 1][1], zTop],
          [rim[i][0], rim[i][1], zTop],
        ],
        color,
      );
    }
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  // ---------------------------------------------------------------- solids
  _shadow(x, y, r, z) {
    // Fades and spreads with height, like a real contact shadow.
    const k = Math.max(0, 1 - z / 9);
    const rr = r * (1 + z * 0.06);
    const pts = [];
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      pts.push([x + Math.cos(a) * rr, y + Math.sin(a) * rr, 0.015]);
    }
    this.ctx.globalAlpha = 0.12 + 0.3 * k;
    this._poly(pts, '#05070a');
    this.ctx.globalAlpha = 1;
  }

  _boxFaces(out, x0, y0, x1, y1, z0, z1, base, banded) {
    const v = [
      [x0, y0, z0],
      [x1, y0, z0],
      [x1, y1, z0],
      [x0, y1, z0],
      [x0, y0, z1],
      [x1, y0, z1],
      [x1, y1, z1],
      [x0, y1, z1],
    ];
    // top, front(-y), back(+y), left, right — with fixed per-normal shading
    const quads = [
      [[4, 5, 6, 7], 1.0],
      [[0, 1, 5, 4], 0.8],
      [[3, 2, 6, 7], 0.62],
      [[0, 3, 7, 4], 0.7],
      [[1, 2, 6, 5], 0.7],
    ];
    for (const [idx, k] of quads) {
      const pts = idx.map((i) => v[i]);
      const face = { pts, fill: shade(base, k), depth: this._depth(pts) };
      if (banded) {
        face.stroke = 'rgba(60,38,14,0.55)';
        face.lw = 1.4;
      }
      out.push(face);
    }
  }

  _depth(pts) {
    let s = 0;
    for (const p of pts) s += this.project(p).depth;
    return s / pts.length;
  }

  _agentFaces(out, agent, pose, role) {
    const color = role === RUNNER ? C.runner : C.tagger;
    const s = AGENT_R * 1.15;
    const c = Math.cos(pose.th);
    const sn = Math.sin(pose.th);
    // local (forward, left, up) -> world
    const P = (fwd, left, up) => [
      pose.x + c * fwd - sn * left,
      pose.y + sn * fwd + c * left,
      pose.z + up,
    ];
    const H = AGENT_H;
    const v = [
      P(-s, -s, 0),
      P(s, -s, 0),
      P(s, s, 0),
      P(-s, s, 0),
      P(-s, -s, H),
      P(s, -s, H),
      P(s, s, H),
      P(-s, s, H),
    ];
    const quads = [
      [[4, 5, 6, 7], 1.06, 'top'],
      [[1, 2, 6, 5], 0.95, 'front'], // +forward face
      [[0, 3, 7, 4], 0.6, 'back'],
      [[0, 1, 5, 4], 0.78, 'sideR'],
      [[3, 2, 6, 7], 0.78, 'sideL'],
    ];
    for (const [idx, k, kind] of quads) {
      const pts = idx.map((i) => v[i]);
      const face = { pts, fill: shade(color, k), depth: this._depth(pts) };
      if (kind === 'front') {
        // Eyes, drawn immediately after the face they sit on.
        face.after = () => {
          const e = 0.02;
          for (const side of [-0.42, 0.42]) {
            this._poly(
              [
                P(s + e, side * s - 0.3 * s, 0.56 * H),
                P(s + e, side * s + 0.3 * s, 0.56 * H),
                P(s + e, side * s + 0.3 * s, 0.79 * H),
                P(s + e, side * s - 0.3 * s, 0.79 * H),
              ],
              '#ffffff',
            );
            this._poly(
              [
                P(s + e * 2, side * s - 0.13 * s, 0.61 * H),
                P(s + e * 2, side * s + 0.13 * s, 0.61 * H),
                P(s + e * 2, side * s + 0.13 * s, 0.73 * H),
                P(s + e * 2, side * s - 0.13 * s, 0.73 * H),
              ],
              '#15181e',
            );
          }
        };
      }
      out.push(face);
    }
  }

  _nameTag(agent, pose, role) {
    this._text(
      [pose.x, pose.y, pose.z + AGENT_H + 1.5],
      role === RUNNER ? 'ALBERT' : 'KAI',
      role === RUNNER ? C.runnerInk : C.taggerInk,
      1.0,
      '700',
    );
  }

  _flash(t) {
    const { ctx, w, h } = this;
    ctx.save();
    ctx.globalAlpha = Math.min(0.42, t * 0.42);
    ctx.fillStyle = C.tagger;
    ctx.fillRect(0, 0, w, h);
    ctx.globalAlpha = Math.min(1, t * 1.6);
    ctx.fillStyle = '#ffffff';
    ctx.font = `800 ${Math.round(h * 0.09)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('TAGGED', w / 2, h * 0.52);
    ctx.restore();
  }
}
