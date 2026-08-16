// PPO with self-play, mirroring the training setup described for Albert & Kai:
//
//  * only one agent is trained at a time; the trainee swaps every N updates
//  * the opponent is drawn from a pool of the 10 most recent frozen brains,
//    re-drawn whenever an environment resets, so the trainee has to beat a
//    spread of past strategies rather than one exploitable snapshot
//  * many environments run simultaneously (200 in the video)

import { RNG } from './rng.js';
import { Brain, sampleRow, argmaxRow } from './nn.js';
import { TagEnv, OBS_DIM, BRANCHES, RUNNER, TAGGER, EPISODE_FRAMES, ROOMS } from './env.js';

export const DEFAULTS = {
  numEnvs: 64,
  rolloutSteps: 64,
  hidden: 64,
  epochs: 4,
  minibatch: 512,
  gamma: 0.995,
  lam: 0.95,
  clip: 0.2,
  lr: 3e-4,
  vfCoef: 0.5,
  entStart: 0.02,
  entEnd: 0.004,
  entDecayUpdates: 400,
  maxGradNorm: 0.5,
  poolSize: 10,
  // Half the pool holds the most recent snapshots, half holds a uniform
  // sample of the whole run. Recent-only pools go unbeatable once one side
  // solves the game, and the loser stops getting any gradient at all.
  leaguePool: true,
  // Updates each role trains before handing over. An array is [runner, tagger]
  // so the harder role can be given more time.
  swapEvery: 8,
  snapshotEvery: 4, // updates between pushing a frozen copy into the pool
  roomIndex: 0,
  shaping: 0.02,
  seed: 20260816,
};

/** Sample one multi-discrete action for row `i` of the brain's last forward. */
function sampleAction(brain, i, rng, out) {
  let logp = 0;
  for (let k = 0; k < brain.branches.length; k++) {
    const n = brain.branches[k];
    const off = i * n;
    const a = sampleRow(brain.probs[k], off, n, rng);
    out[k] = a;
    logp += Math.log(Math.max(brain.probs[k][off + a], 1e-9));
  }
  return logp;
}

function greedyAction(brain, i, out) {
  for (let k = 0; k < brain.branches.length; k++) {
    const n = brain.branches[k];
    out[k] = argmaxRow(brain.probs[k], i * n, n);
  }
}

export class SelfPlayTrainer {
  constructor(opts = {}) {
    this.cfg = { ...DEFAULTS, ...opts };
    const cfg = this.cfg;
    this.rng = new RNG(cfg.seed);

    const maxBatch = Math.max(cfg.minibatch, cfg.numEnvs, 64);
    this.brains = [
      new Brain(OBS_DIM, cfg.hidden, BRANCHES, new RNG(cfg.seed + 1), maxBatch),
      new Brain(OBS_DIM, cfg.hidden, BRANCHES, new RNG(cfg.seed + 2), maxBatch),
    ];
    // Frozen opponents. Each pool entry is a forward-only Brain.
    this.pool = [[], []];
    for (let role = 0; role < 2; role++) {
      for (let i = 0; i < cfg.poolSize; i++) {
        const b = new Brain(OBS_DIM, cfg.hidden, BRANCHES, new RNG(cfg.seed + 99 + i), cfg.numEnvs);
        this.brains[role].cloneInto(b);
        this.pool[role].push(b);
      }
    }

    this.envs = [];
    for (let i = 0; i < cfg.numEnvs; i++) {
      const e = new TagEnv(new RNG(cfg.seed + 1000 + i), cfg.roomIndex, { shaping: cfg.shaping });
      e.reset();
      e.opponentId = this.rng.int(cfg.poolSize);
      this.envs.push(e);
    }

    this.trainee = TAGGER; // Kai gets the first turn, as in the video
    this.sinceSwap = 0;
    this.snapCount = [0, 0];
    this.updates = 0;
    this.envSteps = 0;
    this.frames = 0;

    const T = cfg.rolloutSteps;
    const N = cfg.numEnvs;
    this.nBranch = BRANCHES.length;
    this.bObs = new Float32Array(T * N * OBS_DIM);
    this.bAct = new Int32Array(T * N * this.nBranch);
    this.bLogp = new Float32Array(T * N);
    this.bVal = new Float32Array(T * N);
    this.bRew = new Float32Array(T * N);
    this.bDone = new Uint8Array(T * N);
    this.bTruncV = new Float32Array(T * N);
    this.bAdv = new Float32Array(T * N);
    this.bRet = new Float32Array(T * N);

    this.stepObs = new Float32Array(N * OBS_DIM);
    this.oppObs = new Float32Array(N * OBS_DIM);
    this.mbObs = new Float32Array(maxBatch * OBS_DIM);
    this.act = [new Int32Array(this.nBranch), new Int32Array(this.nBranch)];
    this.idx = new Int32Array(T * N);
    this.groupEnvs = new Int32Array(N);

    // Rolling episode stats (over the last few hundred finished episodes)
    this.hist = [];
    this.histMax = 400;
    this.curve = []; // one point per update, for the charts
  }

  get entCoef() {
    const c = this.cfg;
    const t = Math.min(1, this.updates / c.entDecayUpdates);
    return c.entStart + (c.entEnd - c.entStart) * t;
  }

  setRoom(i) {
    this.cfg.roomIndex = i;
    for (const e of this.envs) {
      e.setRoom(i);
      e.reset();
      e.opponentId = this.rng.int(this.cfg.poolSize);
    }
  }

  _resetEnv(e) {
    e.reset();
    e.opponentId = this.rng.int(this.cfg.poolSize);
  }

  collectRollout() {
    const cfg = this.cfg;
    const N = cfg.numEnvs;
    const T = cfg.rolloutSteps;
    const trainee = this.trainee;
    const opp = 1 - trainee;
    const brain = this.brains[trainee];
    const pool = this.pool[opp];

    for (let t = 0; t < T; t++) {
      // --- trainee: one batched forward over every environment
      for (let i = 0; i < N; i++) this.envs[i].observe(trainee, this.stepObs, i * OBS_DIM);
      brain.forward(this.stepObs, N);
      const base = t * N;
      this.bObs.set(this.stepObs.subarray(0, N * OBS_DIM), base * OBS_DIM);
      for (let i = 0; i < N; i++) {
        const logp = sampleAction(brain, i, this.rng, this.act[trainee]);
        const ao = (base + i) * this.nBranch;
        for (let k = 0; k < this.nBranch; k++) this.bAct[ao + k] = this.act[trainee][k];
        this.bLogp[base + i] = logp;
        this.bVal[base + i] = brain.vOut[i];
        // stash the sampled action per env
        this.envs[i]._pendingTrainee = Array.from(this.act[trainee]);
      }

      // --- opponents: one batched forward per distinct pooled brain
      for (let p = 0; p < pool.length; p++) {
        let m = 0;
        for (let i = 0; i < N; i++) if (this.envs[i].opponentId === p) this.groupEnvs[m++] = i;
        if (m === 0) continue;
        for (let g = 0; g < m; g++) {
          this.envs[this.groupEnvs[g]].observe(opp, this.oppObs, g * OBS_DIM);
        }
        const pb = pool[p];
        pb.forward(this.oppObs, m);
        for (let g = 0; g < m; g++) {
          const e = this.envs[this.groupEnvs[g]];
          sampleAction(pb, g, this.rng, this.act[opp]);
          e._pendingOpp = Array.from(this.act[opp]);
        }
      }

      // --- step every environment
      for (let i = 0; i < N; i++) {
        const e = this.envs[i];
        const actions = [];
        actions[trainee] = e._pendingTrainee;
        actions[opp] = e._pendingOpp;
        const res = e.step(actions);
        this.bRew[base + i] = res.rewards[trainee];
        this.bDone[base + i] = res.done ? 1 : 0;
        this.bTruncV[base + i] = 0;
        this.envSteps++;
        this.frames += 4;

        if (res.done) {
          this.hist.push({ tagged: res.tagged, frames: e.frame });
          if (this.hist.length > this.histMax) this.hist.shift();
        }
      }

      // --- bootstrap value for time-limit truncations (not for real tags)
      let m = 0;
      for (let i = 0; i < N; i++) {
        const e = this.envs[i];
        if (this.bDone[base + i] && !e.tagged) {
          e.observe(trainee, this.oppObs, m * OBS_DIM);
          this.groupEnvs[m++] = i;
        }
      }
      if (m > 0) {
        brain.forward(this.oppObs, m);
        for (let g = 0; g < m; g++) this.bTruncV[base + this.groupEnvs[g]] = brain.vOut[g];
      }

      for (let i = 0; i < N; i++) if (this.bDone[base + i]) this._resetEnv(this.envs[i]);
    }

    // Value of the state we stopped on, for envs still mid-episode.
    for (let i = 0; i < N; i++) this.envs[i].observe(trainee, this.stepObs, i * OBS_DIM);
    brain.forward(this.stepObs, N);
    this.lastVal = Float32Array.from(brain.vOut.subarray(0, N));
  }

  computeGAE() {
    const { numEnvs: N, rolloutSteps: T, gamma, lam } = this.cfg;
    for (let i = 0; i < N; i++) {
      let nextAdv = 0;
      for (let t = T - 1; t >= 0; t--) {
        const k = t * N + i;
        let nextV;
        if (this.bDone[k]) {
          nextV = this.bTruncV[k]; // 0 on a real tag, V(s_T) on a timeout
          nextAdv = 0;
        } else {
          nextV = t === T - 1 ? this.lastVal[i] : this.bVal[(t + 1) * N + i];
        }
        const delta = this.bRew[k] + gamma * nextV - this.bVal[k];
        nextAdv = delta + gamma * lam * nextAdv;
        this.bAdv[k] = nextAdv;
        this.bRet[k] = nextAdv + this.bVal[k];
      }
    }
  }

  optimise() {
    const cfg = this.cfg;
    const brain = this.brains[this.trainee];
    const total = cfg.rolloutSteps * cfg.numEnvs;
    const entCoef = this.entCoef;

    // Normalise advantages over the whole batch.
    let mean = 0;
    for (let i = 0; i < total; i++) mean += this.bAdv[i];
    mean /= total;
    let varAcc = 0;
    for (let i = 0; i < total; i++) {
      const d = this.bAdv[i] - mean;
      varAcc += d * d;
    }
    const std = Math.sqrt(varAcc / total) + 1e-8;

    for (let i = 0; i < total; i++) this.idx[i] = i;

    let sumEnt = 0;
    let sumKL = 0;
    let sumVL = 0;
    let nMB = 0;

    for (let ep = 0; ep < cfg.epochs; ep++) {
      // Fisher-Yates shuffle
      for (let i = total - 1; i > 0; i--) {
        const j = this.rng.int(i + 1);
        const tmp = this.idx[i];
        this.idx[i] = this.idx[j];
        this.idx[j] = tmp;
      }

      for (let start = 0; start < total; start += cfg.minibatch) {
        const mb = Math.min(cfg.minibatch, total - start);
        for (let i = 0; i < mb; i++) {
          const src = this.idx[start + i] * OBS_DIM;
          this.mbObs.set(this.bObs.subarray(src, src + OBS_DIM), i * OBS_DIM);
        }
        brain.forward(this.mbObs, mb);
        brain.zeroGrad();

        const inv = 1 / mb;
        for (let i = 0; i < mb; i++) {
          const s = this.idx[start + i];
          const adv = (this.bAdv[s] - mean) / std;

          let lp = 0;
          for (let k = 0; k < brain.branches.length; k++) {
            const n = brain.branches[k];
            const a = this.bAct[s * this.nBranch + k];
            lp += Math.log(Math.max(brain.probs[k][i * n + a], 1e-9));
          }
          const ratio = Math.exp(lp - this.bLogp[s]);
          const clipped = Math.min(Math.max(ratio, 1 - cfg.clip), 1 + cfg.clip);
          const useUnclipped = ratio * adv <= clipped * adv;
          const dLdlp = useUnclipped ? -adv * ratio * inv : 0;

          sumKL += this.bLogp[s] - lp;

          for (let k = 0; k < brain.branches.length; k++) {
            const n = brain.branches[k];
            const off = i * n;
            const p = brain.probs[k];
            const a = this.bAct[s * this.nBranch + k];
            let H = 0;
            for (let j = 0; j < n; j++) {
              const pj = Math.max(p[off + j], 1e-9);
              H -= pj * Math.log(pj);
            }
            sumEnt += H;
            for (let j = 0; j < n; j++) {
              const pj = Math.max(p[off + j], 1e-9);
              const onehot = j === a ? 1 : 0;
              // policy-gradient term + entropy bonus term
              brain.gLogits[k][off + j] =
                dLdlp * (onehot - pj) + entCoef * pj * (Math.log(pj) + H) * inv;
            }
          }

          const v = brain.vOut[i];
          const d = v - this.bRet[s];
          sumVL += 0.5 * d * d;
          brain.gV[i] = cfg.vfCoef * d * inv;
        }

        brain.backward(this.mbObs, mb);
        brain.step(cfg.lr, cfg.maxGradNorm);
        nMB++;
      }
    }

    const nSamples = nMB * cfg.minibatch;
    return {
      entropy: sumEnt / (nSamples * brain.branches.length),
      kl: sumKL / nSamples,
      valueLoss: sumVL / nSamples,
    };
  }

  /** One full iteration: collect, GAE, optimise, then handle the self-play bookkeeping. */
  runUpdate() {
    const trainee = this.trainee;
    this.collectRollout();
    this.computeGAE();
    const losses = this.optimise();
    this.updates++;

    if (this.updates % this.cfg.snapshotEvery === 0) this._snapshot(trainee);
    this.sinceSwap++;
    const cfg = this.cfg;
    const need = Array.isArray(cfg.swapEvery) ? cfg.swapEvery[this.trainee] : cfg.swapEvery;
    if (this.sinceSwap >= need) {
      this.trainee = 1 - this.trainee;
      this.sinceSwap = 0;
    }

    return { ...this.stats(), ...losses, trainedRole: trainee };
  }

  _snapshot(role) {
    const pool = this.pool[role];
    const n = ++this.snapCount[role];

    if (!this.cfg.leaguePool) {
      // Rotate: oldest entry is recycled to hold the newest weights.
      const b = pool.shift();
      this.brains[role].cloneInto(b);
      pool.push(b);
      return;
    }

    // Slots [0, half) are a FIFO of recent snapshots; slots [half, size) hold a
    // reservoir sample over every snapshot ever taken, so early, beatable
    // opponents stay in circulation and the trainee keeps seeing rounds it can
    // actually win. Without that the loser's advantages go flat and it stops
    // learning entirely.
    const half = Math.max(1, Math.floor(pool.length / 2));
    const recent = pool.shift();
    this.brains[role].cloneInto(recent);
    pool.splice(half - 1, 0, recent);

    const histSlots = pool.length - half;
    if (histSlots <= 0) return;
    if (n <= histSlots) {
      this.brains[role].cloneInto(pool[half + n - 1]);
    } else if (this.rng.next() < histSlots / n) {
      this.brains[role].cloneInto(pool[half + this.rng.int(histSlots)]);
    }
  }

  stats() {
    let tags = 0;
    let frames = 0;
    for (const h of this.hist) {
      if (h.tagged) tags++;
      frames += h.frames;
    }
    const n = Math.max(1, this.hist.length);
    return {
      updates: this.updates,
      envSteps: this.envSteps,
      frames: this.frames,
      episodes: this.hist.length,
      tagRate: tags / n,
      meanSurvival: frames / n / 60, // seconds
      trainee: this.trainee,
      entCoef: this.entCoef,
      roomIndex: this.cfg.roomIndex,
    };
  }

  /** Serialise both brains + progress so a run can be saved and reloaded. */
  serialize() {
    return {
      version: 1,
      hidden: this.cfg.hidden,
      obsDim: OBS_DIM,
      updates: this.updates,
      envSteps: this.envSteps,
      frames: this.frames,
      roomIndex: this.cfg.roomIndex,
      trainee: this.trainee,
      runner: Array.from(this.brains[RUNNER].getFlat()),
      tagger: Array.from(this.brains[TAGGER].getFlat()),
    };
  }

  load(data) {
    if (!data || data.obsDim !== OBS_DIM || data.hidden !== this.cfg.hidden) {
      throw new Error('Saved brains do not match the current network shape.');
    }
    this.brains[RUNNER].setFlat(Float32Array.from(data.runner));
    this.brains[TAGGER].setFlat(Float32Array.from(data.tagger));
    for (let role = 0; role < 2; role++) {
      for (const b of this.pool[role]) this.brains[role].cloneInto(b);
    }
    this.updates = data.updates || 0;
    this.envSteps = data.envSteps || 0;
    this.frames = data.frames || 0;
    this.trainee = data.trainee ?? TAGGER;
    this.hist.length = 0;
  }
}

/**
 * Drive a single environment with two brains — used by the live showcase.
 * `greedy` picks the most likely action instead of sampling.
 */
export function playStep(env, brains, obsBuf, rng, greedy = false) {
  const actions = [];
  for (let role = 0; role < 2; role++) {
    env.observe(role, obsBuf, 0);
    const b = brains[role];
    b.forward(obsBuf, 1);
    const a = new Int32Array(BRANCHES.length);
    if (greedy) greedyAction(b, 0, a);
    else sampleAction(b, 0, rng, a);
    actions[role] = a;
  }
  return env.step(actions);
}

export { OBS_DIM, BRANCHES, ROOMS, EPISODE_FRAMES, RUNNER, TAGGER };
