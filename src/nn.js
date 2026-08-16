// Minimal dense-layer MLP with hand-written backprop and Adam.
// Everything is flat Float32Array so it stays fast in plain JS and is
// trivial to serialise across a Worker boundary.

class Linear {
  constructor(nIn, nOut, rng, gain = 1) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.W = new Float32Array(nOut * nIn);
    this.b = new Float32Array(nOut);
    this.gW = new Float32Array(nOut * nIn);
    this.gb = new Float32Array(nOut);
    // Adam moments
    this.mW = new Float32Array(nOut * nIn);
    this.vW = new Float32Array(nOut * nIn);
    this.mb = new Float32Array(nOut);
    this.vb = new Float32Array(nOut);

    const std = gain / Math.sqrt(nIn);
    for (let i = 0; i < this.W.length; i++) this.W[i] = rng.normal() * std;
  }

  // x: [batch, nIn] -> out: [batch, nOut]
  forward(x, batch, out) {
    const { nIn, nOut, W, b } = this;
    for (let n = 0; n < batch; n++) {
      const xo = n * nIn;
      const oo = n * nOut;
      for (let o = 0; o < nOut; o++) {
        let s = b[o];
        const wo = o * nIn;
        for (let i = 0; i < nIn; i++) s += W[wo + i] * x[xo + i];
        out[oo + o] = s;
      }
    }
  }

  // Accumulates gW/gb; writes dL/dx into gIn when provided.
  backward(x, gOut, batch, gIn) {
    const { nIn, nOut, W, gW, gb } = this;
    if (gIn) gIn.fill(0, 0, batch * nIn);
    for (let n = 0; n < batch; n++) {
      const xo = n * nIn;
      const oo = n * nOut;
      for (let o = 0; o < nOut; o++) {
        const g = gOut[oo + o];
        if (g === 0) continue;
        const wo = o * nIn;
        gb[o] += g;
        for (let i = 0; i < nIn; i++) gW[wo + i] += g * x[xo + i];
        if (gIn) for (let i = 0; i < nIn; i++) gIn[xo + i] += g * W[wo + i];
      }
    }
  }

  zeroGrad() {
    this.gW.fill(0);
    this.gb.fill(0);
  }
}

function adamApply(p, g, m, v, lr, b1, b2, eps, bc1, bc2, scale) {
  for (let i = 0; i < p.length; i++) {
    const gi = g[i] * scale;
    m[i] = b1 * m[i] + (1 - b1) * gi;
    v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    const mh = m[i] / bc1;
    const vh = v[i] / bc2;
    p[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
  }
}

/**
 * Actor-critic brain.
 *
 * obs -> tanh(H) -> tanh(H) -> { one logit head per action branch, value }
 *
 * Matches the video's description: an input layer, two hidden layers and an
 * output layer, producing one number per action branch (move / turn).
 */
export class Brain {
  constructor(obsDim, hidden, branches, rng, maxBatch = 1024) {
    this.obsDim = obsDim;
    this.hidden = hidden;
    this.branches = branches.slice();
    this.maxBatch = maxBatch;

    this.l1 = new Linear(obsDim, hidden, rng, 1.0);
    this.l2 = new Linear(hidden, hidden, rng, 1.0);
    // Small init on the policy heads => near-uniform initial policy.
    this.heads = branches.map((n) => new Linear(hidden, n, rng, 0.01));
    this.value = new Linear(hidden, 1, rng, 1.0);
    this.layers = [this.l1, this.l2, ...this.heads, this.value];

    this.t = 0; // Adam step counter

    // Forward/backward scratch
    this.z1 = new Float32Array(maxBatch * hidden);
    this.a1 = new Float32Array(maxBatch * hidden);
    this.z2 = new Float32Array(maxBatch * hidden);
    this.a2 = new Float32Array(maxBatch * hidden);
    this.logits = branches.map((n) => new Float32Array(maxBatch * n));
    this.probs = branches.map((n) => new Float32Array(maxBatch * n));
    this.vOut = new Float32Array(maxBatch);
    this.gLogits = branches.map((n) => new Float32Array(maxBatch * n));
    this.gV = new Float32Array(maxBatch);
    this.gA2 = new Float32Array(maxBatch * hidden);
    this.gZ2 = new Float32Array(maxBatch * hidden);
    this.gA1 = new Float32Array(maxBatch * hidden);
    this.gZ1 = new Float32Array(maxBatch * hidden);
    this.gTmp = new Float32Array(maxBatch * hidden);
  }

  /** Forward pass. Fills this.probs / this.vOut. */
  forward(obs, batch) {
    const H = this.hidden;
    this.l1.forward(obs, batch, this.z1);
    for (let i = 0; i < batch * H; i++) this.a1[i] = Math.tanh(this.z1[i]);
    this.l2.forward(this.a1, batch, this.z2);
    for (let i = 0; i < batch * H; i++) this.a2[i] = Math.tanh(this.z2[i]);

    for (let k = 0; k < this.heads.length; k++) {
      this.heads[k].forward(this.a2, batch, this.logits[k]);
      softmaxRows(this.logits[k], this.probs[k], batch, this.branches[k]);
    }
    this.value.forward(this.a2, batch, this.vOut);
  }

  zeroGrad() {
    for (const l of this.layers) l.zeroGrad();
  }

  /**
   * Backward from this.gLogits / this.gV (already filled by the loss) down
   * through the trunk. Requires a matching forward() with the same batch.
   */
  backward(obs, batch) {
    const H = this.hidden;
    this.gA2.fill(0, 0, batch * H);

    for (let k = 0; k < this.heads.length; k++) {
      this.heads[k].backward(this.a2, this.gLogits[k], batch, this.gTmp);
      for (let i = 0; i < batch * H; i++) this.gA2[i] += this.gTmp[i];
    }
    this.value.backward(this.a2, this.gV, batch, this.gTmp);
    for (let i = 0; i < batch * H; i++) this.gA2[i] += this.gTmp[i];

    for (let i = 0; i < batch * H; i++) {
      const a = this.a2[i];
      this.gZ2[i] = this.gA2[i] * (1 - a * a);
    }
    this.l2.backward(this.a1, this.gZ2, batch, this.gA1);
    for (let i = 0; i < batch * H; i++) {
      const a = this.a1[i];
      this.gZ1[i] = this.gA1[i] * (1 - a * a);
    }
    this.l1.backward(obs, this.gZ1, batch, null);
  }

  gradNorm() {
    let s = 0;
    for (const l of this.layers) {
      for (let i = 0; i < l.gW.length; i++) s += l.gW[i] * l.gW[i];
      for (let i = 0; i < l.gb.length; i++) s += l.gb[i] * l.gb[i];
    }
    return Math.sqrt(s);
  }

  step(lr, maxGradNorm = 0.5) {
    let scale = 1;
    if (maxGradNorm > 0) {
      const gn = this.gradNorm();
      if (gn > maxGradNorm) scale = maxGradNorm / (gn + 1e-8);
    }
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, this.t);
    const bc2 = 1 - Math.pow(b2, this.t);
    for (const l of this.layers) {
      adamApply(l.W, l.gW, l.mW, l.vW, lr, b1, b2, eps, bc1, bc2, scale);
      adamApply(l.b, l.gb, l.mb, l.vb, lr, b1, b2, eps, bc1, bc2, scale);
    }
  }

  numParams() {
    let n = 0;
    for (const l of this.layers) n += l.W.length + l.b.length;
    return n;
  }

  /** Flatten weights (no optimiser state) for cloning / posting to the UI. */
  getFlat(out) {
    const buf = out || new Float32Array(this.numParams());
    let o = 0;
    for (const l of this.layers) {
      buf.set(l.W, o);
      o += l.W.length;
      buf.set(l.b, o);
      o += l.b.length;
    }
    return buf;
  }

  setFlat(buf) {
    let o = 0;
    for (const l of this.layers) {
      l.W.set(buf.subarray(o, o + l.W.length));
      o += l.W.length;
      l.b.set(buf.subarray(o, o + l.b.length));
      o += l.b.length;
    }
  }

  cloneInto(other) {
    other.setFlat(this.getFlat());
  }
}

export function softmaxRows(logits, probs, batch, n) {
  for (let b = 0; b < batch; b++) {
    const o = b * n;
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (logits[o + i] > mx) mx = logits[o + i];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const e = Math.exp(logits[o + i] - mx);
      probs[o + i] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let i = 0; i < n; i++) probs[o + i] *= inv;
  }
}

export function sampleRow(probs, offset, n, rng) {
  let r = rng.next();
  for (let i = 0; i < n; i++) {
    r -= probs[offset + i];
    if (r <= 0) return i;
  }
  return n - 1;
}

export function argmaxRow(probs, offset, n) {
  let best = 0;
  let bv = -Infinity;
  for (let i = 0; i < n; i++) {
    if (probs[offset + i] > bv) {
      bv = probs[offset + i];
      best = i;
    }
  }
  return best;
}
