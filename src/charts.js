// Single-series training curves.
//
// Two measures with different units (a share and a duration) get two charts
// rather than one dual-axis chart. Each chart carries one series, so identity
// comes from the title plus the always-visible current-value label rather than
// from colour alone; the colour matches the agent the curve is about.

const INK = '#e8ecf3';
const INK2 = '#a7b0c0';
const INK3 = '#6e778a';
const GRID = '#1d222c';
const SURFACE = '#12141a';

export class Curve {
  /**
   * @param canvas
   * @param opts { color, format(v), yMin, yMax, yTicks:[..], hoverLabel(x,v) }
   */
  constructor(canvas, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.opts = opts;
    this.data = []; // { x, y }
    this.hover = null;
    this.dpr = 1;

    canvas.addEventListener('pointermove', (e) => {
      const rect = canvas.getBoundingClientRect();
      this.hover = { px: e.clientX - rect.left, py: e.clientY - rect.top };
      this.draw();
    });
    canvas.addEventListener('pointerleave', () => {
      this.hover = null;
      this.draw();
    });
  }

  push(x, y) {
    this.data.push({ x, y });
    // Keep the series bounded; thin the oldest half when it gets long so the
    // shape of the whole run survives instead of scrolling away.
    if (this.data.length > 1200) {
      const keep = this.data.filter((_, i) => i % 2 === 0 || i > 600);
      this.data = keep;
    }
  }

  clear() {
    this.data = [];
    this.draw();
  }

  _layout() {
    const c = this.canvas;
    const rect = c.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    this.dpr = dpr;
    this.w = w;
    this.h = h;
    this.padL = 34 * dpr;
    this.padR = 46 * dpr; // reserved for the current-value label
    this.padT = 8 * dpr;
    this.padB = 18 * dpr;
    this.plotW = w - this.padL - this.padR;
    this.plotH = h - this.padT - this.padB;
  }

  draw() {
    this._layout();
    const { ctx, w, h, padL, padT, plotW, plotH, opts, dpr } = this;
    ctx.clearRect(0, 0, w, h);

    const n = this.data.length;
    const yMin = opts.yMin;
    const yMax = opts.yMax;
    const xMin = n ? this.data[0].x : 0;
    const xMax = n ? Math.max(this.data[n - 1].x, xMin + 1) : 1;

    const X = (x) => padL + ((x - xMin) / (xMax - xMin)) * plotW;
    const Y = (y) => padT + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    // recessive gridlines + y labels
    ctx.font = `${10 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of opts.yTicks) {
      const y = Y(t);
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      ctx.fillStyle = INK3;
      ctx.fillText(opts.format(t), padL - 6 * dpr, y);
    }

    if (n === 0) {
      ctx.fillStyle = INK3;
      ctx.textAlign = 'center';
      ctx.fillText('waiting for the first update…', padL + plotW / 2, padT + plotH / 2);
      return;
    }

    // x scale hint
    ctx.fillStyle = INK3;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(n > 1 ? `update ${xMin}` : 'update', padL, padT + plotH + 4 * dpr);
    ctx.textAlign = 'right';
    ctx.fillText(`${xMax}`, padL + plotW, padT + plotH + 4 * dpr);

    // the line — 2px, no marker per point
    ctx.strokeStyle = opts.color;
    ctx.lineWidth = 2 * dpr;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const p = this.data[i];
      const px = X(p.x);
      const py = Y(Math.max(yMin, Math.min(yMax, p.y)));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // one selective direct label: the value now
    const last = this.data[n - 1];
    const lx = X(last.x);
    const ly = Y(Math.max(yMin, Math.min(yMax, last.y)));
    ctx.fillStyle = SURFACE;
    ctx.beginPath();
    ctx.arc(lx, ly, 4.5 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = opts.color;
    ctx.beginPath();
    ctx.arc(lx, ly, 3 * dpr, 0, Math.PI * 2);
    ctx.fill();

    // The only number printed on the plot: where the curve is right now.
    // padR reserves room for it so it never sits on top of the line.
    ctx.font = `600 ${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK;
    ctx.fillText(opts.format(last.y), lx + 8 * dpr, ly);

    if (this.hover) this._drawHover(X, Y, xMin, xMax);
  }

  _drawHover(X, Y, xMin, xMax) {
    const { ctx, padL, padT, plotW, plotH, dpr, opts } = this;
    const hx = this.hover.px * dpr;
    if (hx < padL || hx > padL + plotW) return;
    const target = xMin + ((hx - padL) / plotW) * (xMax - xMin);

    let best = 0;
    let bd = Infinity;
    for (let i = 0; i < this.data.length; i++) {
      const d = Math.abs(this.data[i].x - target);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    const p = this.data[best];
    const px = X(p.x);
    const py = Y(Math.max(opts.yMin, Math.min(opts.yMax, p.y)));

    ctx.save();
    ctx.strokeStyle = 'rgba(167,176,192,0.4)';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.moveTo(px, padT);
    ctx.lineTo(px, padT + plotH);
    ctx.stroke();

    ctx.fillStyle = opts.color;
    ctx.beginPath();
    ctx.arc(px, py, 4 * dpr, 0, Math.PI * 2);
    ctx.fill();

    const text = `#${p.x}  ${opts.format(p.y)}`;
    ctx.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    const tw = ctx.measureText(text).width;
    const bw = tw + 14 * dpr;
    const bh = 20 * dpr;
    let bx = px + 8 * dpr;
    if (bx + bw > padL + plotW) bx = px - 8 * dpr - bw;
    const by = padT + 2 * dpr;

    ctx.fillStyle = 'rgba(10,12,17,0.94)';
    ctx.strokeStyle = '#333b4a';
    roundRectPath(ctx, bx, by, bw, bh, 5 * dpr);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = INK2;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, bx + 7 * dpr, by + bh / 2);
    ctx.restore();
  }
}

function roundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
