// Page controller: owns the live showcase match and the UI. Training itself
// happens in a Worker; here we just replay the newest weights it sends us.

import { RNG } from './rng.js';
import { Brain } from './nn.js';
import { TagEnv, OBS_DIM, BRANCHES, ROOMS, RUNNER, TAGGER } from './env.js';
import { playStep } from './trainer.js';
import { ArenaRenderer } from './render.js';
import { Curve } from './charts.js';

const $ = (id) => document.getElementById(id);
const COLORS = { [RUNNER]: '#e86a1c', [TAGGER]: '#2e86e0' };
const HIDDEN = 64;
const DECISIONS_PER_SEC = 15; // 60fps / 4-frame action repeat

// ---------------------------------------------------------------- showcase
const rng = new RNG(7);
const showcase = new TagEnv(new RNG(4242), 0);
showcase.reset();
const brains = [
  new Brain(OBS_DIM, HIDDEN, BRANCHES, new RNG(11), 1),
  new Brain(OBS_DIM, HIDDEN, BRANCHES, new RNG(12), 1),
];
const obsBuf = new Float32Array(OBS_DIM);
let prevPose = null;
let tagFlash = 0;
let holdTimer = 0;
let score = { albert: 0, kai: 0 };

const renderer = new ArenaRenderer($('arena'));

const tagCurve = new Curve($('chart-tag'), {
  color: COLORS[TAGGER],
  yMin: 0,
  yMax: 1,
  yTicks: [0, 0.5, 1],
  format: (v) => `${Math.round(v * 100)}%`,
});
const survCurve = new Curve($('chart-surv'), {
  color: COLORS[RUNNER],
  yMin: 0,
  yMax: 10,
  yTicks: [0, 5, 10],
  format: (v) => `${v.toFixed(1)}s`,
});

// ---------------------------------------------------------------- controls
let speed = 1;
let greedy = false;
let showRays = false;
let training = true;
let roomIndex = 0;
let swapEvery = 8;

const roomBox = $('room-buttons');
ROOMS.forEach((r, i) => {
  const b = document.createElement('button');
  b.textContent = r.name.replace(/^Room \d+ — /, '');
  b.title = r.name;
  b.setAttribute('aria-pressed', String(i === 0));
  b.onclick = () => selectRoom(i);
  roomBox.appendChild(b);
});

function selectRoom(i) {
  roomIndex = i;
  [...roomBox.children].forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
  showcase.setRoom(i);
  newEpisode();
  worker.postMessage({ cmd: 'room', roomIndex: i });
}

$('speed').onchange = (e) => (speed = Number(e.target.value));
$('greedy').onchange = (e) => (greedy = e.target.checked);
$('show-rays').onchange = (e) => (showRays = e.target.checked);

$('play').onclick = () => {
  training = !training;
  worker.postMessage({ cmd: training ? 'start' : 'pause' });
  $('play').textContent = training ? 'Pause training' : 'Resume training';
};

$('reset').onclick = () => {
  if (!confirm('Wipe both brains and start training from scratch?')) return;
  worker.postMessage({ cmd: 'reset', roomIndex });
  tagCurve.clear();
  survCurve.clear();
  score = { albert: 0, kai: 0 };
  newEpisode();
};

$('save').onclick = () => worker.postMessage({ cmd: 'save' });

$('load').onclick = () => $('file').click();
$('file').onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const data = JSON.parse(await f.text());
    worker.postMessage({ cmd: 'load', data });
  } catch (err) {
    alert(`Could not read that file: ${err.message}`);
  }
  e.target.value = '';
};

// ---------------------------------------------------------------- worker
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'update') {
    brains[RUNNER].setFlat(msg.runner);
    brains[TAGGER].setFlat(msg.tagger);
    applyStats(msg.stats);
  } else if (msg.type === 'saved') {
    downloadJSON(msg.data);
  } else if (msg.type === 'ready') {
    swapEvery = msg.config.swapEvery || swapEvery;
    applyStats(msg.stats);
  } else if (msg.type === 'error') {
    console.error(msg.message);
    alert(msg.message);
  }
};

worker.postMessage({ cmd: 'start' });

function applyStats(s) {
  if (!s) return;
  $('s-updates').textContent = s.updates.toLocaleString();
  $('s-hours').textContent = formatSimTime(s.frames / 60);
  if (s.episodes > 20) {
    $('s-tagrate').textContent = `${Math.round(s.tagRate * 100)}%`;
    $('s-survival').textContent = `${s.meanSurvival.toFixed(1)}s`;
    if (typeof s.entropy === 'number') {
      tagCurve.push(s.updates, s.tagRate);
      survCurve.push(s.updates, s.meanSurvival);
      tagCurve.draw();
      survCurve.draw();
    }
  }
  const who = s.trainee === TAGGER ? 'Kai' : 'Albert';
  const el = $('trainee');
  el.textContent = who;
  el.className = `who ${s.trainee === TAGGER ? 'tagger' : 'runner'}`;
  $('swap-bar').style.width = `${((s.updates % swapEvery) / swapEvery) * 100}%`;
  $('swap-bar').style.background = COLORS[s.trainee];
}

function formatSimTime(seconds) {
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

function downloadJSON(data) {
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `ai-tag-brains-u${data.updates}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------- prob bars
const probBars = {};
for (const role of [RUNNER, TAGGER]) {
  for (const [branch, n] of [
    ['move', 3],
    ['turn', 3],
    ['jump', 2],
    ['grab', 2],
  ]) {
    const host = $(`p-${branch}-${role}`);
    const bars = [];
    for (let i = 0; i < n; i++) {
      const el = document.createElement('i');
      el.style.setProperty('--c', COLORS[role]);
      el.title = {
        move: ['hold', 'forward', 'back'],
        turn: ['straight', 'left', 'right'],
        jump: ['stay down', 'jump'],
        grab: ['let go', 'grab'],
      }[branch][i];
      host.appendChild(el);
      bars.push(el);
    }
    probBars[`${branch}-${role}`] = bars;
  }
}

function updateProbBars() {
  for (const role of [RUNNER, TAGGER]) {
    const b = brains[role];
    ['move', 'turn', 'jump', 'grab'].forEach((branch, k) => {
      const bars = probBars[`${branch}-${role}`];
      for (let i = 0; i < bars.length; i++) {
        bars[i].style.setProperty('--p', `${(b.probs[k][i] * 100).toFixed(0)}%`);
      }
    });
  }
}

// ---------------------------------------------------------------- game loop
function newEpisode() {
  showcase.reset();
  prevPose = snapshot();
  tagFlash = 0;
  holdTimer = 0;
}

function snapshot() {
  return showcase.agents.map((a) => ({ x: a.x, y: a.y, z: a.z, th: a.th }));
}

let acc = 0;
let last = performance.now();
newEpisode();

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  if (holdTimer > 0) {
    holdTimer -= dt;
    tagFlash = Math.max(0, tagFlash - dt * 1.4);
    if (holdTimer <= 0) newEpisode();
  } else {
    acc += dt * DECISIONS_PER_SEC * speed;
    let steps = 0;
    while (acc >= 1 && steps < 8) {
      acc -= 1;
      steps++;
      prevPose = snapshot();
      const res = playStep(showcase, brains, obsBuf, rng, greedy);
      if (res.done) {
        if (res.tagged) {
          score.kai++;
          tagFlash = 1;
        } else {
          score.albert++;
        }
        holdTimer = res.tagged ? 0.85 : 0.5;
        acc = 0;
        break;
      }
    }
    updateProbBars();
  }

  renderer.draw(showcase, prevPose, Math.min(1, acc), {
    showRays,
    tagFlash,
    scoreText: `escapes ${score.albert} · tags ${score.kai}`,
  });

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

window.addEventListener('resize', () => {
  tagCurve.draw();
  survCurve.draw();
});
tagCurve.draw();
survCurve.draw();
