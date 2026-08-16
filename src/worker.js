// Training runs here so the arena keeps rendering at 60fps no matter how
// heavy an update gets. The worker owns the trainer; the page only ever
// receives stats and a copy of the current weights.

import { SelfPlayTrainer, DEFAULTS } from './trainer.js';

let trainer = new SelfPlayTrainer();
let running = false;
let scheduled = false;

function post(stats) {
  self.postMessage({
    type: 'update',
    stats,
    runner: trainer.brains[0].getFlat(),
    tagger: trainer.brains[1].getFlat(),
  });
}

function tick() {
  scheduled = false;
  if (!running) return;
  try {
    post(trainer.runUpdate());
  } catch (err) {
    running = false;
    self.postMessage({ type: 'error', message: String((err && err.message) || err) });
    return;
  }
  schedule();
}

function schedule() {
  if (running && !scheduled) {
    scheduled = true;
    // setTimeout rather than a tight loop so incoming commands are handled
    // between updates.
    setTimeout(tick, 0);
  }
}

self.onmessage = (e) => {
  const msg = e.data;
  switch (msg.cmd) {
    case 'start':
      running = true;
      schedule();
      break;
    case 'pause':
      running = false;
      break;
    case 'reset':
      trainer = new SelfPlayTrainer({ roomIndex: msg.roomIndex ?? trainer.cfg.roomIndex });
      post(trainer.stats());
      break;
    case 'room':
      trainer.setRoom(msg.roomIndex);
      post(trainer.stats());
      break;
    case 'save':
      self.postMessage({ type: 'saved', data: trainer.serialize() });
      break;
    case 'load':
      try {
        trainer.load(msg.data);
        post(trainer.stats());
      } catch (err) {
        self.postMessage({ type: 'error', message: String((err && err.message) || err) });
      }
      break;
  }
};

self.postMessage({ type: 'ready', config: DEFAULTS, stats: trainer.stats() });
post(trainer.stats());
