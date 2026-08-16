// Headless training check. Run: node test/train.mjs [updates] [room]
//
// Self-play tag-rate alone is ambiguous (it can stay flat while BOTH agents
// improve), so we also benchmark each brain against a frozen copy of the
// random policy it started from. Those two curves are what should move.

import { RNG } from '../src/rng.js';
import { Brain } from '../src/nn.js';
import { TagEnv, OBS_DIM, BRANCHES, RUNNER, TAGGER } from '../src/env.js';
import { SelfPlayTrainer, playStep } from '../src/trainer.js';

const UPDATES = Number(process.argv[2] || 60);
const ROOM = Number(process.argv[3] || 0);

const trainer = new SelfPlayTrainer({ roomIndex: ROOM });

// Freeze the untrained brains as a fixed yardstick.
const baseline = [
  new Brain(OBS_DIM, trainer.cfg.hidden, BRANCHES, new RNG(1), 1),
  new Brain(OBS_DIM, trainer.cfg.hidden, BRANCHES, new RNG(2), 1),
];
trainer.brains[RUNNER].cloneInto(baseline[RUNNER]);
trainer.brains[TAGGER].cloneInto(baseline[TAGGER]);

const evalBrains = [
  new Brain(OBS_DIM, trainer.cfg.hidden, BRANCHES, new RNG(3), 1),
  new Brain(OBS_DIM, trainer.cfg.hidden, BRANCHES, new RNG(4), 1),
];

// Once Kai is competent we freeze a copy of it, so Albert's survival time
// against a *fixed, non-trivial* chaser becomes a meaningful absolute curve.
// (Survival against the random baseline is always 10s and tells us nothing.)
const refKai = new Brain(OBS_DIM, trainer.cfg.hidden, BRANCHES, new RNG(5), 1);
let refKaiSet = false;

function evaluate(testRole, episodes = 60, useRef = false) {
  // testRole plays its current brain; the other side uses a frozen opponent.
  trainer.brains[testRole].cloneInto(evalBrains[testRole]);
  if (useRef && refKaiSet) refKai.cloneInto(evalBrains[1 - testRole]);
  else baseline[1 - testRole].cloneInto(evalBrains[1 - testRole]);
  const rng = new RNG(777);
  const env = new TagEnv(new RNG(999), ROOM);
  const obs = new Float32Array(OBS_DIM);
  let tags = 0;
  let frames = 0;
  for (let e = 0; e < episodes; e++) {
    env.reset();
    let res;
    do {
      res = playStep(env, evalBrains, obs, rng, false);
    } while (!res.done);
    if (res.tagged) tags++;
    frames += env.frame;
  }
  return { tagRate: tags / episodes, survival: frames / episodes / 60 };
}

const pct = (v) => (v * 100).toFixed(0).padStart(3) + '%';

console.log(
  `obs=${OBS_DIM}  params/brain=${trainer.brains[0].numParams()}  ` +
    `envs=${trainer.cfg.numEnvs}  rollout=${trainer.cfg.rolloutSteps}  room="${ROOM}"`,
);
console.log(
  '\n  upd | trained |  selfplay |  survive | KAI vs random Albert | ALBERT vs random Kai |  ent |    s',
);
console.log(
  '  ----+---------+-----------+----------+----------------------+----------------------+------+-----',
);

const t0 = Date.now();
let lastLog = t0;
for (let u = 1; u <= UPDATES; u++) {
  const s = trainer.runUpdate();
  if (u % 10 === 0 || u === 1 || u === UPDATES) {
    const kai = evaluate(TAGGER);
    if (!refKaiSet && kai.tagRate >= 0.9) {
      trainer.brains[TAGGER].cloneInto(refKai);
      refKaiSet = true;
      console.log(`      ...froze reference Kai at update ${u} (${pct(kai.tagRate)} vs random)`);
    }
    const albert = evaluate(RUNNER, 60, true);
    const now = Date.now();
    console.log(
      `  ${String(u).padStart(3)} | ${s.trainedRole === TAGGER ? '  Kai  ' : 'Albert '} |` +
        ` tag ${pct(s.tagRate)} |  ${s.meanSurvival.toFixed(2)}s |` +
        `      tag ${pct(kai.tagRate)}         |` +
        `   survive ${albert.survival.toFixed(2)}s${refKaiSet ? '*' : ' '}   |` +
        ` ${s.entropy.toFixed(2)} | ${((now - lastLog) / 1000).toFixed(1)}`,
    );
    lastLog = now;
  }
}

const secs = (Date.now() - t0) / 1000;
console.log(
  `\ndone: ${UPDATES} updates in ${secs.toFixed(1)}s ` +
    `(${(secs / UPDATES).toFixed(2)}s/update, ${trainer.envSteps.toLocaleString()} agent decisions, ` +
    `${(trainer.frames / 60 / 3600).toFixed(1)}h of simulated tag)`,
);
