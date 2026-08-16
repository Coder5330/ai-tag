// Balance diagnostic. Trains a pair, then watches how they actually play.
//
//   node test/diagnose.mjs [updates] [room] [taggerSpeed] [roundSeconds]
//
// The question this answers: when Albert loses, is it because the physics are
// stacked against him, or because he never learned to use the jump and crates?
// Those need opposite fixes, so measure before tuning.

import { RNG } from '../src/rng.js';
import {
  TagEnv,
  SPECS,
  TAG,
  OBS_DIM,
  RUNNER,
  TAGGER,
  AGENT_R,
  FRAME_SKIP,
} from '../src/env.js';
import { SelfPlayTrainer, playStep } from '../src/trainer.js';

// --name=value flags, so a sweep can vary one knob at a time.
const flags = {};
for (const a of process.argv.slice(2)) {
  const m = /^--([\w-]+)=(.+)$/.exec(a);
  if (m) flags[m[1]] = m[2];
}
const UPDATES = Number(flags.updates || 240);
const ROOM = Number(flags.room || 0);
const ROUND_S = Number(flags.round || 10);
const NAME = flags.name || 'config';

// Agents hold a live reference to their spec, so mutating it retunes the sim.
if (flags['tagger-speed']) SPECS[TAGGER].maxSpeed = Number(flags['tagger-speed']);
if (flags['runner-jump']) SPECS[RUNNER].jumpV = Number(flags['runner-jump']);
if (flags['tagger-jump']) SPECS[TAGGER].jumpV = Number(flags['tagger-jump']);
if (flags['runner-air']) SPECS[RUNNER].air = Number(flags['runner-air']);
if (flags['tagger-turn']) SPECS[TAGGER].turnRate = Number(flags['tagger-turn']);
if (flags['runner-turn']) SPECS[RUNNER].turnRate = Number(flags['runner-turn']);
if (flags['jump-cd']) SPECS[RUNNER].jumpCd = SPECS[TAGGER].jumpCd = Number(flags['jump-cd']);
if (flags['tag-dist']) TAG.dist = Number(flags['tag-dist']);
if (flags['no-jump']) {
  // Disable jumping outright by making the impulse a no-op.
  SPECS[RUNNER].jumpV = 0;
  SPECS[TAGGER].jumpV = 0;
}

const trainerOpts = { roomIndex: ROOM };
if (flags['no-league']) trainerOpts.leaguePool = false;
if (flags['swap']) trainerOpts.swapEvery = flags['swap'].split(',').map(Number);
const trainer = new SelfPlayTrainer(trainerOpts);
const label =
  `${NAME}  [Kai ${SPECS[TAGGER].maxSpeed}u/s turn${SPECS[TAGGER].turnRate}` +
  ` (radius ${(SPECS[TAGGER].maxSpeed / SPECS[TAGGER].turnRate).toFixed(1)}u)` +
  ` · tagDist ${TAG.dist} · league ${trainerOpts.leaguePool !== false}` +
  ` · swap ${JSON.stringify(trainerOpts.swapEvery ?? 8)} · room ${ROOM}]`;

for (let u = 0; u < UPDATES; u++) trainer.runUpdate();

// ---- watch the trained pair play, and count what Albert actually does
const EPISODES = 300;
const roundFrames = Math.round(ROUND_S * 60);
const rng = new RNG(4321);
const env = new TagEnv(new RNG(8888), ROOM);
const obs = new Float32Array(OBS_DIM);

let tags = 0;
let frames = 0;
let decisions = 0;
let jumpDecisions = 0;
let airborneDecisions = 0;
let grabDecisions = 0;
let heldDecisions = 0;
// A "threatened" moment is Kai closing inside 6 units. Did Albert go up?
let threatened = 0;
let threatenedAirborne = 0;
// Episodes where Albert was off the ground at some point while threatened.
let episodesWithClutchJump = 0;
let escapesWithClutchJump = 0;

for (let e = 0; e < EPISODES; e++) {
  env.reset();
  let clutch = false;
  let res;
  do {
    const a = env.agents[RUNNER];
    const k = env.agents[TAGGER];
    const dist = Math.hypot(k.x - a.x, k.y - a.y);
    const near = dist < 6;

    res = playStep(env, trainer.brains, obs, rng, false);

    decisions++;
    if (a.act[2] === 1) jumpDecisions++;
    if (a.act[3] === 1) grabDecisions++;
    if (!a.grounded) airborneDecisions++;
    if (a.held >= 0) heldDecisions++;
    if (near) {
      threatened++;
      if (!a.grounded) {
        threatenedAirborne++;
        clutch = true;
      }
    }
  } while (!res.done && env.frame < roundFrames);

  if (res.tagged) tags++;
  frames += env.frame;
  if (clutch) {
    episodesWithClutchJump++;
    if (!res.tagged) escapesWithClutchJump++;
  }
}

const pct = (a, b) => `${((100 * a) / Math.max(1, b)).toFixed(1)}%`;

console.log(`\n=== ${label} · ${UPDATES} updates ===`);
console.log(`  tag rate                ${pct(tags, EPISODES)}   (50% = even fight)`);
console.log(`  Albert escapes          ${pct(EPISODES - tags, EPISODES)}`);
console.log(`  mean survival           ${(frames / EPISODES / 60).toFixed(2)}s of ${ROUND_S}s`);
console.log(`  --- what Albert does ---`);
console.log(`  presses jump            ${pct(jumpDecisions, decisions)} of decisions`);
console.log(`  actually airborne       ${pct(airborneDecisions, decisions)} of decisions`);
console.log(`  presses grab            ${pct(grabDecisions, decisions)} of decisions`);
console.log(`  carrying a crate        ${pct(heldDecisions, decisions)} of decisions`);
console.log(`  --- does the jump save him? ---`);
console.log(`  airborne while Kai <6u  ${pct(threatenedAirborne, threatened)} of threatened moments`);
console.log(
  `  rounds with a jump under pressure  ${pct(episodesWithClutchJump, EPISODES)}` +
    `  -> escaped ${pct(escapesWithClutchJump, episodesWithClutchJump)} of those`,
);
console.log(
  `  baseline escape rate without one   ${pct(
    EPISODES - tags - escapesWithClutchJump,
    EPISODES - episodesWithClutchJump,
  )}`,
);
