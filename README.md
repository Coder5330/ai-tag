# AI Tag — Albert & Kai

Two neural networks learning to play tag in a 3D room, trained from scratch in your
browser by deep reinforcement learning. Orange **Albert** runs; blue **Kai** tags.

Inspired by AI Warehouse's *AI Learns to Play Tag* — same recipe (deep RL, self-play,
raycast vision, `[move, turn, jump, grab]` outputs), shrunk down so it converges in
minutes on one CPU core instead of days on a training farm.

No dependencies, no build step, no network access. Everything — the physics, the
neural network, the optimiser, the renderer — is plain ES modules in `src/`.

## Run it

```sh
python3 -m http.server 8000     # or: npx http-server -p 8000
open http://localhost:8000
```

A local server is required: ES modules and Web Workers can't load from `file://`.

Training starts immediately in a Web Worker, so the arena keeps rendering at 60fps
however heavy an update gets. Expect Kai to be catching Albert reliably after about
40 updates (~45 seconds), with Albert's counterplay developing over the next few
minutes.

## Deploy it

It's a static site with no build step, so either host works and both are configured:

```sh
npx vercel deploy --prod        # uses vercel.json
```

For Render, point a new Blueprint at the repo and it picks up `render.yaml`
(static site, no build command, publish path `.`).

Nothing server-side is involved — all the training happens in the visitor's browser,
so the deployment is just files on a CDN. No cross-origin isolation headers are needed
either; the Web Worker doesn't use `SharedArrayBuffer`.

## What's actually happening

**The room is 3D.** Both agents move on the floor, jump under gravity, stand on
crates, and can grab a crate to carry, shove or throw it. A tag only counts when the
two bodies overlap in *height* as well as on the floor, so a well-timed jump over an
incoming Kai is a legal escape. A crate pushed against a block becomes a step up onto
it — reachable by jumping from the crate, but not from the floor.

**Rounds are 6 seconds, and that is the most important number in the game.** In a
closed room a pursuer corners an evader *eventually* whatever the speeds — measured
with `test/balance.mjs`, a scripted evader escapes 0% of 10-second rounds even when it
is faster than the pursuer. The clock, not the physics, decides whether running away
can pay. Albert doesn't have to escape forever, only outlast the buzzer. Kai is faster
in a straight line but turns in a wide 5.7-unit arc against Albert's 2.2-unit one, so
Albert's win condition is to bait a commitment and cut across it.

**The brains.** Each agent owns a 4-layer network: 99 inputs → 64 → 64 → four action
heads plus a value estimate (~11k parameters). Inputs are its own position, height,
heading, velocity, grounded and carrying flags; all of the same for its opponent;
where the crates are; time left in the round; its previous action; and 13 raycasts
reporting distance plus whether the hit was a wall, a crate or the other agent. It
emits four numbers every 4 frames — `[move, turn, jump, grab]`, e.g. `[1, 2, 0, 1]`
meaning "forward, turn right, don't jump, try to grab".

**Rewards.** Tag lands: Kai `+1`, Albert `−1`. Every frame alive: Albert `+0.001`,
Kai `−0.001`, so Kai is pushed to finish fast rather than stall. Albert gets a one-off
`+1` the first time he picks up a crate in a round — without it the crates are simply
never discovered, the same trick the reference uses. A small potential-based distance
term gets the very first chase started; because it telescopes it provably cannot
change which policy is optimal (Ng et al., 1999), only how quickly it is found.

**Self-play, with the losing agent given the training turn.** Only one brain trains at
a time; its opponent is drawn from a league pool — half recent snapshots, half a
reservoir sample of the whole run — so it must beat a spread of strategies rather than
just the newest.

But a *fixed* alternation schedule does not work here, and the reason is the single
most interesting thing this project turned up. Chasing is far easier to learn than
evading, so the tagger compounds a lead until the runner has no winnable rounds left.
At that point every action the runner takes is equally doomed, its advantages go flat,
and it stops learning altogether. Warm-started from the scripted evader it begins at
58% escapes and decays to 0% over 480 updates:

```
fixed schedule:   58% -> 35% -> 15% -> 23% -> 10% -> 3% -> 3% -> 0%
losing-side turn: 52% -> 33% -> 37% -> 32% -> 55% -> 22% -> 20% -> 35%
```

So the trainer measures the real head-to-head score every few updates and hands the
training turn to whichever agent is behind, holding it there until the match returns
near even. That removes the collapse entirely — the runner holds 20-55% indefinitely
across seeds instead of decaying to nothing.

That measurement is deliberately separate from the training histogram. The histogram
scores the trainee against the league pool and can read 60% while the newest-vs-newest
match reads 3%; balancing against it would optimise a number nobody ever sees. It is
also why the tag rate shown in the UI is the head-to-head one.

**The runner is warm-started.** PPO reliably fails to rediscover plain fleeing: against
the same trained tagger, the scripted evader in `test/balance.mjs` escapes 73% of
rounds where a from-scratch policy manages 27%. So the runner is first cloned from that
script by supervised cross-entropy and PPO refines from there. Without it the opening
minutes are a walkover — 10% escapes at update 60, against 52% with it.

**The algorithm** is PPO — clipped surrogate objective, GAE(λ), entropy bonus decayed
over training, Adam, global gradient-norm clipping — written out longhand in
`src/nn.js` and `src/trainer.js`, including the backward pass. 64 environments step in
parallel per rollout.

Kai is faster in a straight line; Albert turns harder. That asymmetry is what makes
the fight interesting: Albert's winning move is to make Kai commit, then cut across
him.

## Layout

| File | |
|---|---|
| `src/env.js` | 3D physics, crates, jumping, raycast vision, rewards, the three rooms |
| `src/nn.js` | dense layers, manual backprop, Adam — flat `Float32Array` throughout |
| `src/trainer.js` | PPO rollout / GAE / update loop, self-play opponent pool |
| `src/worker.js` | runs the trainer off the main thread |
| `src/render.js` | flat-shaded perspective renderer on Canvas2D |
| `src/charts.js` | training curves |
| `src/main.js` | live showcase match + UI |
| `test/train.mjs` | headless training harness (Node) |
| `test/physics.mjs` | contract checks for the rules the agents exploit |
| `test/balance.mjs` | scripted-vs-scripted physics check — no learning involved |
| `test/oracle.mjs` | is the arena survivable at all? learned policy vs a scripted evader |
| `test/diagnose.mjs` | trains a pair, then measures how they actually play |

## Checking it learns

Self-play scores are ambiguous on their own — the tag rate can sit flat while *both*
agents improve. The harness also benchmarks each brain against a frozen opponent, so
the curves mean something in absolute terms:

```sh
node test/train.mjs 340 0     # updates, room index (0 open · 1 crates · 2 maze)
```

`KAI vs random Albert` should climb to 100%, and once a competent Kai has been frozen
as a reference, `ALBERT vs` that fixed chaser should climb from a couple of seconds
toward the round limit. A 340-update run of room 0 takes about 6½ minutes and gives:

```
  upd | trained | selfplay | KAI vs random Albert | ALBERT vs frozen Kai
    1 |   Kai   | tag   0% |      tag   2%        |   survive  ——
   40 |   Kai   | tag  99% |      tag 100%        |   survive 3.70s   <- Kai solved
  150 |   Kai   | tag 100% |      tag 100%        |   survive 8.18s
  340 |   Kai   | tag 100% |      tag 100%        |   survive 9.16s   <- Albert caught up
```

`node test/physics.mjs` checks the rules the agents are meant to exploit — that a jump
really does clear a rival's head, that a tag needs vertical overlap, that crates can be
stood on, carried and thrown, and that nothing escapes the room or goes non-finite.

### Balancing it

Two tools exist because "the runner always loses" has two very different causes and
they need opposite fixes:

- `node test/balance.mjs` runs a scripted evader against a scripted pursuer with no
  networks at all, so the survival time it reports is a property of the arena alone.
  Use it to pick physics — it runs in seconds instead of the six minutes a training
  run costs.
- `node test/oracle.mjs` trains a pair and then swaps the runner for that same scripted
  evader. If the script vastly outperforms the learned policy, the arena is fine and
  the learning is at fault; if both lose equally, the arena is unwinnable and no amount
  of tuning will help.

That distinction matters. Ten trained configurations — turn radius, tag radius, jump
height, air control, jump cooldown, tagger speed, opponent-pool shape — all produced
the same ~3s survival and a 99–100% tag rate, because every one of them was tuning the
wrong layer. Two things actually moved it: the round length (a bounded arena makes long
rounds unwinnable for an evader at any speed) and giving the training turn to whichever
agent is losing.

A warning learned the hard way: measure the head-to-head, not the training histogram.
Several apparent fixes here were the runner beating stale snapshots in its own opponent
pool.

## Notes

- **The scoreboards break.** Slam into the back wall hard enough — or throw a crate
  at it — and the nearest display shatters, with glass shards and a camera jolt. It is
  pure decoration: the effect lives entirely in the renderer, reads the simulation
  without writing to it, and resets each round, so it can never touch physics or
  training. `__tag.renderer.impactAt(22, 11, 14)` in the console smashes one on demand.
- **Save / load brains** writes a JSON file of both networks, so a long run survives a
  reload. Switching rooms keeps the brains and carries on training, the way the
  reference continues on top of previous brains room to room.
- **Show what they see** draws each agent's actual raycasts as a view volume; rays that
  have found the opponent light up.
- **Best move only** switches from sampling the policy to taking its argmax — the same
  brain, played without exploration noise.
- Chart colours are checked with the `dataviz` palette validator against the dark
  surface (`#e86a1c` / `#2e86e0`: lightness band, chroma, CVD separation ΔE 26.5,
  contrast — all pass).
