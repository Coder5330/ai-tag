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

## What's actually happening

**The room is 3D.** Both agents move on the floor, jump under gravity, stand on
crates, and can grab a crate to carry, shove or throw it. A tag only counts when the
two bodies overlap in *height* as well as on the floor, so a well-timed jump over an
incoming Kai is a legal escape. A crate pushed against a block becomes a step up onto
it — reachable by jumping from the crate, but not from the floor.

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

**Self-play.** Training both brains at once collapses into one brittle strategy, so
only one trains at a time and the trainee swaps every 8 updates. Its opponent is drawn
from a pool of the 10 most recent frozen snapshots, re-drawn on every reset, so the
trainee has to beat a spread of past strategies instead of overfitting to the newest
one.

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

## Checking it learns

Self-play scores are ambiguous on their own — the tag rate can sit flat while *both*
agents improve. The harness also benchmarks each brain against a frozen opponent, so
the curves mean something in absolute terms:

```sh
node test/train.mjs 340 0     # updates, room index (0 open · 1 crates · 2 maze)
```

`KAI vs random Albert` should climb to 100%, and once a competent Kai has been frozen
as a reference, `ALBERT vs` that fixed chaser should climb from a couple of seconds
toward the 10s round limit. A 340-update run of room 0 takes about 6½ minutes and
gives:

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
