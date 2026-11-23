# BipedalWalker: Gymnasium's gait

A two-legged walker on rolling grass has to reach the end of the course without falling. Gymnasium ships a hand-written
gait for it, and Gym.NET ships the same one as `BipedalWalkerHeuristics`.

```bash
dotnet run --project examples/BipedalWalker -c Release                   # one episode, headless
dotnet run --project examples/BipedalWalker -c Release -- --render       # the same episode in a window
dotnet run --project examples/BipedalWalker -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/BipedalWalker -c Release -- --episodes 5   # seeds 0-4, and their mean
dotnet run --project examples/BipedalWalker -c Release -- --record docs/environments/box2d/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
BipedalWalker-v3, seed 0: 1598 steps, return 314.5, reached the end of the course
```

## The agent

A state machine with three phases per leg: one leg stays straight to carry the body while the other lifts, swings
forward and comes down; then they swap. A PD controller on the hips keeps the hull level. It times its phases by the
joints' angles, the speed and the feet's contact flags, and never reads the lidar. Seed 0 crosses the whole course,
but it is a lucky course: over seeds 0-99 the gait reaches the end 4 times and falls 91 times, a mean return of
48.8.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| Q / A | The first hip, one way or the other |
| W / S | The first knee |
| U / J | The second hip |
| I / K | The second knee |
| Nothing held | Every motor off |

## Recording the docs page's media

`--record <folder>` plays the seed-0 episode once more (whatever `--seed` says) and writes the media of Bipedal
Walker's page on the docs site (DR-112) into the folder, `docs/environments/box2d/media` for the site:

| File | What it is |
|---|---|
| `BipedalWalker-v3.episode.mp4` | The whole episode, one frame per step at the env's 50 frames a second (1,599 frames: the reset and 1,598 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `BipedalWalker-v3.episode.json` | Every step: the state acted on (the hull's angle and velocities, both hips and knees, the feet's contact flags, the hull's x), the four commands at full precision (`action_dtype` float64) and the reward; the state after the last step as `final` |
| `BipedalWalker-v3.figure.png`, `.figure.json` | The frame after 400 steps, one leg on the ground and the other swinging, taken from the recorder rather than rendered again (a render moves the lidar's sweep on), which the page annotates, and what its labels point at, in the frame's pixels: the hull and its angle, the four leg segments and where each lidar ray stops |

```
BipedalWalker-v3, seed 0: 1598 steps, return 314.5, recorded into docs/environments/box2d/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `new BipedalWalkerHeuristics()` and `StepHeuristic(observation)`: Gymnasium's walking heuristic, a new float64 array
  of four motor commands each step (the caller disposes it)
- `Play.Run` with a generated key map: 80 combinations of eight keys
- `env.Reset(seed: s)` and `env.Step(action)`: their results own the arrays they hold (D19), so the example releases
  each one once the agent has acted on it

## Outside this repository

The project references Gym.NET's source projects so it always builds against the current tree (`Gym.NET.sln` builds it).
In your own project, reference the packages instead:

```bash
dotnet add package Gym.NET.Environments        # Gym.NET, the environments, their registration
dotnet add package Gym.NET.Rendering.Avalonia  # only for --render and --play
```

and drop the `Import` of `Gym.NET.Environments.targets` from the project file: the package brings it, and it registers
the built-in environment ids before `Main` runs.
