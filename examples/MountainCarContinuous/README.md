# MountainCarContinuous: pumping energy with a throttle

The valley of [MountainCar](../MountainCar), with a continuous engine: each step sets a force anywhere in [-1, 1],
reaching the flag pays 100, and every push costs 0.1 times the squared force. The agent rocks the car as MountainCar's
does, accelerating the way it already moves, with one difference that pays: while the car is at rest it pushes left,
away from the flag, so the left slope gives it room to gather speed.

```bash
dotnet run --project examples/MountainCarContinuous -c Release                     # one episode, headless
dotnet run --project examples/MountainCarContinuous -c Release -- --render         # the same episode in a window
dotnet run --project examples/MountainCarContinuous -c Release -- --play           # you play it, from the keyboard
dotnet run --project examples/MountainCarContinuous -c Release -- --episodes 100 --seed 0   # seeds 0-99, and their mean
dotnet run --project examples/MountainCarContinuous -c Release -- --force 0.5      # half the force: slower, cheaper
dotnet run --project examples/MountainCarContinuous -c Release -- --record docs/environments/classic_control/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
MountainCarContinuous-v0, seed 4: 78 steps, return 92.20, reached the flag
```

## The agent

One line decides each step: a force of +1 when the velocity is positive, -1 otherwise. At rest the velocity is 0, so
the first push is left: the car rolls back up the left slope (from seed 4 all the way to the wall, which stops it) and
then swings across the valley to the flag. Each step costs 0.1 at full force, so the return is 100 less 0.1 a step.

| Seeds 0-99 | Steps | Mean return |
|---|---|---|
| This rule, full force | 75 to 86 | 92.03, above the 90 the registry counts as solved |
| Pushing right from rest, as MountainCar's rule does | 105 to 111 | 89.37 |
| This rule, `--force 0.5` | about twice as many | 96.10 |

Gentler pushes pay because the cost is the force's square: half the force costs a quarter as much a step, and the swing
takes only about twice as long. `--force F` takes any F in (0, 1]; the env would clip a larger force to 1 but still
charge its square.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits. The env has no key mapping of its own (Gymnasium's continuous
class has none), so the example passes one:

| Key | Does |
|---|---|
| Left arrow | Push left at full force (-1) |
| Right arrow | Push right at full force (+1) |
| Nothing held | Engine off (0), which costs nothing |

## Recording the docs page's media

`--record <folder>` plays the seed-4 episode at full force once more (whatever `--seed` and `--force` say) and writes
the media of the env's page on the docs site (DR-112) into the folder, `docs/environments/classic_control/media` for
the site:

| File | What it is |
|---|---|
| `MountainCarContinuous-v0.episode.mp4` | The whole episode, one frame per step at the env's 30 frames a second (79 frames: the reset and 78 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `MountainCarContinuous-v0.episode.json` | Every step: the observation the agent acted on (`position`, `velocity`), its force and the reward, for the page's anatomy chart; the state after the last step as `final` |
| `MountainCarContinuous-v0.figure.png`, `.figure.json` | The frame after 75 steps, the car on the right slope three steps from the flag, which the page annotates, and where its labels point: the car's centre, the hill under it and its slope, the flag at 0.45, the starting range, the valley's floor and the wall, in the frame's pixels |

```
MountainCarContinuous-v0, seed 4: 78 steps, return 92.20, recorded into docs/environments/classic_control/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `Gymnasium.Make<NDArray, NDArray>("MountainCarContinuous-v0", kwargs)`: `make`'s stack, `TimeLimit` of 999 steps
  over `OrderEnforcing` over `PassiveEnvChecker` over `MountainCarEnv` with `continuous: true`, the one class both
  MountainCars share
- A continuous action is a one-element float32 array, the action space's dtype: `np.array(new[] { 1f })`
- `Play.Run(env, keysToAction, noop, seed)`: Gymnasium's `play` with arrays as actions; `play` borrows them, so the
  example disposes them afterwards
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
