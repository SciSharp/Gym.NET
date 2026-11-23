# MountainCar: pumping energy into the swing

An underpowered car in a valley has to reach the flag on the right hill. Its engine cannot climb the slope directly, so
the agent rocks it: it always accelerates the way the car is already moving.

```bash
dotnet run --project examples/MountainCar -c Release                   # one episode, headless
dotnet run --project examples/MountainCar -c Release -- --render       # the same episode in a window
dotnet run --project examples/MountainCar -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/MountainCar -c Release -- --episodes 5   # seeds 4-8, and their mean
dotnet run --project examples/MountainCar -c Release -- --record docs/environments/classic_control/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
MountainCar-v0, seed 4: 122 steps, return -122.0, reached the flag
```

## The agent

One line decides each step: accelerate right (action 2) when the velocity is zero or positive, left (action 0)
otherwise. Each push adds to the car's energy, as pumping a swing does, and after a few swings the car carries over the
top.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| Left arrow | Accelerate left |
| Right arrow | Accelerate right |
| Nothing held | Coast |

## Recording the docs page's media

`--record <folder>` plays the seed-4 episode once more (whatever `--seed` says) and writes the media of MountainCar's
page on the docs site (DR-112) into the folder, `docs/environments/classic_control/media` for the site:

| File | What it is |
|---|---|
| `MountainCar-v0.episode.mp4` | The whole episode, one frame per step at the env's 30 frames a second (123 frames: the reset and 122 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `MountainCar-v0.episode.json` | Every step: the observation the agent acted on (`position`, `velocity`), its action and the reward, for the page's anatomy chart; the state after the last step as `final` |
| `MountainCar-v0.figure.png`, `.figure.json` | The frame after 80 steps, the top of the car's back-swing, which the page annotates, and where its labels point: the car's centre, the hill under it and its slope, the flag, the starting range, the valley's floor and the wall, in the frame's pixels |

```
MountainCar-v0, seed 4: 122 steps, return -122.0, recorded into docs/environments/classic_control/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `Gymnasium.Make<NDArray, NDArray>("MountainCar-v0", kwargs)`: `make`'s stack, `TimeLimit` of 200 steps over
  `OrderEnforcing` over `PassiveEnvChecker` over `MountainCarEnv`
- A discrete action is a 0-d int64 array: `NDArray.Scalar(2L)`
- `Play.Run(env, keysToAction, noop: 1L, seed)`: Gymnasium's `play`, with explicit key codes (`KeyCode.Left`)
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
