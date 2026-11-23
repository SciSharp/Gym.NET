# Acrobot: pumping the swing

Two links hang from a fixed pivot, and only the joint between them has a motor. The agent swings the tip of the lower
link above the line by pumping energy into the swing.

```bash
dotnet run --project examples/Acrobot -c Release                   # one episode, headless
dotnet run --project examples/Acrobot -c Release -- --render       # the same episode in a window
dotnet run --project examples/Acrobot -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/Acrobot -c Release -- --episodes 5   # seeds 1-5, and their mean
dotnet run --project examples/Acrobot -c Release -- --record docs/environments/classic_control/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Acrobot-v1, seed 1: 65 steps, return -64.0, swung the tip over the line
```

## The agent

It applies torque in the direction the second joint is already turning: +1 (action 2) when the joint's angular velocity
is positive, -1 (action 0) otherwise. Every swing gains energy, and the tip climbs a little higher each time. Over seeds
0-9 it crosses the line in 83 steps on average.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| A | Torque -1 on the second joint |
| D | Torque +1 |
| Nothing held | No torque |

## Recording the docs page's media

`--record <folder>` plays the seed-1 episode once more (whatever `--seed` says) and writes the media of Acrobot's page
on the docs site (DR-112) into the folder, `docs/environments/classic_control/media` for the site:

| File | What it is |
|---|---|
| `Acrobot-v1.episode.mp4` | The whole episode, one frame per step at the env's 15 frames a second (66 frames: the reset and 65 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `Acrobot-v1.episode.json` | Every step: both joint angles (from the observation's cosines and sines), both speeds and the tip's height above the pivot that the agent acted on, its action and the reward, for the page's anatomy chart; the state after the last step as `final` |
| `Acrobot-v1.figure.png`, `.figure.json` | The frame after 64 steps, the tip 0.80 m above the pivot and one step from the line, which the page annotates, and where its labels point: the pivot, the elbow, the tip and the line, in the frame's pixels |

```
Acrobot-v1, seed 1: 65 steps, return -64.0, recorded into docs/environments/classic_control/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `Gymnasium.Make<NDArray, NDArray>("Acrobot-v1", kwargs)`: `TimeLimit` of 500 steps over `AcrobotEnv`
- `Play.Run` with string keys (`"a"`) and an int no-op
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
