# CarRacing: pure pursuit

A car on a randomly generated track, seen from above. The observation is a 96×96 image, but the agent drives from the
track's geometry, which the unwrapped env exposes.

```bash
dotnet run --project examples/CarRacing -c Release                   # one episode, headless
dotnet run --project examples/CarRacing -c Release -- --render       # the same episode in a window
dotnet run --project examples/CarRacing -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/CarRacing -c Release -- --episodes 5   # seeds 11-15, and their mean
dotnet run --project examples/CarRacing -c Release -- --record docs/environments/box2d/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
CarRacing-v3, seed 11: 1000 steps, return 880.7, truncated by the time limit
```

## The agent

Pure pursuit: find the centre-line point nearest the car, aim at the one five track tiles further on, and steer by the
angle between the car's heading and that point. The throttle stays on while the speed is under a limit that drops as the
angle grows, and the brake comes on in sharp bends. Every new tile visited pays, every step costs 0.1. Seed 11 is its
best course of seeds 0-19: they average 752.2, and none finishes the lap within the 1,000 steps.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| W | Gas |
| A / D | Steer left / right |
| S | Brake |
| W+A, W+D, S+A, S+D | Combined |
| Nothing held | Coast |

## Recording the docs page's media

`--record <folder>` plays the seed-11 episode once more (whatever `--seed` says) and writes the media of Car Racing's
page on the docs site (DR-112) into the folder, `docs/environments/box2d/media` for the site:

| File | What it is |
|---|---|
| `CarRacing-v3.episode.mp4` | The whole episode, one 600×400 frame per step at the env's 50 frames a second (1,001 frames: the reset and 1,000 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `CarRacing-v3.episode.json` | Every step: the state acted on (the hull's position and angle, its speed, the tiles touched so far, of `tiles_total`), the agent's three float32 commands (`action_dtype` float32, each written as the float64 that holds it exactly) and the reward; the state after the last step as `final` |
| `CarRacing-v3.figure.png`, `.figure.json` | The frame after 640 steps, the car turning into a left bend along its kerb, taken from the recorder, which the page annotates, and what its labels point at, in the frame's pixels: the car, the score, and each indicator's bar |
| `CarRacing-v3.observation.png` | The agent's 96×96 view of the same moment, the observation it acts on at step 640 |

```
CarRacing-v3, seed 11: 1000 steps, return 880.7, recorded into docs/environments/box2d/media
```

Each step renders twice, the observation and the video's frame, so the recording takes about half a minute. ffmpeg
must be on PATH to record; every other mode runs without it.

## The API it shows

- `(CarRacingEnv)env.Unwrapped`: the env under `make`'s wrappers, with its `Track` (the centre-line points) and
  `Car.Hull` (a Box2D body: `Position`, `Angle`, `LinearVelocity`)
- `Play.Run` with the CarRacing key map from Gymnasium's own `play` documentation
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
