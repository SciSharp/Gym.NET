# BipedalWalkerHardcore: RL Zoo's TQC expert

The hardcore course adds stumps, pits and stairs to BipedalWalker's. A published agent, RL Zoo's TQC agent, reads the
lidar and clears them.

```bash
dotnet run --project examples/BipedalWalkerHardcore -c Release                   # one episode, headless
dotnet run --project examples/BipedalWalkerHardcore -c Release -- --render       # the same episode in a window
dotnet run --project examples/BipedalWalkerHardcore -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/BipedalWalkerHardcore -c Release -- --episodes 5   # seeds 7-11, and their mean
dotnet run --project examples/BipedalWalkerHardcore -c Release -- --record docs/environments/box2d/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
BipedalWalkerHardcore-v3, seed 7: 968 steps, return 308.4, reached the end of the course
```

## The agent

`policy.json` holds the agent: RL Zoo's TQC (Truncated Quantile Critics) expert, the agent RL Baselines3 Zoo trained and
published for this env. It comes from Hugging Face,
[sb3/tqc-BipedalWalkerHardcore-v3](https://huggingface.co/sb3/tqc-BipedalWalkerHardcore-v3) at revision `4eb282a`,
imported by `tools/agents/import_agents.py`, and is a copy of
[`tests/Gym.Tests/Agents/BipedalWalkerHardcore-v3.json`](../../tests/Gym.Tests/Agents/BipedalWalkerHardcore-v3.json),
where the tests measure it. It is a Stable-Baselines3 network of dense layers of 400 and 300 with relu: each step it
casts the observation to float32, runs it through the network, and squashes the result into the motors' range with tanh,
as Stable-Baselines3's deterministic `predict` does.

Over seeds 0-9 it scores 243.8 in Gym.NET (six seeds of ten finish the course; seed 7, the default, in 968 steps for
308); the model card reports 208.05 +/- 121.38. The model repository states no licence; its training code is MIT.
Gym.NET commits the weights with this attribution, on the maintainer's decision (DR-109).

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

`--record <folder>` plays the seed-7 episode once more (whatever `--seed` says) and writes the media of the hardcore
section of Bipedal Walker's page on the docs site (DR-112) into the folder, `docs/environments/box2d/media` for the
site:

| File | What it is |
|---|---|
| `BipedalWalkerHardcore-v3.episode.mp4` | The whole episode, one frame per step at the env's 50 frames a second (969 frames: the reset and 968 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `BipedalWalkerHardcore-v3.episode.json` | Every step: the state acted on (BipedalWalker's trace columns: the hull's angle and velocities, both hips and knees, the feet's contact flags, the hull's x), the agent's four float32 commands (`action_dtype` float32, each written as the float64 that holds it exactly) and the reward; the state after the last step as `final` |
| `BipedalWalkerHardcore-v3.figure.png`, `.figure.json` | The frame after 110 steps, the walker at the foot of a staircase with a stump and a pit ahead, taken from the recorder rather than rendered again (a render moves the lidar's sweep on), and what its labels point at, in the frame's pixels: the hull, the legs, where each lidar ray stops and the corners of every obstacle in view |

```
BipedalWalkerHardcore-v3, seed 7: 968 steps, return 308.4, recorded into docs/environments/box2d/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `ExpertPolicy.Load(path)` and `policy.Act(observation)`: the agent's network, a new float32 action each step (released
  with the episode's other arrays by `[NDScoped]`)
- `policy.MakeArguments(...)`: the agent's own constructor arguments plus the render mode, for `Gymnasium.Make<NDArray,
  NDArray>(id, kwargs)`
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

Copy `policy.json` along with the code.
