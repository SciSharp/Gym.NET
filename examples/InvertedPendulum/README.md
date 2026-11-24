# InvertedPendulum: Farama's SAC expert

A pole hinged on a cart that slides on a rail, simulated by Gym.NET's own MuJoCo engine (bit for bit with MuJoCo 4.0.0).
A published expert agent, Farama's own, keeps the pole upright.

```bash
dotnet run --project examples/InvertedPendulum -c Release                   # one episode, headless
dotnet run --project examples/InvertedPendulum -c Release -- --render       # the same episode in a window
dotnet run --project examples/InvertedPendulum -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/InvertedPendulum -c Release -- --episodes 5   # seeds 0-4, and their mean
dotnet run --project examples/InvertedPendulum -c Release -- --record docs/environments/mujoco/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
InvertedPendulum-v5, seed 0: 1000 steps, return 1000.0, truncated by the time limit
```

## The agent

`policy.json` holds the agent: Farama's SAC (Soft Actor-Critic) expert, the agent the Farama Foundation, Gymnasium's
maintainers, trained on this env to record the Minari datasets. It comes from Hugging Face,
[farama-minari/InvertedPendulum-v5-SAC-expert](https://huggingface.co/farama-minari/InvertedPendulum-v5-SAC-expert) at
revision `d67981f`, imported by `tools/agents/import_agents.py`, and is a copy of
[`tests/Gym.Tests/Agents/InvertedPendulum-v5.json`](../../tests/Gym.Tests/Agents/InvertedPendulum-v5.json), where the
tests measure it. It is a Stable-Baselines3 network of two dense layers of 256 with relu: each step it casts the
observation to float32, runs it through the network, and squashes the result into the motors' range with tanh, as
Stable-Baselines3's deterministic `predict` does.

Over seeds 0-9 it scores 1,000.0 in Gym.NET (every seed balances all 1,000 steps); the model card reports 1000.00 +/-
0.00. The model repository states no licence; its training code is MIT. Gym.NET commits the weights with this
attribution, on the maintainer's decision (DR-109).

## Watching and playing

`--render` opens the env's window. It is MuJoCo's window, with Gymnasium's controls (DR-70): Tab cycles the cameras, the
mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws MuJoCo's frames in software, every one by
default: when the window falls behind real time, D turns "render every frame" off and it skips frames instead.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| A | Push the cart left (-3) |
| D | Push it right (+3) |
| Nothing held | No force |

## Recording the docs page's media

`--record <folder>` plays the seed-0 episode once more (whatever `--seed` says) and writes the media of
InvertedPendulum's page on the docs site (DR-112) into the folder, `docs/environments/mujoco/media` for the site:

| File | What it is |
|---|---|
| `InvertedPendulum-v5.episode.mp4` | The whole episode, one 480×480 frame per step at the env's 25 frames a second (1,001 frames: the reset and 1,000 steps), through Gymnasium's own camera for this env (the model has none), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `InvertedPendulum-v5.episode.json` | Every step: the state acted on (the cart's position and the pole's angle, and their speeds), the expert's float32 push (`action_dtype` float32, written as the float64 that holds it exactly), the reward and `reward_survive` from the info; the state after the last step as `final` |
| `InvertedPendulum-v5.figure.png`, `.figure.json` | How it works' frame: 760×380 of the reset pose without noise, from MuJoCo's free camera 1.9 m to the side, and the camera with the cart, the pole's tip and the rail's ends projected to the frame's pixels through MuJoCo's camera geometry |

```
InvertedPendulum-v5, seed 0: 1000 steps, return 1000.0, recorded into docs/environments/mujoco/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `ExpertPolicy.Load(path)` and `policy.Act(observation)`: the agent's network, a new float32 action each step (released
  with the episode's other arrays by `[NDScoped]`)
- `policy.MakeArguments(...)`: the agent's own constructor arguments plus the render mode, for `Gymnasium.Make<NDArray,
  NDArray>(id, kwargs)`
- Human mode opens MuJoCo's window, with Gymnasium's controls (DR-70)
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
