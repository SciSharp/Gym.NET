# HumanoidStandup: Farama's SAC expert

A humanoid lies on the floor and is rewarded for raising its head. A published expert agent, Farama's own, gets it up
within a second and keeps it standing.

```bash
dotnet run --project examples/HumanoidStandup -c Release                   # one episode, headless
dotnet run --project examples/HumanoidStandup -c Release -- --render       # the same episode in a window
dotnet run --project examples/HumanoidStandup -c Release -- --episodes 5   # seeds 0-4, and their mean
dotnet run --project examples/HumanoidStandup -c Release -- --record docs/environments/mujoco/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
HumanoidStandup-v5, seed 0: 1000 steps, return 399827.7, truncated by the time limit
```

## The agent

`policy.json` holds the agent: Farama's SAC (Soft Actor-Critic) expert, the agent the Farama Foundation, Gymnasium's
maintainers, trained on this env to record the Minari datasets. It comes from Hugging Face,
[farama-minari/HumanoidStandup-v5-SAC-expert](https://huggingface.co/farama-minari/HumanoidStandup-v5-SAC-expert) at
revision `a8a03ed`, imported by `tools/agents/import_agents.py`, and is a copy of
[`tests/Gym.Tests/Agents/HumanoidStandup-v5.json`](../../tests/Gym.Tests/Agents/HumanoidStandup-v5.json), where the
tests measure it. It is a Stable-Baselines3 network of two dense layers of 256 with relu: each step it casts the
observation to float32, runs it through the network, and squashes the result into the motors' range with tanh, as
Stable-Baselines3's deterministic `predict` does.

The expert was trained seeing positions and velocities only, so the file's `make_kwargs` turns off HumanoidStandup's
four other observation groups, and `policy.MakeArguments` passes them to `Gymnasium.Make`. They change what the agent
sees, never the dynamics.

Over seeds 0-9 it scores 400,097.0 in Gym.NET (every seed stands up within a second and stays up); the model card
reports 395485.91 +/- 33309.92. The model repository states no licence; its training code is MIT. Gym.NET commits the
weights with this attribution, on the maintainer's decision (DR-109).

## Watching and playing

`--render` opens the env's window. It is MuJoCo's window, with Gymnasium's controls (DR-70): Tab cycles the cameras, the
mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws MuJoCo's frames in software, every one by
default: when the window falls behind real time, D turns "render every frame" off and it skips frames instead. The
runner leaves the default free camera's view; Tab to the model's "track" camera, which follows it.

`--play` is not offered: its 17 motors would need 34 keys held in combinations, too many to play.

## Recording the docs page's media

`--record <folder>` plays the seed-0 episode once more (whatever `--seed` says) and writes the media of
HumanoidStandup's page on the docs site (DR-112) into the folder, `docs/environments/mujoco/media` for the site:

| File | What it is |
|---|---|
| `HumanoidStandup-v5.episode.mp4` | The whole episode, one 480×480 frame per step at the env's 67 frames a second (1,001 frames: the reset and 1,000 steps), through the model's `track` camera, H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `HumanoidStandup-v5.episode.json` | Every step: the state acted on (the torso's x, y and height, its upward velocity, each leg's hip pitch and knee), the expert's seventeen float32 commands (`action_dtype` float32, each written as the float64 that holds it exactly), the reward and its three terms from the info; the state after the last step as `final` |
| `HumanoidStandup-v5.figure.png`, `.figure.json` | How it works' frame: 760×480 of the reset pose without noise, from MuJoCo's free camera 1.7 m off, looking down at 55°, and the camera with the torso and each place where joints meet projected to the frame's pixels through MuJoCo's camera geometry |

Both envs are made with the agent's `make_kwargs`, as the example's episodes are, so the trace's episode is the one the
other modes play.

```
HumanoidStandup-v5, seed 0: 1000 steps, return 399827.7, recorded into docs/environments/mujoco/media
```

Gym.NET draws MuJoCo's frames in software, so the recording takes a few minutes. ffmpeg must be on PATH to record; every
other mode runs without it.

## The API it shows

- `ExpertPolicy.Load(path)` and `policy.Act(observation)`: the agent's network, a new float32 action each step (released
  with the episode's other arrays by `[NDScoped]`)
- `policy.MakeArguments(...)`: the agent's own constructor arguments plus the render mode, for `Gymnasium.Make<NDArray,
  NDArray>(id, kwargs)`
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
