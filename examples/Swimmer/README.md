# Swimmer: Farama's PPO expert

Three links in a viscous fluid, with two motors between them, have to swim forward. A published expert agent, Farama's
own, makes them swim with a travelling wave.

```bash
dotnet run --project examples/Swimmer -c Release                   # one episode, headless
dotnet run --project examples/Swimmer -c Release -- --render       # the same episode in a window
dotnet run --project examples/Swimmer -c Release -- --episodes 5   # seeds 0-4, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Swimmer-v5, seed 0: 1000 steps, return 361.0, truncated by the time limit
```

## The agent

`policy.json` holds the agent: Farama's PPO (Proximal Policy Optimization) expert, the agent the Farama Foundation,
Gymnasium's maintainers, trained on this env to record the Minari datasets. It comes from Hugging Face,
[farama-minari/Swimmer-v5-PPO-expert](https://huggingface.co/farama-minari/Swimmer-v5-PPO-expert) at revision `11ae7c5`,
imported by `tools/agents/import_agents.py`, and is a copy of
[`tests/Gym.Tests/Agents/Swimmer-v5.json`](../../tests/Gym.Tests/Agents/Swimmer-v5.json), where the tests measure it. It
is a Stable-Baselines3 network of two dense layers of 64 with tanh: each step it casts the observation to float32, runs
it through the network, and clips the result to the motors' range, as Stable-Baselines3's deterministic `predict` does.

Over seeds 0-9 it scores 364.0 in Gym.NET (from 361 to 366); the model card reports 363.70 +/- 1.81. The model
repository states no licence; its training code is MIT. Gym.NET commits the weights with this attribution, on the
maintainer's decision (DR-109).

## Watching and playing

`--render` opens the env's window. It is MuJoCo's window, with Gymnasium's controls (DR-70): Tab cycles the cameras, the
mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws MuJoCo's frames in software, every one by
default: when the window falls behind real time, D turns "render every frame" off and it skips frames instead. The
runner leaves the default free camera's view; Tab to the model's "track" camera, which follows it.

`--play` is not offered: its frames take about 0.3 s each to draw in software and a step is 40 ms of simulated time, so
keyboard play would run about 7 times slower than real time.

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
