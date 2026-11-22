# Pusher: Farama's SAC expert

A seven-joint arm has to push a puck to a goal on a table. A published expert agent, Farama's own, sweeps to the puck
and pushes it into the goal.

```bash
dotnet run --project examples/Pusher -c Release                   # one episode, headless
dotnet run --project examples/Pusher -c Release -- --render       # the same episode in a window
dotnet run --project examples/Pusher -c Release -- --episodes 5   # seeds 7-11, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Pusher-v5, seed 7: 100 steps, return -20.4, truncated by the time limit
```

## The agent

`policy.json` holds the agent: Farama's SAC (Soft Actor-Critic) expert, the agent the Farama Foundation, Gymnasium's
maintainers, trained on this env to record the Minari datasets. It comes from Hugging Face,
[farama-minari/Pusher-v5-SAC-expert](https://huggingface.co/farama-minari/Pusher-v5-SAC-expert) at revision `3e10360`,
imported by `tools/agents/import_agents.py`, and is a copy of
[`tests/Gym.Tests/Agents/Pusher-v5.json`](../../tests/Gym.Tests/Agents/Pusher-v5.json), where the tests measure it. It
is a Stable-Baselines3 network of two dense layers of 256 with relu: each step it casts the observation to float32, runs
it through the network, and squashes the result into the motors' range with tanh, as Stable-Baselines3's deterministic
`predict` does.

Over seeds 0-9 it scores -28.0 in Gym.NET (seed 7, the default, puts the puck in the goal for -20.4); the model card
reports -22.05 +/- 2.69. The model repository states no licence; its training code is MIT. Gym.NET commits the weights
with this attribution, on the maintainer's decision (DR-103).

## Watching and playing

`--render` opens the env's window. It is MuJoCo's window, with Gymnasium's controls (DR-70): Tab cycles the cameras, the
mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws MuJoCo's frames in software, every one by
default: when the window falls behind real time, D turns "render every frame" off and it skips frames instead.

`--play` is not offered: its 7 motors would need 14 keys held in combinations, too many to play.

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
