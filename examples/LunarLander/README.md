# LunarLander: Gymnasium's heuristic

A lander falls toward a pad between two flags. Gymnasium ships a hand-written controller for it, and Gym.NET ships the
same one as `LunarLanderEnv.Heuristic`, computing the same actions bit for bit.

```bash
dotnet run --project examples/LunarLander -c Release                   # one episode, headless
dotnet run --project examples/LunarLander -c Release -- --render       # the same episode in a window
dotnet run --project examples/LunarLander -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/LunarLander -c Release -- --episodes 5   # seeds 7-11, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
LunarLander-v3, seed 7: 299 steps, return 248.6, landed
```

## The agent

A PD controller aims the lander's angle at a target that leans it toward the pad, and fires the main engine to hold a
descent rate that slows as the lander nears the ground. Once both legs touch, it cuts the engines. The episode ends when
the lander comes to rest (a +100 bonus) or crashes (-100).

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| W | Main engine |
| A | Left orientation engine |
| D | Right orientation engine |
| Nothing held | Engines off |

## The API it shows

- `LunarLanderEnv.Heuristic(env, observation)`: Gymnasium's `heuristic(env, s)`, a 0-d int64 action
- `Gymnasium.Make<NDArray, NDArray>("LunarLander-v3", kwargs)`: `TimeLimit` of 1,000 steps, Box2D 2.3.0 in managed C#
  underneath
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
