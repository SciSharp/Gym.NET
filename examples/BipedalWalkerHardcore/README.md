# BipedalWalkerHardcore: Gymnasium's gait

The hardcore course adds stumps, pits and stairs. Gymnasium's hand-written gait was made for the plain course, and this
example shows what happens when it meets the first obstacle.

```bash
dotnet run --project examples/BipedalWalkerHardcore -c Release                   # one episode, headless
dotnet run --project examples/BipedalWalkerHardcore -c Release -- --render       # the same episode in a window
dotnet run --project examples/BipedalWalkerHardcore -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/BipedalWalkerHardcore -c Release -- --episodes 5   # seeds 10-14, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
BipedalWalkerHardcore-v3, seed 10: 157 steps, return -77.5, fell over
```

## The agent

The same state machine as on the plain course (see `examples/BipedalWalker`): one leg carries the body while the other
swings forward, and a PD controller keeps the hull level. It does not look at the obstacles, so from seed 10 it walks
into a stump and falls at step 157. Solving this course takes a learned policy that reads the lidar.

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

## The API it shows

- `new BipedalWalkerHeuristics()` and `StepHeuristic(observation)`: Gymnasium's walking heuristic, a new float64 array
  of four motor commands each step (the caller disposes it)
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
