# MountainCar: pumping energy into the swing

An underpowered car in a valley has to reach the flag on the right hill. Its engine cannot climb the slope directly, so
the agent rocks it: it always accelerates the way the car is already moving.

```bash
dotnet run --project examples/MountainCar -c Release                   # one episode, headless
dotnet run --project examples/MountainCar -c Release -- --render       # the same episode in a window
dotnet run --project examples/MountainCar -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/MountainCar -c Release -- --episodes 5   # seeds 4-8, and their mean
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
