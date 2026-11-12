# Reacher: inverse kinematics

A two-link arm on a table has to put its fingertip on a target that appears somewhere new each episode. The agent solves
the geometry exactly, then drives the joints to the answer.

```bash
dotnet run --project examples/Reacher -c Release                   # one episode, headless
dotnet run --project examples/Reacher -c Release -- --render       # the same episode in a window
dotnet run --project examples/Reacher -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/Reacher -c Release -- --episodes 5   # seeds 0-4, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Reacher-v5, seed 0: 50 steps, return -3.8, truncated by the time limit
```

## The agent

The arm's inverse kinematics (links 0.1 and 0.11 long) gives the joint angles that put the fingertip on the target.
There are two solutions, elbow one way or the other; the agent takes the one nearer the arm's current pose. A PD
controller (gain 0.5, damping 0.1) then drives each joint to its angle, the error wrapped to [-π, π] so the joint turns
the shorter way round.

Over seeds 0-19 the return is -4.14 per target on average (-7.2 to -0.7): it is the distance to the target summed over
50 steps, plus a small control cost, so even a perfect reach pays for the way there. A linear policy trained by
Augmented Random Search stopped improving near -9.

## Watching and playing

`--render` opens the env's window. It is MuJoCo's window, with Gymnasium's controls (DR-70): Tab cycles the cameras, the
mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws MuJoCo's frames in software, every one by
default: when the window falls behind real time, D turns "render every frame" off and it skips frames instead.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| Q / A | The shoulder's motor, one way or the other |
| W / S | The elbow's motor |
| Nothing held | Both motors off |

## The API it shows

- `Gymnasium.Make<NDArray, NDArray>("Reacher-v5", kwargs)`: `TimeLimit` of 50 steps; Reacher never terminates
- `Play.Run` with a generated key map: every combination of the keys held (Gymnasium's `play` looks up the exact set)
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
