# InvertedPendulum: MuJoCo's cart-pole, balanced by state feedback

A pole hinged on a cart that slides on a rail, simulated by Gym.NET's own MuJoCo engine (bit for bit with MuJoCo 4.0.0).
The agent keeps the pole upright and the cart near the middle.

```bash
dotnet run --project examples/InvertedPendulum -c Release                   # one episode, headless
dotnet run --project examples/InvertedPendulum -c Release -- --render       # the same episode in a window
dotnet run --project examples/InvertedPendulum -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/InvertedPendulum -c Release -- --episodes 5   # seeds 0-4, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
InvertedPendulum-v5, seed 0: 1000 steps, return 1000.0, truncated by the time limit
```

## The agent

The force on the cart is linear in the state: `u = 5.330 x + 8.469 θ + 1.109 ẋ + 0.634 θ̇`, clipped to the motor's ±3.
Random search over the four gains found these: they hold the pole for all 1,000 steps on seeds 0-9, with the cart near
the middle (mean |x| 0.003). A linear policy trained by Augmented Random Search balanced too, but let the cart drift to
the end of its rail.

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

## The API it shows

- `Gymnasium.Make<NDArray, NDArray>("InvertedPendulum-v5", kwargs)`: a MuJoCo env, `TimeLimit` of 1,000 steps
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
