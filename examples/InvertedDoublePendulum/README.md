# InvertedDoublePendulum: two poles, one linear policy

Two poles stacked on a cart, simulated by Gym.NET's MuJoCo engine. A linear policy, trained by Augmented Random Search
on the same engine, holds both upright.

```bash
dotnet run --project examples/InvertedDoublePendulum -c Release                   # one episode, headless
dotnet run --project examples/InvertedDoublePendulum -c Release -- --render       # the same episode in a window
dotnet run --project examples/InvertedDoublePendulum -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/InvertedDoublePendulum -c Release -- --episodes 5   # seeds 0-4, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
InvertedDoublePendulum-v5, seed 0: 1000 steps, return 9359.6, truncated by the time limit
```

## The agent

`policy.json` holds the policy. Each step it normalizes the 9 observation entries by the mean and scale seen in
training, weighs them into one force, and clips it to [-1, 1]: `a = clip(W · (s - mean) / scale, -1, 1) · high`.
`tools/showreel`'s `train` command found it in 2 minutes on 32 threads (Augmented Random Search V2-t, Mania, Guy and
Recht 2018). It holds both poles for all 1,000 steps on seeds 0-4, and the file's `info` records the training settings
and those returns.

Gymnasium sets no camera for this env, and MuJoCo's free camera shows the poles small, so `--render` and `--play` pass a
closer `default_camera_config` (2.6 m away, looking at the poles' middle). It changes the view, never the dynamics.

## Watching and playing

`--render` opens the env's window. It is MuJoCo's window, with Gymnasium's controls (DR-70): Tab cycles the cameras, the
mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws MuJoCo's frames in software, every one by
default: when the window falls behind real time, D turns "render every frame" off and it skips frames instead.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| A | Push the cart left (-1) |
| D | Push it right (+1) |
| Nothing held | No force |

## The API it shows

- `Gymnasium.Make<NDArray, NDArray>(id, new { render_mode = "human", default_camera_config = ... })`: a dictionary
  keyword argument, by MuJoCo's camera field names
- `((Box)env.ActionSpace).High`: the action bounds the policy scales its outputs by
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
