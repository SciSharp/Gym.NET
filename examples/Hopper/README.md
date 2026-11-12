# Hopper: a linear policy

A one-legged robot (thigh, leg, foot) in a plane has to hop forward without falling. A linear policy, trained by
Augmented Random Search on Gym.NET's own MuJoCo engine, makes it hop.

```bash
dotnet run --project examples/Hopper -c Release                   # one episode, headless
dotnet run --project examples/Hopper -c Release -- --render       # the same episode in a window
dotnet run --project examples/Hopper -c Release -- --episodes 5   # seeds 0-4, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Hopper-v5, seed 0: 1000 steps, return 3614.8, truncated by the time limit
```

## The agent

`policy.json` holds the policy. Each step it normalizes the observation by the mean and scale seen in training, weighs
it into one command per motor, clips each to [-1, 1] and scales it to the motor's range: `a = clip(W · (s - mean) /
scale, -1, 1) · high`. `tools/showreel`'s `train` command found it in 5 minutes on 32 threads, with Augmented Random
Search V2-t (Mania, Guy and Recht 2018): it perturbs the weights in random directions, plays an episode with each, and
steps toward the better ones. The file's `info` records the training settings and the returns on seeds 0-4.

The training subtracted Hopper's survival bonus from the reward, as the paper does: with it, standing still is a strong
local optimum.

## Watching and playing

`--render` opens the env's window. It is MuJoCo's window, with Gymnasium's controls (DR-70): Tab cycles the cameras, the
mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws MuJoCo's frames in software, every one by
default: when the window falls behind real time, D turns "render every frame" off and it skips frames instead. The
runner leaves the default free camera's view; Tab to the model's "track" camera, which follows it.

`--play` is not offered: its frames take about 0.3 s each to draw in software and a step is 8 ms of simulated time, so
keyboard play would run about 37 times slower than real time.

## The API it shows

- `Gymnasium.Make<NDArray, NDArray>("Hopper-v5", kwargs)`: a MuJoCo env on Gym.NET's own engine, bit for bit with MuJoCo
  4.0.0
- `((Box)env.ActionSpace).High`: the motors' ranges, which the policy scales its outputs by
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
