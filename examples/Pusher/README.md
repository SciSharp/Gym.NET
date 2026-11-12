# Pusher: steering the fingertip

A seven-joint arm has to push a puck to a goal on a table. The agent does not think in joints: it steers the fingertip,
and lets the arm's Jacobian turn that into seven torques.

```bash
dotnet run --project examples/Pusher -c Release                   # one episode, headless
dotnet run --project examples/Pusher -c Release -- --render       # the same episode in a window
dotnet run --project examples/Pusher -c Release -- --episodes 5   # seeds 73-77, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Pusher-v5, seed 73: 100 steps, return -39.0, truncated by the time limit
```

## The agent

Three phases: hover behind the puck, on the line from the goal through it; descend to the puck's height; push toward a
point just past the puck. Losing the puck sideways sends the agent back to hovering. The torques are `Jᵀ (kp (target -
tip) - kd · tip velocity) - 0.3 q̇`, clipped to the motors' ±2.

The fingertip's 3×7 Jacobian comes from finite differences of the forward kinematics, on a scratch copy of the
simulation (`MjData`), and every position is read from that copy. After a step, MuJoCo's body positions (`xpos`) still
describe the step's start while the joint positions (`qpos`) have moved on, so differencing one against the other gives
a wrong Jacobian.

It does not solve Pusher: the point behind the puck often lies inside the reach the arm's joint limits forbid. Over
seeds 0-199 its best episode, seed 73 (the default), leaves the puck 0.145 from the goal. A linear policy trained by
Augmented Random Search barely moved the arm (-46.7).

## Watching and playing

`--render` opens the env's window. It is MuJoCo's window, with Gymnasium's controls (DR-70): Tab cycles the cameras, the
mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws MuJoCo's frames in software, every one by
default: when the window falls behind real time, D turns "render every frame" off and it skips frames instead.

`--play` is not offered: its 7 motors would need 14 keys held in combinations, too many to play.

## The API it shows

- `(MujocoEnv)env.Unwrapped`: the env under `make`'s wrappers, with its `Model` and `Data`
- `new MjData(model)`, `Mujoco.MjSetState`, `Mujoco.MjKinematics`, `data.Body(name).Xpos`: MuJoCo's own functions, under
  Gym.NET's names (DR-68)
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
