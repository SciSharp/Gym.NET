# Pendulum: swinging up and balancing

A pendulum hangs from a motor that is too weak to lift it straight up. The agent swings it up by pumping energy, then
balances it upside down.

```bash
dotnet run --project examples/Pendulum -c Release                   # one episode, headless
dotnet run --project examples/Pendulum -c Release -- --render       # the same episode in a window
dotnet run --project examples/Pendulum -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/Pendulum -c Release -- --episodes 5   # seeds 5-9, and their mean
dotnet run --project examples/Pendulum -c Release -- --record docs/environments/classic_control/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Pendulum-v1, seed 5: 200 steps, return -114.1, truncated by the time limit
```

## The agent

Far from the top, it applies full torque in the direction of the swing until the pendulum's energy, `w²/2 + 15 cos θ`
for Gymnasium's g = 10 and m = l = 1, reaches that of the upright rest. Within about 32° of the top (cos θ > 0.85), a PD
controller, `u = -(12 θ + 2.5 w)`, catches the pendulum and holds it. The reward penalizes the angle, the speed and the
torque, so the return is negative even when it works.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| A | Full torque one way (-2) |
| D | Full torque the other way (+2) |
| Nothing held | No torque |

## Recording the docs page's media

`--record <folder>` plays the seed-5 episode once more (whatever `--seed` says) and writes the media of Pendulum's page
on the docs site (DR-112) into the folder, `docs/environments/classic_control/media` for the site:

| File | What it is |
|---|---|
| `Pendulum-v1.episode.mp4` | The whole episode, one frame per step at the env's 30 frames a second (201 frames: the reset and 200 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `Pendulum-v1.episode.json` | Every step: the angle from upright (`atan2` of the observation's sine and cosine) and the angular velocity the agent acted on, its torque, the controller's regime (0 pump, 1 brake, 2 hold) and the reward, for the page's anatomy chart; the state after the last step as `final` |
| `Pendulum-v1.figure.png`, `.figure.json` | The frame after 10 steps, the rod swinging up the right side under full torque, which the page annotates, and where its labels point: the pivot, the rod's end and the arrow's size, in the frame's pixels |

```
Pendulum-v1, seed 5: 200 steps, return -114.1, recorded into docs/environments/classic_control/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `Gymnasium.Make<NDArray, NDArray>("Pendulum-v1", kwargs)`: `TimeLimit` of 200 steps over `PendulumEnv`
- A continuous action is a float32 array: `np.array(new[] { 1.5f })`
- `Play.Run` with float32 arrays as actions and an explicit no-op
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
