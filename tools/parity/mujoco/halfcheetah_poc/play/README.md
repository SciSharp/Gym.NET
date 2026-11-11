# HalfCheetah-v5, playable

The proof of concept's managed core (`../CheetahCore.cs`, bit-exact with MuJoCo 3.15.0, see
[`MUJOCO_ASSESSMENT.md` §10](../../../../../docs/plans/MUJOCO_ASSESSMENT.md)) running in a Gym.NET viewer window. An
autopilot drives it; Tab hands the six motors to you.

```bash
dotnet run --project tools/parity/mujoco/halfcheetah_poc/play -c Release
```

| Key | Does |
|---|---|
| Tab | Autopilot on or off |
| Q/A, W/S, E/D | Back thigh, shin, foot: +1 or -1 torque while held |
| U/J, I/K, O/L | Front thigh, shin, foot |
| R | New episode (the next Gymnasium seed) |
| Space | Pause |
| + and - | Twice or half the speed |
| Esc | Quit |

## What you see

- **The physics:** the core steps exactly as MuJoCo's `mj_step` does, as the checker in the folder above proves. Each env
  step holds the action for five physics steps, in real time.
- **The view:** a side view of the x-z plane the model moves in, drawn with Gym.Rendering from the core's capsule poses.
  It is not MuJoCo's renderer (not ported, DR-31), so the state is exact but the pixels are not Gymnasium's.
- **The torques:** a leg segment turns red under a positive motor torque and blue under a negative one. The bars at
  the top right show all six.

## The autopilot

`policy.json` is a linear policy trained by Augmented Random Search on the managed core. It uses V2-t (Mania, Guy and
Recht 2018) with the paper's HalfCheetah settings. The training took 600 iterations, 38.4M env steps and 3 min 46 s on
12 threads. The best snapshot, at iteration 540, averages 5,822 over Gymnasium seeds 0-4.

It was trained only in C#, yet it scores the same in Gymnasium's HalfCheetah-v5 on MuJoCo, to the last bit:

```bash
dotnet run --project tools/parity/mujoco/halfcheetah_poc/play -c Release -- score --seeds 0-9
python tools/parity/mujoco/halfcheetah_poc/play/replay_in_gymnasium.py --seeds 0-9
```

All ten returns were identical when measured on 2026-11-11 (seed 0: 5832.688804393127 on both). The replay evaluates the
policy in the C# arithmetic order (`LinearPolicy.Act`). NumPy's matrix product sums in another order and would make the
episodes drift apart.

## Other commands

| Command | Does |
|---|---|
| `train [--iterations 600] [--threads N] [--seed 1] [--out policy.json]` | Trains a new autopilot |
| `video FILE.mp4 [--seconds 30] [--seed 0]` | Records the autopilot at 50 frames per second (needs ffmpeg on PATH) |
| `frames FOLDER [--steps 200] [--every 20]` | Writes PNG stills |
| `score [--seeds 0-4]` | Prints exact episode returns |

Pass the command after `--` in `dotnet run`. `half_cheetah.mjmodel.json` is MuJoCo's compiled model, dumped by
`../oracle.py`.
