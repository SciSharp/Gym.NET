# HalfCheetah: the autopilot, and the playable HalfCheetah

A two-legged planar "cheetah" with six motors runs forward, simulated by Gym.NET's own MuJoCo engine
(`src/Gym.Environments/MuJoCo`, bit for bit with MuJoCo 4.0.0, see
[`MUJOCO_ASSESSMENT.md` §10 and §14](../../docs/plans/MUJOCO_ASSESSMENT.md)). An autopilot trained by Augmented Random
Search drives it; in the playable version, Tab hands the six motors to you.

```bash
dotnet run --project examples/HalfCheetah -c Release                   # one episode on the autopilot, headless
dotnet run --project examples/HalfCheetah -c Release -- --render       # the same episode in MuJoCo's window
dotnet run --project examples/HalfCheetah -c Release -- --play         # the playable HalfCheetah, you driving
dotnet run --project examples/HalfCheetah -c Release -- --episodes 5   # seeds 0-4, and their mean
```

Output, the same on every run since the reset is seeded and the autopilot has no randomness of its own (measured on an
x64 CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters, and what HalfCheetah returns without
FMA):

```
HalfCheetah-v5, seed 0: 1000 steps, return 5797.0, truncated by the time limit
```

The default, `--render` and `--episodes` play the product's env, the one `Gymnasium.Make("HalfCheetah-v5")` builds, as
every example in `examples/` does (DR-74). `--play` opens the game, which moved here from
`tools/parity/mujoco/halfcheetah_poc/play`. It steps its own copy of the engine and draws its own side view, cheap
enough for real time, while Gym.NET draws MuJoCo's own frames in software at about 0.1 s each.

## The playable HalfCheetah

`--play` starts with the keyboard driving. The `play` command (`-- play`) starts with the autopilot instead, until you
press Tab.

| Key | Does |
|---|---|
| Tab | Autopilot on or off |
| Q/A, W/S, E/D | Back thigh, shin, foot: +1 or -1 torque while held |
| U/J, I/K, O/L | Front thigh, shin, foot |
| R | New episode (the next Gymnasium seed) |
| Space | Pause |
| + and - | Twice or half the speed |
| Esc | Quit |

What you see:

- **The physics:** the core steps exactly as MuJoCo's `mj_step` does, as the checker in
  `tools/parity/mujoco/halfcheetah_poc` proves. Each env step holds the action for five physics steps, in real time.
- **The view:** a side view of the x-z plane the model moves in, drawn with Gym.Rendering from the core's capsule poses.
  It is the game's own drawing, not MuJoCo's renderer (`--render` shows that one), so the state is exact but the pixels
  are not Gymnasium's.
- **The torques:** a leg segment turns red under a positive motor torque and blue under a negative one. The bars at
  the top right show all six.

## MuJoCo's window

`--render` opens the env's window (MuJoCo's, with Gymnasium's controls, DR-70): Tab cycles the cameras (the model's
"track" camera follows the runner), the mouse orbits, pans and zooms, Space pauses and H lists the rest. Gym.NET draws
MuJoCo's frames in software, every one by default: when the window falls behind real time, D turns "render every frame"
off and it skips frames instead.

## The autopilot

`policy.json` is a linear policy trained by Augmented Random Search on the managed core. It uses V2-t (Mania, Guy and
Recht 2018) with the paper's HalfCheetah settings. The training took 600 iterations, 38.4M env steps and 3 min 46 s on
12 threads. The best snapshot, at iteration 540, averaged 5,822 over Gymnasium seeds 0-4 on the engine of the time,
which followed MuJoCo 3.15.0.

It was trained only in C#, yet it scores the same in Gymnasium's HalfCheetah-v5 on MuJoCo, to the last bit:

```bash
dotnet run --project examples/HalfCheetah -c Release -- score --seeds 0-9
python examples/HalfCheetah/replay_in_gymnasium.py --seeds 0-9
```

All ten returns were identical when measured on 2026-11-11 against MuJoCo 3.15.0 (seed 0: 5832.688804393127 on both),
and again on 2026-11-12 against MuJoCo 4.0.0, the engine having followed the pin (seed 0: 5796.983819567387 on both; the
policy is the same, the physics rounds differently). The replay evaluates the
policy in the C# arithmetic order (`LinearPolicy.Act`). NumPy's matrix product sums in another order and would make the
episodes drift apart.

## Other commands

| Command | Does |
|---|---|
| `play [--seed 0] [--manual]` | The game, the autopilot driving until Tab (or the keyboard first, with `--manual`) |
| `train [--iterations 600] [--threads N] [--seed 1] [--out policy.json]` | Trains a new autopilot |
| `video FILE.mp4 [--seconds 30] [--seed 0]` | Records the autopilot at 50 frames per second (needs ffmpeg on PATH) |
| `frames FOLDER [--steps 200] [--every 20]` | Writes PNG stills |
| `score [--seeds 0-4]` | Prints exact episode returns |

Pass the command after `--` in `dotnet run`. `half_cheetah.json` is MuJoCo's compiled model as Gym.Environments embeds
it (`src/Gym.Environments/Assets/mujoco`, recorded by `tools/parity/mujoco/record_models.py`).

## How it is built

The game reaches the engine's internals (the model's sizes, the data's arrays, `mj_step` itself), so the project
compiles the engine's source in from `src/Gym.Environments/MuJoCo` rather than only referencing the package; the
compiled-in copy shadows Gym.Environments' (warning CS0436, silenced). The default and `--render` modes touch no MuJoCo
type: they go through `Gymnasium.Make`, as an app outside this repository would, with the `Gym.NET.Environments` and
`Gym.NET.Rendering.Avalonia` packages.
