# Examples

One project per environment the front-page showreel plays (DR-74), and one more, MountainCarContinuous, whose docs page
needed an episode of its own (DR-124): each plays its env with an agent, the showreel's where it has one, through
Gym.NET's public API. CartPole also learns a policy from scratch, with the cross-entropy method.

```bash
dotnet run --project examples/Hopper -c Release                    # one seeded episode, headless: steps and return
dotnet run --project examples/Hopper -c Release -- --render        # the same episode in a window
dotnet run --project examples/Taxi -c Release -- --play            # you play it, from the keyboard (where offered)
dotnet run --project examples/Taxi -c Release -- --episodes 5 --seed 100   # five episodes from seed 100, and their mean
```

Every project is self-contained, so it can be copied out on its own. Each README explains its agent, the output of the
default run, the keys, and the API it shows. The default seed is the episode the showreel plays, and every run of it on
one machine prints the same numbers ([which machine](#the-numbers)).

Each episode is one `[NDScoped]` method (DR-80): NumSharp.Build, which every project references, releases every array
the episode creates (the results, the actions) when the method returns, the way the guides release what an env
returns. The env is built and disposed inside that method, the guides' rule for an env in a scoped method.

| Example | Agent | Default episode | Keyboard |
|---|---|---|---|
| **Classic control** | | | |
| [CartPole](CartPole) | The cross-entropy method (learned in seconds); `--controller`: hand-written state feedback around a swaying set point | Seed 3: 500 steps, return 500 | A / D |
| [MountainCar](MountainCar) | Accelerate the way the car moves (energy pumping) | Seed 4: the flag in 122 steps | ← / → |
| [MountainCarContinuous](MountainCarContinuous) | Full force the way the car moves, left from rest (energy pumping); `--force F` | Seed 4: the flag in 78 steps, return 92.2 | ← / → |
| [Pendulum](Pendulum) | Energy pumping, then a PD controller near the top | Seed 5: 200 steps, return -114.1 | A / D |
| [Acrobot](Acrobot) | Torque the second joint the way it turns | Seed 1: over the line in 65 steps | A / D |
| **MuJoCo, control** | | | |
| [InvertedPendulum](InvertedPendulum) | Farama's SAC expert | Seed 0: 1,000 steps balanced | A / D |
| [InvertedDoublePendulum](InvertedDoublePendulum) | Farama's SAC expert | Seed 0: 1,000 steps, return 9,355.8 | A / D |
| [Reacher](Reacher) | Farama's SAC expert | Seed 0: return -2.8 | Q/A, W/S |
| [Pusher](Pusher) | Farama's SAC expert | Seed 7: the puck in the goal, return -20.4 | Not offered (7 motors) |
| **Box2D** | | | |
| [LunarLander](LunarLander) | Gymnasium's heuristic | Seed 7: landed in 299 steps, return 248.6 | W / A / D |
| [BipedalWalker](BipedalWalker) | Gymnasium's gait | Seed 0: the whole course in 1,598 steps, return 314.5 | Q/A W/S U/J I/K |
| [BipedalWalkerHardcore](BipedalWalkerHardcore) | RL Zoo's TQC agent | Seed 7: the whole course in 968 steps, return 308.4 | Q/A W/S U/J I/K |
| [CarRacing](CarRacing) | Pure pursuit on the track's centre line | Seed 11: 1,000 steps, return 880.7 | W / A / S / D |
| **MuJoCo, locomotion** | | | |
| [HalfCheetah](HalfCheetah) | Farama's TQC expert; `--autopilot`: its linear autopilot, trained by Augmented Random Search, which also drives the playable HalfCheetah | Seed 0: 1,000 steps, return 17,648.0 (autopilot 5,797.0) | Its own game: Q/A W/S E/D U/J I/K O/L, Tab |
| [Hopper](Hopper) | Farama's SAC expert | Seed 0: 1,000 steps, return 4,187.0 | Not offered (frames too slow) |
| [Walker2d](Walker2d) | Farama's SAC expert | Seed 0: 1,000 steps, return 6,943.4 | Not offered (frames too slow) |
| [Ant](Ant) | Farama's SAC expert | Seed 0: 1,000 steps, return 5,705.8 | Not offered (8 motors) |
| [Humanoid](Humanoid) | Farama's TQC expert | Seed 0: 1,000 steps, return 10,563.5 | Not offered (17 motors) |
| [HumanoidStandup](HumanoidStandup) | Farama's SAC expert | Seed 0: up within a second, return 399,827.7 | Not offered (17 motors) |
| [Swimmer](Swimmer) | Farama's PPO expert | Seed 0: 1,000 steps, return 361.0 | Not offered (frames too slow) |
| **Toy text** | | | |
| [FrozenLake](FrozenLake) | Value iteration over the env's transition table | Seed 1: the goal in 41 steps | Arrows |
| [FrozenLake8x8](FrozenLake8x8) | Value iteration over the env's transition table | Seed 2: the goal in 63 steps | Arrows |
| [CliffWalking](CliffWalking) | Value iteration over the env's transition table | Seed 0: the goal in 13 steps | Arrows |
| [Taxi](Taxi) | Value iteration over the env's transition table | Seed 12: delivered in 16 steps, return 5 | Arrows, P, D |
| [Blackjack](Blackjack) | Basic strategy | Seed 40: won (hands 40-44: four won of five) | H / S |

## Watching and playing

`--render` makes the env with `render_mode = "human"` and Avalonia's viewer (`Gymnasium.HumanViewerFactory =
AvaloniaEnvViewer.Factory`). The MuJoCo envs open MuJoCo's window with Gymnasium's controls (DR-70): Tab cycles the
cameras, the mouse orbits, pans and zooms, H lists the rest. Gym.NET draws MuJoCo's frames in software, which costs 20
ms (InvertedPendulum) to almost 0.4 s (HumanoidStandup) a frame. By default every frame is drawn, so a heavy scene plays
in slow motion; press D and the window skips frames instead.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`, DR-44), with a key map per env. A motor of
a walker or an arm takes a key pair, and the map lists every combination of the keys held, since Gymnasium's `play`
looks up the exact set. The toy texts are turn-based: they step only while a key is held. Keyboard play is not offered
for Pusher, Ant, Humanoid and HumanoidStandup, whose 7 to 17 motors would need 14 to 34 keys. It is not offered for
Hopper, Walker2d and Swimmer either: `play` draws a frame per step, and their frames take 0.1 to 0.3 s against 8 to 40
ms of simulated time per step, which would run the game 7 to 37 times slower than real time. HalfCheetah has its own
game, which draws a cheap side view in real time.

## Recording a docs page's media

An example whose env has a page on the docs site records that page's media (DR-112): `--record <folder>` plays its
seeded episode once more and writes `<id>.episode.mp4` (the whole episode, one frame per step at the env's render fps,
through `RecordVideo` and `FfmpegVideoWriter`), `<id>.episode.json` (every step's state, action and reward, for the
page's anatomy chart) and `<id>.figure.png` with `<id>.figure.json` (the frame How it works annotates, and the pixels
its labels point at). Classic control's five record, into `docs/environments/classic_control/media`, and toy text's
five, into `docs/environments/toy_text/media`; a grid world's (FrozenLake, FrozenLake8x8, CliffWalking, Taxi) also
writes `<id>.board.png`, the reset's frame, which the page's anatomy draws the episode's path on (DR-131), and
Blackjack records the page's own hand, seed 1,574,758, rather than its default seed (DR-128):

```bash
dotnet run --project examples/Acrobot -c Release -- --record docs/environments/classic_control/media
dotnet run --project examples/CliffWalking -c Release -- --record docs/environments/toy_text/media
```

ffmpeg must be on PATH to record; every other mode runs without it. Each README lists its files.

## The numbers

The outputs in the READMEs were measured on an x64 CPU with FMA (an i9-13900K), where Gym.NET's tests prove the envs bit
for bit against Gymnasium. The physics call .NET's `Math.Sin`, `Cos`, `Exp` and their kin, which come from the C
runtime, and a CPU without FMA gets other last bits from them. Measured on a Windows 11 desktop with a Xeon E5-2690 v2
(no FMA, no AVX2): `sin`, `cos`, `exp`, `log`, `atan2` and `tanh` differed in the last bits from the i9's, `pow` did
not. The episodes checked there, MountainCar's, Taxi's, CartPole's controller and InvertedDoublePendulum's, came out the
same. HalfCheetah's seed-0 return was 5,895.8 instead of 5,797.0, because its contact-rich, chaotic dynamics grow a
last-bit difference over 1,000 steps.

## The expert agents

The eleven MuJoCo examples and BipedalWalkerHardcore play published expert agents (DR-109): the agents the Farama
Foundation, Gymnasium's maintainers, trained on these v5 envs to record the Minari datasets (Stable-Baselines3's SAC,
TQC and PPO), and RL Zoo's TQC agent for the hardcore course. Each sits beside its program as `policy.json`
(HalfCheetah's as `expert.json`), a copy of the agent in [`tests/Gym.Tests/Agents`](../tests/Gym.Tests/Agents), where
the tests measure every one, and each program carries its own copy of the reader, `ExpertPolicy`. The networks are two
dense layers of 256 (Swimmer's 64, the hardcore walker's 400 and 300); `tools/agents/import_agents.py` writes them from
their pinned Hugging Face revisions. The model repositories state no licence; their training code is MIT. Gym.NET commits
the weights with that attribution, on the maintainer's decision.

They replaced weaker agents: linear policies trained by Augmented Random Search, which kept Humanoid up for about 200
steps, and hand-written controllers for InvertedPendulum, Reacher and Pusher. HalfCheetah keeps its linear autopilot,
trained by its own `train` command, for `--autopilot` and the playable game: Gymnasium reproduces its returns to the
last bit.

## Outside this repository

The projects reference Gym.NET's source projects so they always build against the current tree (`Gym.NET.sln` builds
them all, so an API change that breaks one fails the build). In your own project, reference the packages instead
(`dotnet add package Gym.NET.Environments`, and `Gym.NET.Rendering.Avalonia` for `--render` and `--play`) and drop the
`Import` of `Gym.NET.Environments.targets` from the project file: the package brings it.
