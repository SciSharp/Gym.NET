# Examples

One project per environment the front-page showreel plays (DR-74): each plays its env with an agent, the same one the
showreel shows, through Gym.NET's public API. CartPole also learns a policy from scratch, with the cross-entropy method.

```bash
dotnet run --project examples/Hopper -c Release                    # one seeded episode, headless: steps and return
dotnet run --project examples/Hopper -c Release -- --render        # the same episode in a window
dotnet run --project examples/Taxi -c Release -- --play            # you play it, from the keyboard (where offered)
dotnet run --project examples/Taxi -c Release -- --episodes 5 --seed 100   # five episodes from seed 100, and their mean
```

Every project is self-contained, so it can be copied out on its own. Each README explains its agent, the output of the
default run, the keys, and the API it shows. The default seed is the episode the showreel plays, and every run of it on
one machine prints the same numbers ([which machine](#the-numbers)).

| Example | Agent | Default episode | Keyboard |
|---|---|---|---|
| **Classic control** | | | |
| [CartPole](CartPole) | The cross-entropy method (learned in seconds); `--controller`: hand-written state feedback around a swaying set point | Seed 3: 500 steps, return 500 | A / D |
| [MountainCar](MountainCar) | Accelerate the way the car moves (energy pumping) | Seed 4: the flag in 122 steps | ← / → |
| [Pendulum](Pendulum) | Energy pumping, then a PD controller near the top | Seed 5: 200 steps, return -114.1 | A / D |
| [Acrobot](Acrobot) | Torque the second joint the way it turns | Seed 1: over the line in 65 steps | A / D |
| **MuJoCo, control** | | | |
| [InvertedPendulum](InvertedPendulum) | Linear state feedback | Seed 0: 1,000 steps balanced | A / D |
| [InvertedDoublePendulum](InvertedDoublePendulum) | A linear policy trained by Augmented Random Search | Seed 0: 1,000 steps, return 9,359.6 | A / D |
| [Reacher](Reacher) | Inverse kinematics and a joint PD controller | Seed 0: return -3.8 | Q/A, W/S |
| [Pusher](Pusher) | Operational-space control of the fingertip | Seed 73: return -39.0 | Not offered (7 motors) |
| **Box2D** | | | |
| [LunarLander](LunarLander) | Gymnasium's heuristic | Seed 7: landed in 299 steps, return 248.6 | W / A / D |
| [BipedalWalker](BipedalWalker) | Gymnasium's gait | Seed 0: the whole course in 1,598 steps, return 314.5 | Q/A W/S U/J I/K |
| [BipedalWalkerHardcore](BipedalWalkerHardcore) | The same gait, on the hardcore course | Seed 10: falls at step 157 | Q/A W/S U/J I/K |
| [CarRacing](CarRacing) | Pure pursuit on the track's centre line | Seed 11: 1,000 steps, return 880.7 | W / A / S / D |
| **MuJoCo, locomotion** | | | |
| [HalfCheetah](HalfCheetah) | Its autopilot, a linear policy trained by Augmented Random Search; the playable HalfCheetah | Seed 0: 1,000 steps, return 5,797.0 | Its own game: Q/A W/S E/D U/J I/K O/L, Tab |
| [Hopper](Hopper) | A linear policy trained by Augmented Random Search | Seed 0: 1,000 steps, return 3,614.8 | Not offered (frames too slow) |
| [Walker2d](Walker2d) | A linear policy trained by Augmented Random Search | Seed 0: 1,000 steps, return 2,446.0 | Not offered (frames too slow) |
| [Ant](Ant) | A linear policy trained by Augmented Random Search | Seed 0: 1,000 steps, return 3,131.8 | Not offered (8 motors) |
| [Humanoid](Humanoid) | A linear policy trained by Augmented Random Search | Seed 73: falls at step 203, return 993.4 | Not offered (17 motors) |
| [HumanoidStandup](HumanoidStandup) | A linear policy trained by Augmented Random Search | Seed 0: 1,000 steps, return 124,315.4 | Not offered (17 motors) |
| [Swimmer](Swimmer) | A linear policy trained by Augmented Random Search | Seed 0: 1,000 steps, return 357.0 | Not offered (frames too slow) |
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

## The numbers

The outputs in the READMEs were measured on an x64 CPU with FMA (an i9-13900K), where Gym.NET's tests prove the envs bit
for bit against Gymnasium. The physics call .NET's `Math.Sin`, `Cos`, `Exp` and their kin, which come from the C
runtime, and a CPU without FMA gets other last bits from them. Measured on a Windows 11 desktop with a Xeon E5-2690 v2
(no FMA, no AVX2): `sin`, `cos`, `exp`, `log`, `atan2` and `tanh` differed in the last bits from the i9's, `pow` did
not. The episodes checked there, MountainCar's, Taxi's, CartPole's controller and InvertedDoublePendulum's, came out the
same. HalfCheetah's seed-0 return was 5,895.8 instead of 5,797.0, because its contact-rich, chaotic dynamics grow a
last-bit difference over 1,000 steps.

## The trained policies

InvertedDoublePendulum, Hopper, Walker2d, Ant, Humanoid, HumanoidStandup and Swimmer play a `policy.json` beside their
program: a linear policy, `a = clip(W · (s - mean) / scale, -1, 1) · high`, trained by Augmented Random Search V2-t
(Mania, Guy and Recht 2018) with `tools/showreel`'s `train` command. HalfCheetah's autopilot is the same kind, trained
by its own `train` command. Each file's `info` records how it was trained and its returns on seeds 0-4.

## Outside this repository

The projects reference Gym.NET's source projects so they always build against the current tree (`Gym.NET.sln` builds
them all, so an API change that breaks one fails the build). In your own project, reference the packages instead
(`dotnet add package Gym.NET.Environments`, and `Gym.NET.Rendering.Avalonia` for `--render` and `--play`) and drop the
`Import` of `Gym.NET.Environments.targets` from the project file: the package brings it.
