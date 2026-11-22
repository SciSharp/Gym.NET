# The showreel

Makes Gym.NET's front-page media from the environments themselves: every panel shows an env's own `rgb_array` frames,
played by a scripted or trained policy, composed in C# over a synthwave backdrop and encoded by ffmpeg.

| Output | Where it is used |
|---|---|
| `docs/media/gym-net-showreel.mp4`: 14 lettered scenes in the order `video-order.txt` gives (82.8 s in letter order), 1280x720, 30 fps, H.264 (about 13 MB) | The docs site's landing page, and linked from the README |
| `docs/media/gym-net-hero.webp`: the tour as the video showed it before its scenes (85 s), cut to 51 s, animated WebP, 800x450, 15 fps (about 8 MB) | The top of `README.md` (GitHub and nuget.org), and the site's fallback where video does not play |
| `docs/website-src/images/envs/*.webp`, `*.png`, `envs.json` | The site's 8-bit "select your environment" gallery |
| `docs/website-src/images/showreel-poster.jpg`, `gymnet-logo.png`, `gymnet-icon.png` | The video's poster, the site's navbar logo and favicon |

```bash
dotnet run --project tools/showreel -c Release                  # everything (about 13 minutes on 32 threads: the video 7.5, the README cut 2.5)
dotnet run --project tools/showreel -c Release -- video         # or one output: video, hero, gallery
dotnet run --project tools/showreel -c Release -- scenes        # where each scene of the video's order starts, and its length
dotnet run --project tools/showreel -c Release -- video --letters     # a preview with each scene's letter in the corner
dotnet run --project tools/showreel -c Release -- video --order "A D B I L M N" --out short.mp4
dotnet run --project tools/showreel -c Release -- hero --format gif   # the README cut as a GIF instead (19 MB)
dotnet run --project tools/showreel -c Release -- stills --board hero --at 2.6,17.5,40 --out stills
dotnet run --project tools/showreel -c Release -- parity        # only the Gymnasium comparison
dotnet run --project tools/showreel -c Release -- train --env Hopper-v5   # retrain a committed MuJoCo policy
```

Pass the command after `--`. Options: `--repo DIR` (default: the checkout it runs in), `--ffmpeg EXE`, `--python EXE`,
`--gymnasium DIR` (default `refs/Gymnasium`), `--work DIR` (the Python exchange files, about 1 GB while it runs),
`--no-parity`, `--width N` (the README animation's, default 800; `--gif-width` is the same option), `--format webp|gif`,
for `video`, `scenes` and `stills --board video`, `--order "LETTERS"` (in place of `video-order.txt`) and `--letters`
(each scene's letter in the bottom-right corner; without `--out` the preview goes to the temp folder,
`gymnet-showreel-letters.mp4`, and never over the published video or its poster), and for `train`, `--env ID[,ID...]`
and `--minutes M`.

## Requirements

- ffmpeg on `PATH` (checked with 7.0).
- For the parity segment (the video, the README cut, `stills` and `parity`): a Python with the parity suite's reference
  stack (`tools/parity/requirements.txt`: Python 3.12, NumPy 2.4.2, pygame-ce 2.5.8, Box2D 2.3.10;
  `pwsh tools/parity/setup_venv.ps1` makes one), passed with `--python`, and the `refs/Gymnasium` submodule
  initialized. `--no-parity` renders without that segment.
- Time: the MuJoCo frames cost 17 ms (InvertedPendulum) to 380 ms (HumanoidStandup) each, drawn in software on every
  core, so they take most of the render.
- Size: the README's image comes straight from `raw.githubusercontent.com`. GitHub's camo proxy, which refuses images
  over 5 MB, only fronts images on other hosts (checked with GitHub's markdown renderer), so the size costs load time,
  not display. Measured on the 51 s cut: animated WebP 7.2 MB, the GIF 19 MB (one palette for 24 envs dithers and
  smears the grey MuJoCo skies), AVIF 3.0 MB at 1280x720 (newer browsers only); DR-73 chose WebP.

## The scenes

The video is 14 scenes, each with a letter (DR-100). `video-order.txt` lists the letters in the order they play: leave
one out to drop its scene, write one twice to play it twice. Each scene starts and ends on an empty stage, so any order
plays: its panels fly in after its start and are gone a moment after the next scene starts, while the next scene's
panels fly in. `scenes` prints where each one starts, and `video --letters` renders a preview with the letters on
screen.

| Letter | Scene | What plays | Seconds |
|---|---|---|---|
| A | Title | The logo drops in with a bounce, the pitch types on | 4.0 |
| B | Classic control | CartPole alone, then beside MountainCar, then four with Pendulum and Acrobot | 11.3 |
| C | MuJoCo control | CartPole beside InvertedPendulum, then four: InvertedPendulum, InvertedDoublePendulum, Reacher, Pusher | 8.7 |
| D | LunarLander | Alone: the descent and `LANDED!` | 7.0 |
| E | The walkers | BipedalWalker beside BipedalWalkerHardcore, which trips over its first stump | 5.5 |
| F | CarRacing | Alone, pure pursuit | 5.6 |
| G | HalfCheetah | Alone, the autopilot already running (joined at step 40) | 4.2 |
| H | Hopper and Walker2d | Side by side | 4.3 |
| I | MuJoCo's four | Ant, Humanoid, Swimmer, HumanoidStandup | 5.1 |
| J | Toy text's four | FrozenLake8x8, CliffWalking, Taxi, Blackjack | 6.3 |
| K | FrozenLake | Alone, to the goal | 5.4 |
| L | Parity | Gymnasium beside Gym.NET, the pixel counts between them; left out with `--no-parity` | 6.8 |
| M | Finale | Four panels flipping through all 24 ids | 4.2 |
| N | Outro | The logo, the install line, the address | 4.4 |

The seconds are each scene's share of the video, from its start to the next scene's; the last scene runs until its
panels are gone. In letter order the video is 82.8 s.

The composer (`Storyboards.Video`) lays the scenes end to end and adds what spans them. A caption that consecutive
scenes share types on once: BOX2D over D, E and F; MUJOCO · LOCOMOTION over G, H and I; TOY TEXT over J and K. The
small header logo shows over every scene but the title and the outro. The flashes belong to the title and the outro.

Two hand-overs of the 85 s video crossed from one section into the next and could not survive reordering. CartPole now
enters scene C afresh beside InvertedPendulum, 0.3 s after it, so that after B it never meets B's CartPole leaving.
CarRacing leaves at the end of F instead of driving beside HalfCheetah, and HalfCheetah has scene G to itself.

Shortening a scene other than by dropping it is left for later (the maintainer, 2026-11-22: "Ill say when i need").

## What it shows

The look and structure are the maintainer's choices (DR-59, 2026-11-11): a synthwave video in which a playful layout
switches between one, two and four envs; below it on the site, an 8-bit cartridge-select gallery; one id per visually
distinct env (twins such as LunarLanderContinuous-v3 or the MuJoCo v4 ids look the same and are left out); and a parity
segment. The MuJoCo envs were inserted between the existing sections when they landed (DR-72, 2026-11-12: "insert
between existing envs the new envs"):

intro, classic control, MuJoCo's control tasks (CartPole stays on beside InvertedPendulum, MuJoCo's cart-pole), Box2D,
MuJoCo's locomotion (CarRacing stays on beside HalfCheetah), toy text, the parity segment, a finale flipping through all
24 ids, the outro.

Since DR-100 (2026-11-22: "Kindly split into segments lettered so i can reorder and shorten time") the video is 14
lettered scenes, played in the order `video-order.txt` gives ([the scenes](#the-scenes)); the two hand-overs above
became scenes of their own.

The README shows the 85 s tour, as it was before the scenes, cut to 51 s (DR-73, 2026-11-12: "the gif is not great,
redo the showreel but -2/5 of the time of the mp4"; DR-100 left it as it is): every section and env, each on screen
about 60% as long, in place of DR-59's 12 s loop of seven envs. Where a panel became too short for the moment its
episode is known for, it joins mid-episode so that moment still lands on screen: MountainCar's summit, Acrobot's
crossing, the toy-text goals (`Storyboards.Hero`). LunarLander's story is its descent, which ends at step 117 of 299,
so it plays from its start at twice its speed instead.

Stamps such as `LANDED!` mark how an episode ended.

LunarLander and the BipedalWalkers render after every step, even the steps no frame shows
(`EpisodeRecipe.RenderEveryStep`). Their `render()` changes what the next frame shows, as in Gymnasium: LunarLander ages
and removes its exhaust particles there, and BipedalWalker sweeps its lidar one ray a frame. Rendering only the shown
frames left the particles of the unshown steps unaged, so a mid-episode join started on a cloud of stale particles.

| Id | Seed | Policy | The episode |
|---|---|---|---|
| CartPole-v1 | 3 | Linear feedback around a set point that sways, so the cart glides | 500 steps, return 500 |
| MountainCar-v0 | 4 | Accelerate with the velocity | Summit in 122 steps |
| Pendulum-v1 | 5 | Energy pumping, then a PD catch at the top | Return -114.1 |
| Acrobot-v1 | 1 | Torque with the second joint's swing | Over the line in 65 steps |
| InvertedPendulum-v5 | 0 | Linear state feedback found by random search | 1,000 steps, the cart centred (mean \|x\| 0.003 on seeds 0-9) |
| InvertedDoublePendulum-v5 | 0 | Linear policy trained by ARS | 1,000 steps, return 9,359.6 |
| Reacher-v5 | 0-3 | Analytic inverse kinematics and a joint-space PD | Mean return -4.14 a target (seeds 0-19) |
| Pusher-v5 | 73 | Operational-space control of the fingertip (numerical Jacobian) | Sweeps to the puck; it ends 0.145 from the goal, return -39.0 |
| LunarLander-v3 | 7 | Gymnasium's heuristic (`LunarLanderEnv.Heuristic`) | Landed, return 248.6 |
| BipedalWalker-v3 | 0 | Gymnasium's gait (`BipedalWalkerHeuristics`) | The whole course, return 314.5 |
| BipedalWalkerHardcore-v3 | 10 | The same gait | Trips over a stump at step 157 (on purpose: GAME OVER) |
| CarRacing-v3 | 11 | Pure pursuit on the track's centre line | Return 880.7 in 1,000 steps |
| HalfCheetah-v5 | 0 | The ARS autopilot of `examples/HalfCheetah`, trained on the same physics | 1,000 steps, return 5,797.0; Gymnasium's is the same to the last bit (that folder's `replay_in_gymnasium.py`) |
| Hopper-v5 | 0 | Linear policy trained by ARS | 1,000 steps, return 3,614.8 |
| Walker2d-v5 | 0 | Linear policy trained by ARS | 1,000 steps, return 2,446.0 |
| Ant-v5 | 0 | Linear policy trained by ARS | 1,000 steps, return 3,131.8 |
| Humanoid-v5 | 73 | Linear policy trained by ARS, played at three-quarter speed | Falls at step 203, return 993.4 (the longest of seeds 0-199; the policy averages about 120 steps) |
| HumanoidStandup-v5 | 0 | Linear policy trained by ARS | 1,000 steps, return 124,315.4 |
| Swimmer-v5 | 0 | Linear policy trained by ARS, played at twice its speed | 1,000 steps, return 357.0 |
| Blackjack-v1 | 40-44 | Basic strategy | Four hands won of five |
| FrozenLake-v1, FrozenLake8x8-v1 | 1, 2 | Value iteration over the env's transition table `P` | Goal in 41 and 63 steps |
| CliffWalking-v1 | 0 | Value iteration | The edge path, 13 steps |
| Taxi-v4 | 12 | Value iteration | Delivered in 16 steps, return 5 |

### The MuJoCo policies

`train` runs Augmented Random Search V2-t (Mania, Guy and Recht 2018; `Ars.cs`) through `Gymnasium.Make`, with the
paper's hyperparameters where it reports the env and a time budget per env (`Ars.For`). Locomotion trains with the
survival bonus subtracted, as in the paper, except Humanoid: without its bonus the policy still fell within about 100
steps after 5M env steps (the paper trains on the order of 100M), and keeping it teaches standing first. The best
snapshot is chosen by the true return on seeds 0-4, and each
`policies/<id>.json` records its iteration, steps, settings and evaluation returns. Three envs are scripted instead,
each because its trained policy looked wrong on screen: InvertedPendulum's drifted to the end of its rail, Reacher's
plateaued at -9, and Pusher's barely moved the arm.

### Presentation arguments

Two constructor arguments change what the camera sees, never the dynamics:

- The seven locomotion envs name their model's own `track` camera (`camera_name="track"`, mode `trackcom`). With no
  camera named, Gymnasium 1.4 forces MuJoCo's free camera, which never follows a body, so a runner leaves the frame
  within a second; Gym.NET draws it the same way.
- InvertedDoublePendulum gets a closer `default_camera_config`: Gymnasium sets none for it, and MuJoCo's free camera
  showed it a few pixels tall in a quarter panel.

The site's gallery shows each env's exact `Gymnasium.Make` call, these arguments included.

## The parity segment

The video's "same seed, same actions, same pixels" segment is a measurement, not a mock-up. For LunarLander-v3 and
CarRacing-v3 the tool records the actions its policy takes, runs `gymnasium_replay.py` to replay them in Gymnasium
(the real one, from `refs/Gymnasium`), replays them again in Gym.NET, and compares every frame pixel, observation
byte, reward bit and flag. Any difference stops the tool with the step and what differed. The badges print the measured
count, and both columns show exactly the frames that were compared.

Both columns come from those recorded every-step renders rather than from a live player. LunarLander ages its exhaust
particles inside `render()`, as Gymnasium does, so a panel that rendered only the displayed steps would show other
particles than the replay that was proven equal.

The segment stays on those two envs: MuJoCo's physics is bit for bit, but its frames are drawn by Gym.NET's own renderer
and differ from Gymnasium's GPU frames on 0.07% to 16% of the pixels, by about one level (`MUJOCO_ASSESSMENT.md` §17),
so "0 pixels different" would not hold. The MuJoCo caption says what does: "MUJOCO PHYSICS · BIT FOR BIT, IN C#".

## Where to change things

- `Episodes.cs`: the recipes (id, seed, policy, speed, stamp, the gallery's blurb, presentation arguments).
- `video-order.txt`: the video's scene order.
- `Storyboard.cs`: the video's scenes (`Storyboards.Scenes`: which panel is where, when, with what easing) and their
  composer (`Storyboards.Video`: captions, the header logo, the letters), and the README cut's timeline
  (`Storyboards.Hero`); captions and titles.
- `Synthwave.cs`, `Compositor.cs`, `PixelFont.cs`: the backdrop, the panels and post-processing, the lettering. The
  site's gallery script ports the same 5x7 font (`docs/website-src/templates/gymnet/public/main.js`).
- `Ars.cs`, `LinearPolicy.cs`, `policies/`: the MuJoCo trainer, the policy format, the committed policies.

## The docs site

`docfx docs/docfx.json` builds the site into `docs/website-src/_site`, after
`dotnet build src/Gym.SourceGenerators/Gym.SourceGenerators.csproj -c Release`: DocFX loads Gym.NET's source generator
from that Release output, and without it every generated member fails to compile (CS0534). Preview it with
`docfx docs/docfx.json --serve`; the gallery fetches `envs.json`, so it needs to be served over http, not opened as a
file. `.github/workflows/docs.yml` publishes it to GitHub Pages from master.
