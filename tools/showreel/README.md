# The showreel

Makes Gym.NET's front-page media from the environments themselves: every frame comes from an env's own `rgb_array`
render, played by a scripted policy, then composed in C# over a synthwave backdrop and encoded by ffmpeg.

| Output | Where it is used |
|---|---|
| `docs/media/gym-net-showreel.mp4`: 60 s, 1280x720, 30 fps, H.264 (about 8 MB) | The docs site's landing page, and linked from the README |
| `docs/media/gym-net-hero.gif`: a 12 s loop, 800x450, 15 fps (about 4 MB) | The top of `README.md` (GitHub and nuget.org) |
| `docs/website-src/images/envs/*.webp`, `*.png`, `envs.json` | The site's 8-bit "select your environment" gallery |
| `docs/website-src/images/showreel-poster.jpg`, `gymnet-logo.png`, `gymnet-icon.png` | The video's poster, the site's navbar logo and favicon |

```bash
dotnet run --project tools/showreel -c Release                  # everything (about 4 minutes)
dotnet run --project tools/showreel -c Release -- video         # or one output: video, hero, gallery
dotnet run --project tools/showreel -c Release -- stills --board video --at 2.6,13.5,48 --out stills
dotnet run --project tools/showreel -c Release -- parity        # only the Gymnasium comparison
```

Pass the command after `--`. Options: `--repo DIR` (default: the checkout it runs in), `--ffmpeg EXE`, `--python EXE`,
`--gymnasium DIR` (default `refs/Gymnasium`), `--work DIR` (the Python exchange files, about 1 GB while it runs),
`--no-parity`, `--gif-width N`.

## Requirements

- ffmpeg on `PATH` (checked with 7.0).
- For the parity segment (the video and `parity` only): a Python with the parity suite's reference stack
  (`tools/parity/requirements.txt`: Python 3.12, NumPy 2.4.2, pygame-ce 2.5.8, Box2D 2.3.10;
  `pwsh tools/parity/setup_venv.ps1` makes one), passed with `--python`, and the `refs/Gymnasium` submodule
  initialized. `--no-parity` renders the video without that segment.

## What it shows

The look and structure are the maintainer's choices (2026-11-11): a synthwave video in which a playful layout switches
between one, two and four envs; below it on the site, an 8-bit cartridge-select gallery; the 13 visually distinct
registered ids (the continuous and slippery twins look the same and are left out); and a parity segment. Stamps such as
`LANDED!` mark how an episode ended.

| Id | Seed | Policy | The episode |
|---|---|---|---|
| CartPole-v1 | 3 | Linear feedback around a set point that sways, so the cart glides | 500 steps, return 500 |
| MountainCar-v0 | 4 | Accelerate with the velocity | Summit in 122 steps |
| Pendulum-v1 | 5 | Energy pumping, then a PD catch at the top | Return -114.1 |
| Acrobot-v1 | 1 | Torque with the second joint's swing | Over the line in 65 steps |
| LunarLander-v3 | 7 | Gymnasium's heuristic (`LunarLanderEnv.Heuristic`) | Landed, return 248.6 |
| BipedalWalker-v3 | 0 | Gymnasium's gait (`BipedalWalkerHeuristics`) | The whole course, return 314.5 |
| BipedalWalkerHardcore-v3 | 10 | The same gait | Trips over a stump at step 157 (on purpose: GAME OVER) |
| CarRacing-v3 | 11 | Pure pursuit on the track's centre line | Return 880.7 in 1,000 steps |
| Blackjack-v1 | 40-44 | Basic strategy | Four hands won of five |
| FrozenLake-v1, FrozenLake8x8-v1 | 1, 2 | Value iteration over the env's transition table `P` | Goal in 41 and 63 steps |
| CliffWalking-v1 | 0 | Value iteration | The edge path, 13 steps |
| Taxi-v4 | 12 | Value iteration | Delivered in 16 steps, return 5 |

## The parity segment

The video's "same seed, same actions, same pixels" segment is a measurement, not a mock-up. For LunarLander-v3 and
CarRacing-v3 the tool records the actions its policy takes, runs `gymnasium_replay.py` to replay them in Gymnasium
(the real one, from `refs/Gymnasium`), replays them again in Gym.NET, and compares every frame pixel, observation
byte, reward bit and flag. Any difference stops the tool with the step and what differed. The badges print the measured
count, and both columns show exactly the frames that were compared.

Both columns come from those recorded every-step renders rather than from a live player. LunarLander ages its exhaust
particles inside `render()`, as Gymnasium does, so a panel that rendered only the displayed steps would show other
particles than the replay that was proven equal.

## Where to change things

- `Episodes.cs`: the recipes (id, seed, policy, speed, stamp, the gallery's blurb).
- `Storyboard.cs`: the two timelines: which panel is where, when, with what easing; captions and titles.
- `Synthwave.cs`, `Compositor.cs`, `PixelFont.cs`: the backdrop, the panels and post-processing, the lettering. The
  site's gallery script ports the same 5x7 font (`docs/website-src/templates/gymnet/public/main.js`).

## The docs site

`docfx docs/docfx.json` builds the site into `docs/website-src/_site`. Preview it with
`docfx docs/docfx.json --serve`; the gallery fetches `envs.json`, so it needs to be served over http, not opened as a
file. `.github/workflows/docs.yml` publishes it to GitHub Pages from master.
