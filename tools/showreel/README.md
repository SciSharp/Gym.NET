# The showreel

Makes Gym.NET's front-page media from the environments themselves: every panel shows an env's own `rgb_array` frames,
played by a scripted policy or a published expert agent, composed in C# over a 16-bit dusk beach and encoded by ffmpeg.

| Output | Where it is used |
|---|---|
| `docs/media/gym-net-showreel.mp4`: 15 lettered scenes in the order `video-order.txt` gives (78.4 s in letter order), 1280x720, 30 fps, H.264 (about 10 MB) | The docs site's landing page, and linked from the README |
| `docs/media/gym-net-hero.webp`: the same scenes in the same order, each about 60% as long (49.4 s), animated WebP, 800x450, 15 fps (about 8 MB) | The top of `README.md` (GitHub and nuget.org), and the site's fallback where video does not play |
| `docs/website-src/images/envs/*.webp`, `*.png`, `envs.json` | The site's 8-bit "select your environment" gallery |
| `docs/website-src/images/showreel-poster.jpg`, `gymnet-logo.png`, `gymnet-icon.png` | The video's poster, the site's navbar logo and favicon |

```bash
dotnet run --project tools/showreel -c Release                  # everything (about 10 minutes on 32 threads: the video 6, the README cut 2)
dotnet run --project tools/showreel -c Release -- video         # or one output: video, hero, gallery
dotnet run --project tools/showreel -c Release -- scenes        # where each scene of the video's order starts, and its length
dotnet run --project tools/showreel -c Release -- video --letters     # a preview with each scene's letter in the corner
dotnet run --project tools/showreel -c Release -- video --order "A D B I M N O" --out short.mp4
dotnet run --project tools/showreel -c Release -- video --backdrop synthwave --out synthwave.mp4   # DR-59's backdrop
dotnet run --project tools/showreel -c Release -- hero --format gif   # the README cut as a GIF instead
dotnet run --project tools/showreel -c Release -- stills --board hero --at 2,12,30 --out stills
dotnet run --project tools/showreel -c Release -- parity        # only the Gymnasium comparison
dotnet run --project tools/showreel -c Release -- train --env Hopper-v5   # retrain an ARS policy (no recipe plays them since DR-109)
```

Pass the command after `--`. Options: `--repo DIR` (default: the checkout it runs in), `--ffmpeg EXE`, `--python EXE`,
`--gymnasium DIR` (default `refs/Gymnasium`), `--work DIR` (the Python exchange files, about 1 GB while it runs),
`--no-parity`, `--width N` (the README animation's, default 800; `--gif-width` is the same option), `--format webp|gif`,
`--backdrop beach|synthwave` (default `beach`), for `video`, `hero`, `scenes` and `stills`, `--order "LETTERS"` (in
place of `video-order.txt`), for `video` and `stills --board video`, `--letters` (each scene's letter in the
bottom-right corner; without `--out` the preview goes to the temp folder, `gymnet-showreel-letters.mp4`, and never over
the published video or its poster), and for `train`, `--env ID[,ID...]` and `--minutes M`.

## Requirements

- ffmpeg on `PATH` (checked with 7.0).
- For the parity scene (the video, the README cut, `stills` and `parity`): a Python with the parity suite's reference
  stack for Box2D (`tools/parity/requirements.txt`: Python 3.12, NumPy 2.4.2, pygame-ce 2.5.8, Box2D 2.3.10;
  `pwsh tools/parity/setup_venv.ps1` makes one), passed with `--python`, and the `refs/Gymnasium` submodule
  initialized. `--no-parity` renders without that scene.
- Time: the MuJoCo frames cost 17 ms (InvertedPendulum) to 380 ms (HumanoidStandup) each, drawn in software on every
  core, so they take most of the render.
- Size: the README's image comes straight from `raw.githubusercontent.com`. GitHub's camo proxy, which refuses images
  over 5 MB, only fronts images on other hosts (checked with GitHub's markdown renderer), so the size costs load time,
  not display. Measured on DR-73's 51 s cut: animated WebP 7.2 MB, a GIF 19 MB (one palette for 24 envs dithers and
  smears the grey MuJoCo skies), AVIF 3.0 MB at 1280x720 (newer browsers only); DR-73 chose WebP.

## The scenes

The video is 15 scenes, each with a letter (DR-100, DR-107). `video-order.txt` lists the letters in the order they
play: leave one out to drop its scene, write one twice to play it twice. Each scene starts and ends on an empty stage, so
any order plays: its panels fly in after its start and are gone a moment after the next scene starts, while the next
scene's panels fly in. `scenes` prints where each one starts, and `video --letters` renders a preview with the letters
on screen. The README's animation plays the same scenes in the same order, each in a short form (DR-110).

| Letter | Scene | What plays | Video, s | README, s |
|---|---|---|---|---|
| A | Title | The logo drops in with a bounce, the pitch types on at 64 letters a second | 2.5 | 2.5 |
| B | Classic control | CartPole alone, then beside MountainCar, then four with InvertedDoublePendulum and Acrobot; the caption retypes "CLASSIC CONTROL + MUJOCO" | 11.3 | 6.4 |
| C | Reacher and Pusher | Side by side; Pusher's puck reaches the goal (`PUSHED!`) | 6.0 | 3.6 |
| D | LunarLander | The descent at its own speed, then its rest on the pad fast-forwarded to `LANDED!` | 4.0 | 3.0 |
| E | The walkers | BipedalWalker beside BipedalWalkerHardcore, which climbs stairs and crosses pits | 5.5 | 3.3 |
| F | CarRacing | The S-bends, steps 430-630 of the lap | 4.0 | 2.6 |
| G | HalfCheetah | Alone, from a standstill to a full gallop | 4.2 | 2.6 |
| H | Hopper and Walker2d | Side by side | 4.3 | 2.7 |
| I | Humanoid and HumanoidStandup | Humanoid walks; HumanoidStandup gets up off the floor and stands | 5.0 | 3.1 |
| J | Ant and Swimmer | Side by side | 4.5 | 2.8 |
| K | Toy text's four | FrozenLake8x8, CliffWalking, Taxi, Blackjack | 6.3 | 3.7 |
| L | FrozenLake | Alone, to the goal | 5.4 | 3.35 |
| M | Parity | Gymnasium beside Gym.NET, the pixel counts between them; left out with `--no-parity` | 6.8 | 3.6 |
| N | Finale | Four panels flipping through the video's 22 ids | 4.2 | 3.15 |
| O | Outro | The logo, the install line, the address | 4.4 | 3.0 |

The seconds are each scene's share, from its start to the next scene's; the last scene runs until its panels are gone.
In letter order the video is 78.4 s and the README cut 49.4 s.

The composer (`Storyboards.Video`, `Storyboards.Hero`) lays the scenes end to end and adds what spans them. A caption
that consecutive scenes share types on once: BOX2D over D, E and F; MUJOCO · LOCOMOTION over G to J; TOY TEXT over K
and L. A scene may end the shared caption early and type its own (`Scene.CaptionUntil`): B retypes "CLASSIC CONTROL +
MUJOCO" when InvertedDoublePendulum, a MuJoCo env, joins the classic four. The small header logo shows over every scene
but the title and the outro. The flashes belong to the title and the outro.

DR-107 (the maintainer, 2026-11-22) reshaped DR-100's fourteen scenes without changing their order ("dont shuffle envs
order"): Pendulum and InvertedPendulum left the video, two of its four vertical poles; InvertedDoublePendulum took
Pendulum's place in the classic four; Reacher and Pusher became a pair ("move reacher and pusher to a 2 in a scene"),
and MuJoCo's four became two pairs ("Show the humanoid and its kind 2 in scene and another scene with other 2"). The
title shrank to 2.5 s, LunarLander to 4 s ("lunar lander landing cut earlier, 4s total") and CarRacing to the 4 s of
its S-bends, the window chosen from three. Shortening a scene other than by dropping it waits for the maintainer ("Ill
say when i need", DR-100).

## What it shows

The structure is the maintainer's (DR-59, 2026-11-11): a video in which a playful layout switches between one, two and
four envs; below it on the site, an 8-bit cartridge-select gallery; one id per visually distinct env (twins such as
LunarLanderContinuous-v3 or the MuJoCo v4 ids look the same and are left out); and a parity segment. The MuJoCo envs
joined between the existing sections when they landed (DR-72), the video became lettered scenes (DR-100) and then took
its present shape (DR-107). The gallery still shows all 24 ids; the video shows 22.

The backdrop is the 16-bit dusk beach (DR-108, `DuskBeach.cs`), which the maintainer chose from eight candidates
animated live under the video's panels: pixel art drawn at 320x180 and scaled up four times, an ordered-dithered sunset
sky, the sun half-sunk in the sea with gaps rolling down it, a far skyline's blinking lights, the sun's reflection
shimmering on the sea bands, and two swaying palms. DR-59's synthwave grid stays available with `--backdrop
synthwave` (`Synthwave.cs`).

The README's animation is the same scenes' short form (DR-110, "Re-cut from the new video"), in the video's order: each
scene about 60% as long. Where a panel became too short for the moment its episode is known for, it joins mid-episode
so that moment still lands on screen: MountainCar's summit, Acrobot's crossing, Pusher's push, the toy-text goals
(`Storyboards.ShortScenes`). LunarLander plays at one and a half times its speed and fast-forwards its rest.

Stamps such as `LANDED!` mark how an episode ended.

LunarLander and the BipedalWalkers render after every step, even the steps no frame shows
(`EpisodeRecipe.RenderEveryStep`). Their `render()` changes what the next frame shows, as in Gymnasium: LunarLander ages
and removes its exhaust particles there, and BipedalWalker sweeps its lidar one ray a frame. Rendering only the shown
frames left the particles of the unshown steps unaged, so a mid-episode join started on a cloud of stale particles.

| Id | Seed | Agent | The episode |
|---|---|---|---|
| CartPole-v1 | 3 | Linear feedback around a set point that sways, so the cart glides | 500 steps, return 500 |
| MountainCar-v0 | 4 | Accelerate with the velocity | Summit in 122 steps |
| Pendulum-v1 | 5 | Energy pumping, then a PD catch at the top (the gallery only, since DR-107) | Return -114.1 |
| Acrobot-v1 | 1 | Torque with the second joint's swing | Over the line in 65 steps |
| InvertedPendulum-v5 | 0 | Farama's SAC expert (the gallery only, since DR-107) | 1,000 steps balanced |
| InvertedDoublePendulum-v5 | 0 | Farama's SAC expert | 1,000 steps, return 9,355.8 |
| Reacher-v5 | 0-3 | Farama's SAC expert | Four targets reached; the fourth's return -5.4 |
| Pusher-v5 | 7 | Farama's SAC expert | The puck in the goal, return -20.4 |
| LunarLander-v3 | 7 | Gymnasium's heuristic (`LunarLanderEnv.Heuristic`) | Landed, return 248.6 |
| BipedalWalker-v3 | 0 | Gymnasium's gait (`BipedalWalkerHeuristics`) | The whole course, return 314.5 |
| BipedalWalkerHardcore-v3 | 7 | RL Zoo's TQC agent | The whole course in 968 steps, return 308.4 |
| CarRacing-v3 | 11 | Pure pursuit on the track's centre line | Return 880.7 in 1,000 steps |
| HalfCheetah-v5 | 0 | Farama's TQC expert | 1,000 steps, return 17,648.0 |
| Hopper-v5 | 0 | Farama's SAC expert | 1,000 steps, return 4,187.0 |
| Walker2d-v5 | 0 | Farama's SAC expert | 1,000 steps, return 6,943.4 |
| Ant-v5 | 0 | Farama's SAC expert | 1,000 steps, return 5,705.8 |
| Humanoid-v5 | 0 | Farama's TQC expert, played at three-quarter speed | 1,000 steps, return 10,563.5 |
| HumanoidStandup-v5 | 0 | Farama's SAC expert | Up within a second; 1,000 steps, return 399,827.7 |
| Swimmer-v5 | 0 | Farama's PPO expert, played at twice its speed | 1,000 steps, return 361.0 |
| Blackjack-v1 | 40-44 | Basic strategy | Four hands won of five |
| FrozenLake-v1, FrozenLake8x8-v1 | 1, 2 | Value iteration over the env's transition table `P` | Goal in 41 and 63 steps |
| CliffWalking-v1 | 0 | Value iteration | The edge path, 13 steps |
| Taxi-v4 | 12 | Value iteration | Delivered in 16 steps, return 5 |

### The expert agents

The eleven MuJoCo envs and the hardcore walker play published expert agents (DR-109): the agents the Farama
Foundation, Gymnasium's maintainers, trained on these v5 envs to record the Minari datasets (Stable-Baselines3's SAC,
TQC and PPO), and RL Zoo's TQC agent for the hardcore course. The maintainer, 2026-11-22: "All of our envs must be good
or state of the art". Their home is [`tests/Gym.Tests/Agents`](../../tests/Gym.Tests/Agents), whose README lists each
one's source, licence situation and measured returns; the build copies the files next to the program as
`agents/<id>.json` and compiles that folder's reader, `ExpertPolicy.cs`, into the tool, so the showreel plays exactly
the agents the tests measure (`ExpertAgent.cs` adapts the reader to the recipes).

They replaced weaker agents: linear policies trained by Augmented Random Search (`train`, `Ars.cs`; Humanoid's fell at
step 203), and hand-written controllers for InvertedPendulum, Reacher and Pusher, which `Episodes.cs` keeps, unused, as
the record of what the experts replaced, as `policies/` keeps the ARS files. The experts run fast enough to leave a
MuJoCo model's finite floor, past which a panel shows grey void: HalfCheetah's at step 96 (17 m/s), Humanoid's near
step 350. The scenes end before those steps, Humanoid plays at three-quarter speed, and the gallery's loops stop at the
recipe's `LoopTicks` (90 and 300).

### Presentation arguments

Constructor arguments that change what the camera sees, or what an agent observes, never the dynamics:

- The seven locomotion envs name their model's own `track` camera (`camera_name="track"`, mode `trackcom`). With no
  camera named, Gymnasium 1.4 forces MuJoCo's free camera, which never follows a body, so a runner leaves the frame
  within a second; Gym.NET draws it the same way.
- InvertedDoublePendulum gets a closer `default_camera_config`: Gymnasium sets none for it, and MuJoCo's free camera
  showed it a few pixels tall in a quarter panel.
- Pusher gets one too (2.4 m away at a 135° azimuth): Gymnasium's camera stands 4 m off, which left the arm a small
  figure in a half panel.
- HumanoidStandup turns off its four extra observation groups (`include_cinert_in_observation` and the three like it):
  Farama's expert was trained seeing positions and velocities only, and its file's `make_kwargs` says so.

The site's gallery shows each env's exact `Gymnasium.Make` call, these arguments included (a dictionary argument by
name).

## The parity segment

The video's "same seed, same actions, same pixels" scene is a measurement, not a mock-up. For LunarLander-v3 and
CarRacing-v3 the tool records the actions its policy takes, runs `gymnasium_replay.py` to replay them in Gymnasium
(the real one, from `refs/Gymnasium`), replays them again in Gym.NET, and compares every frame pixel, observation
byte, reward bit and flag. Any difference stops the tool with the step and what differed. The badges print the measured
count, and both columns show exactly the frames that were compared.

Both columns come from those recorded every-step renders rather than from a live player. LunarLander ages its exhaust
particles inside `render()`, as Gymnasium does, so a panel that rendered only the displayed steps would show other
particles than the replay that was proven equal.

The scene stays on those two envs: MuJoCo's physics is bit for bit, but its frames are drawn by Gym.NET's own renderer
and differ from Gymnasium's GPU frames on a share of the pixels, by about one level (`MUJOCO_ASSESSMENT.md` §17, §18),
so "0 pixels different" would not hold for every MuJoCo env. The MuJoCo caption says what does: "MUJOCO PHYSICS · BIT
FOR BIT, IN C#".

## Where to change things

- `video-order.txt`: the scene order, of the video and the README cut.
- `Episodes.cs`: the recipes (id, seed, agent, speed, stamp, the gallery's blurb, presentation arguments, loop limits).
- `Storyboard.cs`: the video's scenes (`Storyboards.Scenes`: which panel is where, when, with what easing), the README
  cut's short forms (`Storyboards.ShortScenes`), their composer (`Compose`: captions, the header logo, the letters), and
  the titles and captions.
- `DuskBeach.cs`, `Synthwave.cs`, `Compositor.cs`, `PixelFont.cs`: the backdrops, the panels and post-processing, the
  lettering. The site's gallery script ports the same 5x7 font (`docs/website-src/templates/gymnet/public/main.js`).
- `tests/Gym.Tests/Agents` and `tools/agents/import_agents.py`: the expert agents and how they are imported;
  `ExpertAgent.cs`: their adapter here.
- `Ars.cs`, `LinearPolicy.cs`, `policies/`: the ARS trainer, its policy format, the policies it trained.

## The docs site

`docfx docs/docfx.json` builds the site into `docs/website-src/_site`, after
`dotnet build src/Gym.SourceGenerators/Gym.SourceGenerators.csproj -c Release`: DocFX loads Gym.NET's source generator
from that Release output, and without it every generated member fails to compile (CS0534). Preview it with
`docfx docs/docfx.json --serve`; the gallery fetches `envs.json`, so it needs to be served over http, not opened as a
file. `.github/workflows/docs.yml` publishes it to GitHub Pages from master.
