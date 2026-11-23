# The environment docs

| | |
|---|---|
| **Status** | Planned 2026-11-22; the page designed whole on 2026-11-23. E1 (the Environments section) and E2 (the game-manual panel) are done; E3 to E8 are not started. The designs are approved in [`DESIGN_REVIEW.md`](DESIGN_REVIEW.md), round 33 (DR-102…DR-106) and round 35 (DR-111…DR-116) |
| **Asked for** | The maintainer: "We need to plan for envs docs properly. the SELECT YOUR ENV side menu needs to be improved AND a button linking to proper docfx env doc page. We will need to decide how env doc page looks and contains. including indepth, external links, original source ported from (you can find the sessions using am cli), A GIF playing 1/2 width of the doc article, api, code examples, observable, reward, alllllll other" |
| **Delivers** | 23 environment pages on the docs site, one per page of Gymnasium's environment docs; an Environments page holding the gallery beside the pages' sidebar; the gallery's NOW PLAYING panel as a game manual whose button opens the env's page |
| **Gates** | A step starts only when its designs are approved (they are: DR-102…DR-106, DR-111…DR-116). A question none of them answers goes back to the maintainer first |

---

## 1. What the reader gets

1. **The gallery explains each env and leads to its page.** The NOW PLAYING panel becomes a game manual (DR-102): HOW
   TO PLAY in game words (CONTROLS = the actions, YOU SEE = the observation, SCORE = the reward, GAME OVER = the
   episode's end), a READ THE MANUAL button that opens the env's page, and small links to its example, Gymnasium's
   page and the API. Enter opens the manual while the gallery has focus.
2. **An Environments section.** The navbar's Environments opens an Environments page (DR-106): the same
   SELECT YOUR ENVIRONMENT gallery, with the section's sidebar on its left listing the 23 pages by family. The landing
   page keeps its own gallery under the video.
3. **A page per environment, as Gymnasium has** (DR-104): 23 pages under `docs/environments/<family>/`, each
   describing its env in depth, in the floating-infobox look (DR-103) that round 35 designed whole on CartPole's and
   HalfCheetah's pages (DR-111…DR-116), written by hand and kept true by a test (DR-105). Each env's example records
   the page's media (DR-112).

## 2. The page

The maintainer chose from HTML mockups built from real facts (decision aids, not committed). Round 33 settled the
look. Round 35 settled the whole page: CartPole's in three drafts, then HalfCheetah's with the same template, which
held for MuJoCo. This section is the specification they settled.

### 2.1 Layout (DR-103, DR-111, DR-113)

- **The banner** opens the page (DR-111). It is set in the gallery's 5x7 pixel font on the arcade's dark blue: the
  env's name in large white letters with an orange shadow, then a gold line with its family and the sizes of its
  spaces (`HALF CHEETAH`, then `MUJOCO  6 ACTIONS  17 OBSERVATIONS`). It makes no parity claim. The lead paragraph
  follows it.
- **The infobox**, floated right at half the article's width. On top, the whole seeded episode the env's example
  plays (DR-112): one frame per step at the env's render fps, autoplaying, muted, looping, without controls or pointer
  input, as DR-101 has it. Below it, a caption naming the agent with its seed, steps and return, then the facts table.
- **The sections** flow beside and below it. Tables beside the box shrink to fit next to it instead of dropping below
  it (each table sits in a block formatting context, `display: flow-root`); under 768 px the box spans the article
  above the text. The quick start clears the float and spans the article's full width (DR-113), so the description
  is the only section beside the box's top.
- DocFX's conceptual layout around it: the Environments sidebar on the left, "In this article" on the right.

### 2.2 The facts table (in the infobox)

| Row | Content | Checked against (DR-105's test) |
|---|---|---|
| Ids | every id the page covers, each with its step limit | the registry: ids, `MaxEpisodeSteps` |
| Action space | the space as `ToString()` prints it (Gymnasium's repr) | `env.ActionSpace` |
| Observation space | the same | `env.ObservationSpace` |
| Reward | per step in a phrase, the registered threshold | `RewardThreshold` |
| Physics | the step: its integrator, time step and steps a second (CartPole: Euler steps of 0.02 s, float64; MuJoCo: 5 steps of 0.01 s per env step) | the env's `tau`/`dt`, `frame_skip` |
| Render modes | modes, frame size, fps | `metadata["render_modes"]`, `["render_fps"]`, a rendered frame's shape |
| Make | `Gymnasium.Make("<id>")` | the id resolves |
| Class | the env class, its options record, its typed info record if any (DR-91, DR-95) | the types exist |
| Vector env | the vector entry point where one is registered (CartPole), else "none of its own: sync or async" | `VectorEntryPoint` |
| Parity | badges saying what the family's tests prove (DR-111): "bit for bit" and "pixel for pixel"; for MuJoCo, "physics bit for bit" and an amber badge with the frames' measured gap until DR-98 closes it | the parity gate, `MUJOCO_ASSESSMENT.md` §18 |
| Gymnasium | the page's twin in Gymnasium 1.4.0's docs | the URL answers 200 at release |

### 2.3 Sections, in order (DR-113)

| # | Section | What it holds | Where it comes from |
|---|---|---|---|
| 1 | (banner and lead) | the 8-bit banner (2.1), then one paragraph: what the env is and what makes it hard | written |
| 2 | Description | the problem, where it comes from (its paper), what the agent must learn | Gymnasium's docstring, rewritten in Gym.NET's voice (it is MIT); the paper |
| 3 | Quick start | full width, below the infobox: make, reset and step in C# and Python (Gymnasium) tabs, the package to add; then "Play it, or watch the agent": the example's `--play` command and keys where it is playable, and the command that shows the infobox's agent in a window | compiled samples (DR-105), `examples/<Env>` |
| 4 | How it works | the figure (2.6), the mechanics as math, the constants or joints table with Gym.NET's and Gymnasium's names and footgun notes, what simulates the env (DR-114) | the code, the model, Gymnasium's source |
| 5 | Action space | a table: each action or each dimension, its meaning, its range (MuJoCo: each motor and its gear) | the docstring, checked against the space |
| 6 | Observation space | a table: index, meaning, joint, type, unit or min and max; notes where the bounds differ from what an episode allows | the docstring, checked against the space |
| 7 | Rewards | how each step is scored, the options that change it | the docstring and the code |
| 8 | Starting state | the reset distribution, the reset options | the docstring and the code |
| 9 | Episode end | termination conditions, truncation (the `TimeLimit` that `Make` adds) | the docstring and the registry |
| 10 | Arguments | three columns (DR-116): Gymnasium's keyword over the C# parameter, the type over the default, the effect with its footguns; the generated options record; settable physics properties | the constructor, checked by the test |
| 11 | Info | the keys, their members, their meaning, whether reset sends them; the typed info record | the info record (DR-91, DR-95) |
| 12 | Rendering | each render mode, the frame, human mode's viewer; MuJoCo's cameras (which one draws by default), `depth_array`, `rgbd_tuple` | the code |
| 13 | Episode anatomy | the infobox's episode beside an SVG chart of every step, a cursor following the video and a live readout (DR-115) | the example's trace (2.6) |
| 14 | Code examples | about six per env, each with C# and Python (Gymnasium) tabs: the make/reset/step loop, then what is particular to the env (reset options, the options record, physics, cameras, wrappers, vector envs) | compiled samples (DR-105) |
| 15 | Agents and examples | the infobox's agent (policy, seed, steps, return), the `examples/<Env>` project (`<Env>Program.cs`), its commands and keyboard keys, `--record` | `examples/`, `tests/Gym.Tests/Agents` |
| 16 | Vectorized environment | `MakeVec`: the vector entry point where one exists, sync and async | the registry |
| 17 | Variants | one subsection per other id on the page (2.4) | the registry, the code |
| 18 | Parity with Gymnasium | what is proven and by which tests: reset, dynamics, actions and errors, injected states, spaces and metadata, frames, `make`'s stack, Gymnasium's own checkers on Gym.NET's env, vector envs, goldens; where bit for bit holds (MuJoCo: the reference machine, an x64 CPU with FMA) | `tools/parity/audit/parity_matrix.py`, `tests/Gym.Tests.Parity/CLAUDE.md` §5.1 |
| 19 | Performance | steps a second, Gym.NET against Gymnasium on the reference machine | the code review's §4 where measured (nine envs, 2026-11-11), the rest measured for the page (DR-36's `Performance` utilities against Gymnasium in the parity venv) |
| 20 | Reference sources | every upstream file the env reproduces, as pinned permalinks (2.5) | the code's `Reference:` links, `THIRD-PARTY-NOTICES.md`, `src/Gym.Environments/Assets/README.md`, the sessions |
| 21 | Version history | every Gymnasium version of the env, and Gym.NET's (0.2.0, then 1.0.0 with Gymnasium 1.4.0's behaviour) | the docstring, the release notes |
| 22 | References | the papers, numbered (the env's origin, the engine, the infobox agent's algorithm) | written |
| 23 | External links | Gymnasium's page, the engine's documentation, the env's source and example on GitHub | written |
| 24 | API reference | the class, options record, info record and vector env pages of the API reference | the types |

### 2.4 Variants (DR-104)

| Page | Variant ids, as sections | Episode (DR-112) |
|---|---|---|
| CartPole | `CartPole-v0` (200 steps, threshold 195) | none of its own: it looks the same |
| Lunar Lander | `LunarLanderContinuous-v3` | none (the same look) |
| Bipedal Walker | `BipedalWalkerHardcore-v3`, anchor `#hardcore` | its own, recorded by `examples/BipedalWalkerHardcore` |
| Cliff Walking | `CliffWalkingSlippery-v1` | none (the same look) |
| Frozen Lake | `FrozenLake8x8-v1`, anchor `#8x8` | its own, recorded by `examples/FrozenLake8x8` |
| each MuJoCo env | its v4 id: what v5 changed, and that the frames are identical (measured: 0 differing pixels from the same seed, InvertedDoublePendulum's only through v4's camera, which Gymnasium's v5 leaves unused); `Pusher-v4` refuses on MuJoCo 3+, as Gymnasium's does | none |

`MountainCarContinuous-v0` keeps its own page, as in Gymnasium (it differs from MountainCar-v0 in the flag, at 0.45
instead of 0.5, and in its continuous action). It has no cartridge and no example. DR-104 gave it a loop from a
docs-only recipe in `tools/showreel/Episodes.cs` (energy pumping with the continuous force), written by the `gallery`
command without a cartridge entry in `envs.json`. Since DR-112, the pages' episodes come from the examples, so this
page needs one of two sources. **Open, to ask before E4**: a new `examples/MountainCarContinuous`, which records it as
the others do, or DR-104's showreel recipe as the one exception.

### 2.5 Reference sources

The maintainer asked for the "original source ported from". The section is called **Reference sources** and keeps
the docs' own voice (CLAUDE.md's "Documentation voice": Gym.NET's docs never say "ported"): it lists every upstream
file the env reproduces, each as a pinned permalink, with one line on what it supplies. What the code cites today:

| Source | Pin | Cited | For |
|---|---|---|---|
| Gymnasium | 1.4.0 (`8ba8c3a`) | 129 links | every env's own file (`cartpole.py`, `lunar_lander.py`, `<env>_v5.py`, ...), `car_dynamics.py`, the toy-text and classic-control utils, `mujoco_rendering.py` |
| pybox2d's Box2D | 2.3.10 (Box2D 2.3.0) | 57 links | the Box2D envs' physics: collision, contacts, joints, the world step |
| MuJoCo | 4.0.0 | 35 links | the MuJoCo envs' engine (`engine_*.c`) and renderer (`render/classic/*.c`) |
| pygame-ce | 2.5.8 | 17 links | drawing (`draw.c`, `gfxdraw`, `transform`), through `Gym.Rendering` |
| SDL_gfx | 2.0.26 (as pygame-ce ships it) | in `Gym.Rendering` | `GfxDraw`'s anti-aliased shapes |
| SDL_ttf | 2.24 | 1 link | CarRacing's score glyphs |
| Gymnasium's assets | 1.4.0 | `Assets/README.md` | the toy-text images, the 14 MuJoCo models |

Each page lists its own subset; the sessions that built each family give the narrative where the code does not (the
Agentmaster CLI, `am_inspect.py turns <session> --compact`): `b2666ae5` "core impl" (the core API, CartPole made
byte-exact), `4995aa31` "GYMNET" (classic control, toy text, Box2D in managed C#, and the per-env parity audit, its
turn 11), `d819f6c7` "GYMNET mujocu" (MuJoCo: a HalfCheetah core written from scratch, then the product engine bit for
bit with MuJoCo 4.0.0), `3830e9c7` "all rendering same" (DR-98, DR-99), `bcb6488d` "gymnet video" (the showreel's
agents). The session ids are research aids only; the pages cite public sources.

Gymnasium's MuJoCo action-space figures (`docs/environments/mujoco/action_space_figures`) are not copied: their
credits grant Gymnasium's repository alone ("he provided permission to use them on this repo"). The pages link to
Gymnasium's page, which shows them; the pages' own figures are frames that Gym.NET's renderer draws (DR-114).

### 2.6 The page's media (DR-112, DR-114, DR-115)

Each env's example records its page's media:
`dotnet run --project examples/<Env> -c Release -- --record docs/environments/<family>/media`. The example plays the
seeded episode it already plays, with the same agent, seed and kwargs, and writes:

| File | What | How |
|---|---|---|
| `<id>.episode.mp4` | the whole episode, one frame per step at the env's `render_fps`, for the infobox and Episode anatomy | `RecordVideo` with `FfmpegVideoWriter` (libx264, yuv420p, as moviepy writes); ffmpeg is needed on PATH only to record |
| `<id>.episode.json` | the seed, the steps, the return, and every step's state or its named terms, action, reward and reward terms | the example's step loop |
| `<id>.figure.png`, `<id>.figure.json` | How it works' frame, and the pixel of each point its labels mark | the env's own render. CartPole: step 125 of the episode, placed by the renderer's 125 px a metre. A MuJoCo env: the reset pose without noise (`reset_noise_scale` 0), seen from a free side camera of known placement (`default_camera_config`). Each joint's body position is projected through MuJoCo's free-camera geometry (`mjv_updateCamera`: forward and up from the azimuth and elevation, the eye at lookat minus distance times forward, the vertical field of view `vis.global.fovy` over the frame's height). Box2D: the bodies by the renderer's scale |

Measured on the mockups: CartPole's episode is 501 frames at 50 fps (40 KB); HalfCheetah's is 1,001 frames at
20 fps (976 KB), seen through the model's `track` camera as the example plays it. HalfCheetah's figure is 760x380, from
1.6 m at azimuth 90 and elevation -8. The MuJoCo episodes are the largest files; the whole set is measured when it is
recorded. Each example's `Program.cs` becomes `<Env>Program.cs` with its class `<Env>Program` (DR-112), so a page
names the file that recorded it.

## 3. The gallery (DR-102, DR-106)

- **The manual panel.** Under NOW PLAYING, the env's family in the label's gold; under the screen and its HUD, a HOW
  TO PLAY box with four rows in plain game words (CONTROLS, YOU SEE, SCORE, GAME OVER), the READ THE MANUAL button,
  and a line of links (example, Gymnasium's page, API). The blurb and the make line leave the panel (the page's
  infobox has the make call); the blurb stays as the loop's description for screen readers.
- **Data.** Each cartridge's four manual lines are written beside its blurb in `tools/showreel/Episodes.cs`
  (`GameManual`), and the `gallery` command writes them into `envs.json` (`manual`), with `docs` (the page, and the
  anchor for BipedalWalkerHardcore-v3 and FrozenLake8x8-v1), `example`, `gymnasium` and `api`. The pages take
  Gymnasium's slugs (`Page`): `docs/environments/classic_control/cart_pole.html` beside Gymnasium's
  `environments/classic_control/cart_pole/`.
- **Only written pages.** READ THE MANUAL, and Enter's hint, show only for a cartridge whose page the Environments
  sidebar lists (`docs/environments/toc.json`, read beside the manifest): the pages come family by family (E3 to E7),
  and the button never leads to a page not written yet.
- **Keys.** Enter opens the selected env's page while focus is in the gallery; the arrows keep moving the cursor.
- **Sticky floor.** The manual makes the panel taller than DR-101's tallest (516 px), so every env's panel was measured
  again (E2): the tallest is BipedalWalkerHardcore-v3's, 724 px once READ THE MANUAL shows (671 px while no page
  exists), and the floor moved from 660 px to 840 px of window height. Each manual row keeps to two lines in the
  440 px panel (a range and a number with its unit never break apart), which holds it there; a longer line raises it.
- **The Environments page.** `docs/environments/index.md` holds the same gallery: the arcade's markup moves to one
  include file used by both pages, and `main.js` already fills any `.gn-arcade` it finds. In the docs layout the shelf
  narrows to two columns beside the 440 px panel. The navbar's Environments and the sidebar's Overview open it.

## 4. Production and checks (DR-105)

- **Hand-written pages**, one Markdown file each, HTML only for the infobox, the banner, the figure's labels and the
  anatomy chart's markup; the equations in DocFX's math syntax.
- **The facts test** (`tests/Gym.Tests/Docs/EnvDocPagesTests.cs`): for each page, parse its facts table and its
  Arguments table and compare them with the code: the ids and their limits and thresholds, both spaces, the render
  modes and fps, the constructor's keywords and defaults, the info record's keys, and that the media files exist
  (2.6), with each caption's seed, steps and return equal to its `<id>.episode.json`'s. An API change that leaves a page
  wrong fails the unit gate.
- **Code samples** are written in compiled files under `tests/Gym.Tests/Docs/EnvPageSamples/` and included into the
  pages with DocFX's code-snippet syntax (`[!code-csharp[](...#region)]`), so the unit gate runs every one; the Python
  twins are run against Gymnasium in the parity venv when written.
- **The site build**: `docfx docs/docfx.json` with no new warnings; every internal link resolves; external links
  answer 200 at release; the headless checks of DR-101 extended to the manual panel (READ THE MANUAL and Enter open
  the right page and anchor) and the Environments page.

## 5. Steps

Each step: a commit on `master`, the DocFX build, the facts test where pages exist, the headless checks, the catalog;
pushed when the maintainer says so.

| Step | Work | Designs |
|---|---|---|
| E1 | The Environments section: `docs/environments/index.md` with the gallery (the shared include, `website-src/includes/arcade.md`), `toc.yml` with Overview (each family's entries come with its pages, E3 to E7), the navbar and docfx.json content entries; cartridge labels that wrap where the docs layout's shelf is narrow. **Done 2026-11-23** | DR-104, DR-106 |
| E2 | The manual panel: 24 cartridges' manual lines in `Episodes.cs`, `envs.json`'s new fields, `main.js`/`main.css`, Enter, the sticky floor re-measured. **Done 2026-11-23** | DR-102 |
| E3 | CartPole's page whole: the banner, the infobox and its CSS, the full-width quick start, the figure and the math, the code tabs and the anatomy chart's script in the `gymnet` template, every section, its samples; `examples/CartPole`'s `--record` and its media; every example's `Program.cs` renamed `<Env>Program.cs`; the facts test, the samples' test file. Reviewed by the maintainer before the rest | DR-103, DR-105, DR-111…DR-116 |
| E4 | Classic control: Acrobot, Mountain Car, Mountain Car Continuous (its episode's source asked first, 2.4), Pendulum; their examples' `--record` | DR-103…DR-105, DR-111…DR-116 |
| E5 | Toy text: Blackjack, Cliff Walking, Frozen Lake (with the 8x8 section's loop), Taxi; their examples' `--record`, and the grid figure (DR-114's reading to confirm) | |
| E6 | Box2D: Bipedal Walker (with Hardcore), Car Racing, Lunar Lander (with Continuous); their examples' `--record` with the bodies' pixels | |
| E7 | MuJoCo: the 11 pages with their v4 sections, cameras and depth rendering; their examples' `--record` with the joints projected (HalfCheetah's mockup is the model) | |
| E8 | The Performance rows not measured yet; the external links checked; the headless pass over every page and button | |

## 6. Risks

- **Drift.** Hand-written pages go stale when the code moves; the facts test catches the facts, the compiled samples
  the code, and the prose is reviewed with each env change.
- **MuJoCo frames.** DR-98 is moving the MuJoCo frames toward Gymnasium's; each page's Parity row says what holds when
  it is written, and changes with it.
- **Other CPUs.** Bit for bit holds on the reference machine (an x64 CPU with FMA). On a CPU without FMA, .NET's math
  functions differ in their last bits, and a MuJoCo episode can become another one (HalfCheetah's autopilot, seed 0:
  5,895.8 instead of 5,797.0). Record the media on the reference machine, so each caption's return is the one the
  example prints there.
- **Numbers that drift apart.** A page, its example's README and the gallery quote the same agents. HalfCheetah's
  gallery blurb says "17,629 in a thousand steps": that is the expert's mean over seeds 0-9 (the example's README).
  The episode the gallery and the page play is seed 0's, which returns 17,648.0. Each page says which episode a
  number is from.
- **New envs.** The casino games being researched (session `8f2178ce`, 2026-11-22) would each need a page, a loop and
  a cartridge; this plan's template and facts test are the checklist.
- **Shared files.** Other sessions edit `tools/showreel` (on `master`, `11fd929`'s PosterShot stills) and
  `main.css`/`main.js` (on `journey1`); re-read them before each step and merge by hand where needed.
