# The environment docs

| | |
|---|---|
| **Status** | Planned 2026-11-22; nothing implemented yet. The designs are approved in [`DESIGN_REVIEW.md`](DESIGN_REVIEW.md), round 33 (DR-102…DR-106) |
| **Asked for** | The maintainer: "We need to plan for envs docs properly. the SELECT YOUR ENV side menu needs to be improved AND a button linking to proper docfx env doc page. We will need to decide how env doc page looks and contains. including indepth, external links, original source ported from (you can find the sessions using am cli), A GIF playing 1/2 width of the doc article, api, code examples, observable, reward, alllllll other" |
| **Delivers** | 23 environment pages on the docs site, one per page of Gymnasium's environment docs; an Environments page holding the gallery beside the pages' sidebar; the gallery's NOW PLAYING panel as a game manual whose button opens the env's page |
| **Gates** | A step starts only when its designs are approved (they are: DR-102…DR-106). A question none of them answers goes back to the maintainer first |

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
   describing its env in depth, in the floating-infobox look (DR-103), written by hand and kept true by a test
   (DR-105).

## 2. The page

The maintainer chose from HTML mockups built from CartPole's real facts (decision aids, not committed); this section
is the specification they settled.

### 2.1 Layout (DR-103)

- **The infobox**, floated right at half the article's width: the env's loop on top (the gallery's full-size MP4,
  `images/envs/<id>.mp4`, autoplaying, muted, looping, without controls or pointer input, as DR-101 has it), a caption
  naming the agent that plays it with its seed, steps and return, and the facts table.
- **The sections** flow beside and below it. Tables beside the box shrink to fit next to it instead of dropping below
  it (each table sits in a block formatting context, `display: flow-root`); under 768 px the box spans the article
  above the text.
- DocFX's conceptual layout around it: the Environments sidebar on the left, "In this article" on the right.

### 2.2 The facts table (in the infobox)

| Row | Content | Checked against (DR-105's test) |
|---|---|---|
| Ids | every id the page covers, each with its step limit | the registry: ids, `MaxEpisodeSteps` |
| Action space | the space as `ToString()` prints it (Gymnasium's repr) | `env.ActionSpace` |
| Observation space | the same | `env.ObservationSpace` |
| Reward | per step in a phrase, the registered threshold | `RewardThreshold` |
| Render modes | modes, frame size, fps | `metadata["render_modes"]`, `["render_fps"]`, a rendered frame's shape |
| Make | `Gymnasium.Make("<id>")` | the id resolves |
| Class | the env class, its options record, its typed info record if any (DR-91, DR-95) | the types exist |
| Vector env | the vector entry point where one is registered (CartPole) | `VectorEntryPoint` |
| Parity | "bit for bit" and "pixel for pixel" (MuJoCo: what DR-98 has reached by then) | the parity gate |
| Gymnasium | the page's twin in Gymnasium 1.4.0's docs | the URL answers 200 at release |

### 2.3 Sections, in order

| # | Section | What it holds | Where it comes from |
|---|---|---|---|
| 1 | (lead) | one paragraph: what the env is, its family | written |
| 2 | Description | the problem, where it comes from (its paper), what the agent must learn | Gymnasium's docstring, rewritten in Gym.NET's voice (it is MIT); the paper |
| 3 | Action space | a table: each action or each dimension, its meaning, its range | the docstring, checked against the space |
| 4 | Observation space | a table: index, meaning, min, max; notes where the bounds differ from what an episode allows | the docstring, checked against the space |
| 5 | Rewards | how each step is scored, the options that change it | the docstring and the code |
| 6 | Starting state | the reset distribution, the reset options | the docstring and the code |
| 7 | Episode end | termination conditions, truncation (the `TimeLimit` that `Make` adds) | the docstring and the registry |
| 8 | Arguments | a table: Gymnasium's keyword, the C# parameter, type, default, effect; the generated options record; settable physics properties | the constructor, checked by the test |
| 9 | Info | the keys and their meaning; the typed info record | the info record (DR-91, DR-95) |
| 10 | Rendering | each render mode, the frame, human mode's viewer; MuJoCo's cameras, `depth_array`, `rgbd_tuple` | the code |
| 11 | Code examples | C# beside Python (Gymnasium): the make/reset/step loop, then what is particular to the env (reset options, the options record, wrappers) | compiled samples (DR-105) |
| 12 | Agents and examples | the loop's agent (the showreel's recipe: policy, seed, steps, return), the `examples/<Env>` project, its commands and keyboard keys | `tools/showreel/Episodes.cs`, `examples/` |
| 13 | Vectorized environment | `MakeVec`: the vector entry point where one exists, sync and async | the registry |
| 14 | Variants | one subsection per other id on the page (2.4) | the registry, the code |
| 15 | Parity with Gymnasium | what is proven and by which tests: reset, dynamics, actions and errors, injected states, spaces and metadata, frames, `make`'s stack, Gymnasium's own checkers on Gym.NET's env, vector envs, goldens | `tools/parity/audit/parity_matrix.py`, `tests/Gym.Tests.Parity/CLAUDE.md` §5.1 |
| 16 | Performance | steps a second, Gym.NET against Gymnasium on the reference machine | the code review's §4 where measured (nine envs, 2026-11-11), the rest measured for the page (DR-36's `Performance` utilities against Gymnasium in the parity venv) |
| 17 | Reference sources | every upstream file the env reproduces, as pinned permalinks (2.5) | the code's `Reference:` links, `THIRD-PARTY-NOTICES.md`, `src/Gym.Environments/Assets/README.md`, the sessions |
| 18 | Version history | Gymnasium's versions of the env; Gym.NET's (0.2.0, then 1.0.0 with Gymnasium 1.4.0's behaviour) | the docstring, the release notes |
| 19 | External links | Gymnasium's page, the papers, the env's source and example on GitHub | written |
| 20 | API reference | the class, options record, info record and vector env pages of the API reference | the types |

### 2.4 Variants (DR-104)

| Page | Variant ids, as sections | Loop |
|---|---|---|
| CartPole | `CartPole-v0` (200 steps, threshold 195) | none of its own: it looks the same |
| Lunar Lander | `LunarLanderContinuous-v3` | none (the same look) |
| Bipedal Walker | `BipedalWalkerHardcore-v3`, anchor `#hardcore` | its own (the gallery's) |
| Cliff Walking | `CliffWalkingSlippery-v1` | none (the same look) |
| Frozen Lake | `FrozenLake8x8-v1`, anchor `#8x8` | its own (the gallery's) |
| each MuJoCo env | its v4 id: what v5 changed, and that the frames are identical (measured: 0 differing pixels from the same seed, InvertedDoublePendulum's only through v4's camera, which Gymnasium's v5 leaves unused); `Pusher-v4` refuses on MuJoCo 3+, as Gymnasium's does | none |

`MountainCarContinuous-v0` keeps its own page, as in Gymnasium (it differs from MountainCar-v0 in the flag, at 0.45
instead of 0.5, and in its continuous action). It has no cartridge, so its loop is new: a recipe in
`tools/showreel/Episodes.cs` that plays it for the docs only (energy pumping with the continuous force), written by the
`gallery` command beside the others without a cartridge entry in `envs.json`.

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
Gymnasium's page, which shows them; a figure of our own, if wanted, would be drawn by Gym.NET's renderer.

## 3. The gallery (DR-102, DR-106)

- **The manual panel.** Under the screen and its HUD: a HOW TO PLAY box with four rows in plain game words (CONTROLS,
  YOU SEE, SCORE, GAME OVER), the READ THE MANUAL button, and a line of links (example, Gymnasium's page, API). The
  blurb and the make line leave the panel (the page's infobox has the make call).
- **Data.** Each cartridge's four manual lines are written beside its blurb in `tools/showreel/Episodes.cs`, and the
  `gallery` command writes them into `envs.json` (`manual`), with `docs` (the page, and the anchor for
  BipedalWalkerHardcore-v3 and FrozenLake8x8-v1), `example` and `gymnasium`.
- **Keys.** Enter opens the selected env's page while focus is in the gallery; the arrows keep moving the cursor.
- **Sticky floor.** The manual makes the panel taller than DR-101's tallest (516 px): re-measure every env's panel
  and move the 660 px floor with it.
- **The Environments page.** `docs/environments/index.md` holds the same gallery: the arcade's markup moves to one
  include file used by both pages, and `main.js` already fills any `.gn-arcade` it finds. In the docs layout the shelf
  narrows to two columns beside the 440 px panel. The navbar's Environments and the sidebar's Overview open it.

## 4. Production and checks (DR-105)

- **Hand-written pages**, one Markdown file each, HTML only for the infobox.
- **The facts test** (`tests/Gym.Tests/Docs/EnvDocPagesTests.cs`): for each page, parse its facts table and its
  Arguments table and compare them with the code: the ids and their limits and thresholds, both spaces, the render
  modes and fps, the constructor's keywords and defaults, the info record's keys, and that the loop files exist. An
  API change that leaves a page wrong fails the unit gate.
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
| E1 | The Environments section: `docs/environments/index.md` with the gallery (the shared include), `toc.yml` with the 23 pages by family, the navbar and docfx.json content entries | DR-104, DR-106 |
| E2 | The manual panel: 24 cartridges' manual lines in `Episodes.cs`, `envs.json`'s new fields, `main.js`/`main.css`, Enter, the sticky floor re-measured | DR-102 |
| E3 | CartPole's page whole (the infobox CSS, every section, its samples), the facts test, the samples' test file; reviewed by the maintainer before the rest | DR-103, DR-105 |
| E4 | Classic control: Acrobot, Mountain Car, Mountain Car Continuous (with its loop), Pendulum | DR-103…DR-105 |
| E5 | Toy text: Blackjack, Cliff Walking, Frozen Lake (with the 8x8 section's loop), Taxi | |
| E6 | Box2D: Bipedal Walker (with Hardcore), Car Racing, Lunar Lander (with Continuous) | |
| E7 | MuJoCo: the 11 pages with their v4 sections, cameras and depth rendering | |
| E8 | The Performance rows not measured yet; the external links checked; the headless pass over every page and button | |

## 6. Risks

- **Drift.** Hand-written pages go stale when the code moves; the facts test catches the facts, the compiled samples
  the code, and the prose is reviewed with each env change.
- **MuJoCo frames.** DR-98 is moving the MuJoCo frames toward Gymnasium's; each page's Parity row says what holds when
  it is written, and changes with it.
- **New envs.** The casino games being researched (session `8f2178ce`, 2026-11-22) would each need a page, a loop and
  a cartridge; this plan's template and facts test are the checklist.
- **Shared files.** Other sessions edit `tools/showreel` (on `master`, `11fd929`'s PosterShot stills) and
  `main.css`/`main.js` (on `journey1`); re-read them before each step and merge by hand where needed.
