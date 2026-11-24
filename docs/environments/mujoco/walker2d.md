---
title: Walker2d
description: Make a two-legged planar robot walk forward without falling, with motors at both hips, knees and ankles. Walker2d-v5 and Walker2d-v4 in Gym.NET, on Gym.NET's own MuJoCo engine, bit for bit with MuJoCo 4.0.0, as Gymnasium 1.4.0 defines them.
---

<h1 class="visually-hidden">Walker2d</h1>
<div class="gn-env-banner">
<canvas class="gn-pixel" data-text="WALKER2D" data-scale="6" data-color="#fcfcfc" data-shadow="#e45c10"></canvas>
<canvas class="gn-pixel" data-text="MUJOCO  6 ACTIONS  17 OBSERVATIONS" data-scale="2" data-color="#f8b800"></canvas>
</div>

<p class="gn-env-lead">Make a two-legged robot walk forward as far and as fast as it can without falling, with a motor at each hip, knee and ankle. The physics is MuJoCo's, run in Gym.NET's own engine, written from scratch in C# and bit for bit with MuJoCo 4.0.0.</p>

<aside class="gn-infobox" aria-label="Walker2d at a glance">
<video src="media/Walker2d-v5.episode.mp4" poster="media/Walker2d-v5.figure.png" autoplay muted loop playsinline disablepictureinpicture disableremoteplayback aria-label="Walker2d-v5, seed 0: Farama's SAC expert walks for all 1,000 steps"></video>
<p class="gn-infobox-caption">The whole episode <code>examples/Walker2d</code> plays and records: Farama's SAC expert through the model's <code>track</code> camera, seed 0, 1,000 steps, return 6,943.4.</p>
<table>
<tr><td>Ids</td><td><code>Walker2d-v5</code> 1,000 steps<br><code>Walker2d-v4</code> 1,000 steps</td></tr>
<tr><td>Action space</td><td><code>Box(-1.0, 1.0, (6,), float32)</code></td></tr>
<tr><td>Observation space</td><td><code>Box(-inf, inf, (17,), float64)</code></td></tr>
<tr><td>Reward</td><td>1 a step for staying up, plus the forward speed, less 0.001 × the torques squared; no registered threshold</td></tr>
<tr><td>Physics</td><td>MuJoCo 4.0.0 in Gym.NET's own engine, bit for bit: 4 Runge-Kutta steps of 0.002 s per env step</td></tr>
<tr><td>Render modes</td><td><code>human</code>, <code>rgb_array</code>, <code>depth_array</code>, <code>rgbd_tuple</code> · 480×480 · 125 fps</td></tr>
<tr><td>Make</td><td><code>Gymnasium.Make(<wbr>"Walker2d-v5")</code></td></tr>
<tr><td>Class</td><td><a href="xref:Gym.Environments.Envs.MuJoCo.Walker2dEnvV5"><code>Walker2dEnvV5</code></a>, options <a href="xref:Gym.Environments.Envs.MuJoCo.Walker2dEnvV5Options"><code>Walker2dEnvV5Options</code></a>, info <a href="xref:Gym.Environments.Envs.MuJoCo.Walker2dInfoV5"><code>Walker2dInfoV5</code></a></td></tr>
<tr><td>Vector env</td><td>none of its own: <code>MakeVec</code> builds sync or async</td></tr>
<tr><td>Parity</td><td><span class="gn-badge gn-badge-ok">physics bit for bit</span><span class="gn-badge gn-badge-warn">frames: 10% of pixels differ</span><br>with Gymnasium 1.4.0 on MuJoCo 4.0.0</td></tr>
<tr><td>Gymnasium</td><td><a href="https://gymnasium.farama.org/v1.4.0/environments/mujoco/walker2d/">Walker2d</a> in Gymnasium 1.4.0's docs</td></tr>
</table>
</aside>

## Description

The walker is Hopper with a second leg: a torso on two legs of three segments each, thigh, leg and foot, and it moves
only in the vertical plane along the x axis, its legs side by side in that plane. Six hinges join the parts, at both
hips, knees and ankles, each with a motor. Each step the agent sets the six torques; the reward pays a fixed amount for
every step the walker stays upright, and for forward speed, and charges a little for torque. The episode ends as soon
as the walker falls: the torso too low or too high, or tilted too far.

The env builds on Hopper, itself from T. Erez, Y. Tassa and E. Todorov's model-predictive control of periodic tasks,
and Gymnasium keeps it as one of its standard locomotion tasks. Gym.NET runs it on its own MuJoCo engine, a C# reimplementation of MuJoCo
4.0.0 that computes every float of every step as MuJoCo does on the reference machine, so the episodes are Gymnasium's,
value for value. Gym.NET also draws MuJoCo's frames itself, in software, close to Gymnasium's but not yet equal (see
[Parity](#parity-with-gymnasium)).

## Quick start

# [C#](#tab/csharp)

[!code-csharp[](../../../tests/Gym.Tests/Docs/EnvPageSamples/Walker2dSamples.cs?range=1-4,37-43&dedent=12)]

# [Python (Gymnasium)](#tab/python)

```python
import gymnasium as gym

env = gym.make("Walker2d-v5")
# 11 numbers: the torso's height and angle, 3 joint angles, 6 velocities
observation, info = env.reset(seed=42)
# three torques, each from -1 to 1: thigh, leg, foot
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
```

---

`dotnet add package Gym.NET.Environments`; the six longer examples are in [Code examples](#code-examples).

### Watch the agent

```bash
# from a clone of the repository: the agent in the video walks in MuJoCo's window
dotnet run --project examples/Walker2d -c Release -- --render
```

Keyboard play is not offered: Gym.NET draws a frame of this model in about 0.1 s in software, while a step is 8 ms of
simulated time, so a game would run about 12 times slower than real time.

## How it works

<div class="gn-figure gn-figure-narrow">
<img src="media/Walker2d-v5.figure.png" alt="Walker2d-v5 at its reset pose without noise, seen from the side: the torso upright on two straight legs in the same place, the feet flat on the floor" width="480" height="600">
<svg viewBox="0 0 480 600" font-family="system-ui, sans-serif" role="img" aria-label="Labels: the torso, whose height and angle are the observation's items 0 and 1 and whose velocities are items 8 to 10; the hips, items 2 and 5 for the right and left angles and 11 and 14 for their speeds, driven by actions 0 and 3; the knees, items 3, 6, 12 and 15, actions 1 and 4; the ankles, items 4, 7, 13 and 16, actions 2 and 5; every motor at gear 100">
<line x1="240" y1="162.47" x2="266" y2="66" stroke="#000" stroke-width="3" stroke-linecap="round"/><line x1="240" y1="162.47" x2="266" y2="66" stroke="#ffffff" stroke-width="1.3" stroke-linecap="round"/>
<line x1="240" y1="224.15" x2="209" y2="192" stroke="#000" stroke-width="3" stroke-linecap="round"/><line x1="240" y1="224.15" x2="209" y2="192" stroke="#ffd43b" stroke-width="1.3" stroke-linecap="round"/>
<line x1="240" y1="359.69" x2="274" y2="332" stroke="#000" stroke-width="3" stroke-linecap="round"/><line x1="240" y1="359.69" x2="274" y2="332" stroke="#ffd43b" stroke-width="1.3" stroke-linecap="round"/>
<line x1="240" y1="505.22" x2="204" y2="432" stroke="#000" stroke-width="3" stroke-linecap="round"/><line x1="240" y1="505.22" x2="204" y2="432" stroke="#ffd43b" stroke-width="1.3" stroke-linecap="round"/>
<circle cx="240" cy="162.47" r="4" fill="#ffffff" stroke="#000" stroke-width="1.2"/>
<text x="270" y="70" text-anchor="start" fill="#ffffff" font-weight="700" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="13">torso: height [0], angle [1]</text>
<text x="270" y="86" text-anchor="start" fill="#ffffff" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="12">speeds [8] forward, [9] up,</text>
<text x="270" y="102" text-anchor="start" fill="#ffffff" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="12">[10] turning; x not observed</text>
<circle cx="240" cy="224.15" r="4" fill="#ffd43b" stroke="#000" stroke-width="1.2"/>
<text x="205" y="196" text-anchor="end" fill="#ffd43b" font-weight="700" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="13">hips · actions 0, 3</text>
<text x="205" y="212" text-anchor="end" fill="#ffd43b" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="12">angles [2] [5]</text>
<text x="205" y="228" text-anchor="end" fill="#ffd43b" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="12">speeds [11] [14], gear 100</text>
<circle cx="240" cy="359.69" r="4" fill="#ffd43b" stroke="#000" stroke-width="1.2"/>
<text x="278" y="336" text-anchor="start" fill="#ffd43b" font-weight="700" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="13">knees · actions 1, 4</text>
<text x="278" y="352" text-anchor="start" fill="#ffd43b" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="12">angles [3] [6]</text>
<text x="278" y="368" text-anchor="start" fill="#ffd43b" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="12">speeds [12] [15], gear 100</text>
<circle cx="240" cy="505.22" r="4" fill="#ffd43b" stroke="#000" stroke-width="1.2"/>
<text x="200" y="436" text-anchor="end" fill="#ffd43b" font-weight="700" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="13">ankles · actions 2, 5</text>
<text x="200" y="452" text-anchor="end" fill="#ffd43b" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="12">angles [4] [7]</text>
<text x="200" y="468" text-anchor="end" fill="#ffd43b" font-weight="600" paint-order="stroke" stroke="#000" stroke-width="3" stroke-linejoin="round" font-size="12">speeds [13] [16], gear 100</text>
</svg>
</div>
<p class="gn-figure-caption">The reset pose without noise (<code>reset_noise_scale</code> 0), rendered by Gym.NET with MuJoCo's free camera 2.4 m from a point 0.8 m above the floor, at azimuth 90° and elevation −5°. At the reset pose the two legs stand in the same place and the left's purple hides the right, so each dot is a joint of
both legs, projected through MuJoCo's camera geometry and labelled with both legs' items in the observation and their
motors. Gymnasium's frame of the same pose differs in 15,598 of its 288,000 pixels, by 1 or 2 of 255 levels in all but one, nineteen in twenty on the floor.</p>

MuJoCo steps the model with the fourth-order Runge-Kutta integrator every 0.002 s, and an env step holds the action
for four of those steps, 0.008 s, hence 125 steps and frames a second. Each motor applies its gear, 100, times its
control, so a control of 1 is a torque of 100 N·m; the joints stop at their ranges. The torso's own three degrees of
freedom are free: x and z slide, the angle turns.

The reward of a step is a healthy bonus while the walker is up, plus its average forward speed, less a cost on the
controls:

$$
r_t = h\,[\text{healthy}] + w_f\,\frac{x_{t+1} - x_t}{\Delta t} - w_c \sum_{j=1}^{6} a_{t,j}^2
$$

with $\Delta t = 0.008$ s, $h = 1$, $w_f = 1$ and $w_c = 0.001$. The walker is healthy while its torso's height stays
between 0.8 and 2 m and its angle within ±1 rad; the step that leaves those ranges ends the episode.

| Constant | Gym.NET | Gymnasium | Value | Effect |
|---|---|---|--:|---|
| Physics step | `Model.Opt.Timestep` | `model.opt.timestep` | 0.002 s | MuJoCo's step, Runge-Kutta 4 |
| Frame skip | `FrameSkip` | `frame_skip` | 4 | Physics steps per env step: an env step is 0.008 s |
| Healthy reward | `HealthyRewardWeight` | `_healthy_reward` | 1.0 | Paid for every step the walker is healthy |
| Forward weight | `ForwardRewardWeight` | `_forward_reward_weight` | 1.0 | Multiplies the speed in the reward |
| Control cost | `CtrlCostWeight` | `_ctrl_cost_weight` | 0.001 | Multiplies the controls squared |
| Healthy height | `HealthyZRange` | `_healthy_z_range` | 0.8 to 2.0 | The torso's height while healthy, m |
| Healthy angle | `HealthyAngleRange` | `_healthy_angle_range` | −1 to 1 | The torso's angle while healthy, rad |
| Reset noise | `ResetNoiseScale` | `_reset_noise_scale` | 0.005 | The starting state's spread |

> [!NOTE]
> The healthy bonus is worth up to 1,000 an episode on its own, about a seventh of the expert's return. Walker2d-v4 also
> paid it on the step that ends the episode by a fall, which v5 fixed (Gymnasium's issue 526), and v4's model gave the
> right foot a friction of 0.9 against the left's 1.9, where v5's gives both 1.9 (issue 477): v4's right foot slides
> more. The reward's speed is the torso's average over the step, while the observation's item 8 is MuJoCo's velocity at
> the step's end.

## Action space

`Box(-1.0, 1.0, (6,), float32)`: one control per motor, its torque the control times the motor's gear.

| Index | Motor | Gear (N·m) | Joint range |
|---|---|--:|---|
| 0 | Right hip (`thigh_joint`) | 100 | −150° to 0° |
| 1 | Right knee (`leg_joint`) | 100 | −150° to 0° |
| 2 | Right ankle (`foot_joint`) | 100 | −45° to 45° |
| 3 | Left hip (`thigh_left_joint`) | 100 | −150° to 0° |
| 4 | Left knee (`leg_left_joint`) | 100 | −150° to 0° |
| 5 | Left ankle (`foot_left_joint`) | 100 | −45° to 45° |

MuJoCo clamps each control to the motor's range of −1 to 1 before it acts; the control cost reads the action as given.

## Observation space

`Box(-inf, inf, (17,), float64)`: the joint positions and velocities, `qpos` without the torso's x, then `qvel`.

| Index | Meaning | Unit |
|---|---|---|
| 0 | The torso's height (`rootz`) | m |
| 1 | The torso's angle (`rooty`) | rad |
| 2 to 4 | The right hip's, knee's and ankle's angles | rad |
| 5 to 7 | The left hip's, knee's and ankle's angles | rad |
| 8 | The torso's forward velocity (`rootx`) | m/s |
| 9 | Its vertical velocity (`rootz`) | m/s |
| 10 | Its angular velocity (`rooty`) | rad/s |
| 11 to 16 | The six joints' angular velocities, in the order of items 2 to 7 | rad/s |

`exclude_current_positions_from_observation` false puts the torso's x first, making 18 items. The x position is
always in the info.

## Rewards

`healthy_reward` for every step the walker is healthy, plus the forward speed times `forward_reward_weight`, less
`ctrl_cost_weight` times the sum of the squared controls; the info splits the three as `reward_survive`,
`reward_forward` and `reward_ctrl`. The registry sets no threshold for Walker2d; the expert in the video makes
6,943.4.

## Starting state

The model's pose, the torso upright with its centre 1.25 m above the floor and both legs straight below it, plus
noise:
every position and every velocity gets a uniform draw from ±`reset_noise_scale` (0.005). With `reset_noise_scale` 0
every seed starts from the pose in the figure, at rest.

## Episode end

| Ends by | When |
|---|---|
| Termination | The walker is unhealthy, if `terminate_when_unhealthy` (the default): its height outside 0.8 to 2 m, or its angle beyond ±1 rad |
| Truncation | 1,000 steps: the `TimeLimit` that `Make` wraps around the env |

## Arguments

By Gymnasium's name in `Make`'s kwargs, or by the C# name in the constructor and in
<xref:Gym.Environments.Envs.MuJoCo.Walker2dEnvV5Options>.

| Argument | Type and default | Effect |
|---|---|---|
| `xml_file`<br>`xmlFile` | `string`<br>`"walker2d_v5.xml"` | The model: the bundled one by name, or a path to your own MJCF file |
| `frame_skip`<br>`frameSkip` | `int`<br>`4` | Physics steps per env step; changes the step's duration and the frame rate |
| `default_camera_config`<br>`defaultCameraConfig` | `IReadOnlyDictionary<string, object?>?`<br>`null` | The free camera's `distance`, `azimuth`, `elevation`, `lookat` or `trackbodyid`; null takes the env's own, 4 m off and 20° above, which does not follow the walker |
| `forward_reward_weight`<br>`forwardRewardWeight` | `double`<br>`1.0` | The weight of the speed in the reward |
| `ctrl_cost_weight`<br>`ctrlCostWeight` | `double`<br>`1e-3` | The weight of the control cost |
| `healthy_reward`<br>`healthyReward` | `double`<br>`1.0` | The reward for each healthy step |
| `terminate_when_unhealthy`<br>`terminateWhenUnhealthy` | `bool`<br>`true` | End the episode when the walker is unhealthy |
| `healthy_z_range`<br>`healthyZRange` | `(double Low, double High)?`<br>`null` | The torso's healthy height; null is (0.8, 2.0) |
| `healthy_angle_range`<br>`healthyAngleRange` | `(double Low, double High)?`<br>`null` | The torso's healthy angle; null is (−1, 1) |
| `reset_noise_scale`<br>`resetNoiseScale` | `double`<br>`5e-3` | The starting state's noise; 0 starts from the model's pose |
| `exclude_current_positions_from_observation`<br>`excludeCurrentPositionsFromObservation` | `bool`<br>`true` | Leave the torso's x out of the observation |
| `render_mode`<br>`renderMode` | `string?` or `RenderMode`<br>`null` | `"human"` opens MuJoCo's window; `"rgb_array"`, `"depth_array"` and `"rgbd_tuple"` make `Render()` return frames |
| `width`<br>`width` | `int`<br>`480` | The frame's width in pixels |
| `height`<br>`height` | `int`<br>`480` | The frame's height in pixels |
| `camera_id`<br>`cameraId` | `int?`<br>`null` | A camera by index (−1 the free camera); not with `camera_name` |
| `camera_name`<br>`cameraName` | `string?`<br>`null` | A camera by name: the model's `"track"` follows the body's centre of mass |
| `max_geom`<br>`maxGeom` | `int`<br>`1000` | The renderer's scene capacity |
| `visual_options`<br>`visualOptions` | `IReadOnlyDictionary<int, bool>?`<br>`null` | MuJoCo's visualization flags by index (contact points, forces, ...) |
| (Gym.NET)<br>`viewerFactory` | `EnvViewerFactory?`<br>`null` | The window for `human`: Avalonia, WinForms, or headless |

[!code-csharp[](../../../tests/Gym.Tests/Docs/EnvPageSamples/Walker2dSamples.cs#Arguments)]

## Info

<xref:Gym.Environments.Envs.MuJoCo.Walker2dInfoV5>: `x_position` and `z_distance_from_origin` (the torso's height above
its start) on every reset and step; `x_velocity`, `reward_forward`, `reward_ctrl` and `reward_survive` on every step.

## Rendering

| Mode | What you get |
|---|---|
| `rgb_array` | `Render()` returns the frame: `(480, 480, 3)` uint8 by default, `width` by `height` |
| `depth_array` | `Render()` returns MuJoCo's depth buffer, `(480, 480)` float32: 0 at the near plane, 1 at the far one, not metres |
| `rgbd_tuple` | `Render()` returns both, `RenderResult.RgbDepth`, as Python's `(rgb, depth)` |
| `human` | MuJoCo's window from the viewer factory, drawn after every reset and step, with Gymnasium's controls |
| none | `Render()` warns and returns nothing, as Gymnasium's does |

With no camera named, Gymnasium 1.4 renders through MuJoCo's free camera with the env's `default_camera_config`: for
Walker2d 4 m from a point 1.15 m above the start, 20° above it. The config names the right thigh as `trackbodyid`, but
a free camera tracks nothing, so the camera stays where it is and a walking walker leaves the frame.
`camera_name = "track"` takes the model's own camera, 3 m to the side, which follows the body's centre of mass, as the
video does. In `human` mode, Tab cycles the cameras, the mouse
orbits, pans and zooms, and H lists Gymnasium's other keys. Gym.NET draws this model's frames in software in about
0.1 s each; by default every frame is drawn, and D switches to skipping frames.

## Episode anatomy

The whole episode, step by step: the distance walked, the speed, the torso's height and angle, with a cursor following
the video. The expert walks with a stride of about 114 steps (0.9 s), at 6.4 m/s on average after its first 100 steps, 47.6 m in all; its torso's angle swings between −0.95 and 0.29 rad, within 0.05 rad of the healthy limit at its widest.

<div class="gn-anatomy" data-trace="media/Walker2d-v5.episode.json">
<div><video src="media/Walker2d-v5.episode.mp4" poster="media/Walker2d-v5.figure.png" autoplay muted loop playsinline disablepictureinpicture disableremoteplayback aria-label="Walker2d-v5, seed 0, the episode the chart follows"></video><div class="gn-anatomy-readout" aria-live="off"></div></div>
<div><div class="gn-anatomy-chart"></div><div class="gn-anatomy-legend"></div></div>
<script type="application/json" class="gn-anatomy-spec">{"label":"Walker2d-v5, seed 0: the distance walked, the speed, the torso's height and angle","bands":[{"height":70,"range":[-1,50],"ticks":[0,20,40],"series":[{"column":"x","color":"#f8b800","name":"x, the distance walked (m)"}]},{"height":70,"range":[-1.5,9],"ticks":[0,4,8],"series":[{"column":"vx","color":"#3cbcfc","name":"the forward speed (m/s)"}]},{"height":70,"range":[0.7,2.1],"ticks":[0.8,1.2,2],"series":[{"column":"z","color":"#40c057","name":"the torso's height (m); healthy from 0.8 to 2"}]},{"height":60,"range":[-60,60],"scale":57.29577951308232,"ticks":[-45,0,45],"unit":"°","series":[{"column":"angle","color":"#b8b8f8","name":"the torso's angle; healthy within ±57.3°"}]}],"readout":[{"label":"step","column":"@step"},{"label":"x","column":"x","digits":1,"unit":" m"},{"label":"height","column":"z","digits":2,"unit":" m"},{"label":"return so far","column":"@return","digits":1}]}</script>
</div>

## Code examples

### Run an episode

# [C#](#tab/csharp)

[!code-csharp[](../../../tests/Gym.Tests/Docs/EnvPageSamples/Walker2dSamples.cs#RunAnEpisode)]

# [Python (Gymnasium)](#tab/python)

```python
import gymnasium as gym

env = gym.make("Walker2d-v5")
observation, info = env.reset(seed=42)
episode_return, steps = 0.0, 0
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_return, steps = episode_return + reward, steps + 1
    if terminated or truncated:
        break
# random torques fall within about 23 steps: 4.3 on average
print(f"{steps} steps, return {episode_return:.1f}")
```

---

### The reward's terms

# [C#](#tab/csharp)

[!code-csharp[](../../../tests/Gym.Tests/Docs/EnvPageSamples/Walker2dSamples.cs#RewardTerms)]

# [Python (Gymnasium)](#tab/python)

```python
import numpy as np

env = gym.make("Walker2d-v5")
env.reset(seed=42)
action = np.full(6, 0.5, dtype=np.float32)  # every motor at half torque
observation, reward, terminated, truncated, info = env.step(action)
forward, healthy, control = info["reward_forward"], info["reward_survive"], info["reward_ctrl"]
# 1.0799 = 0.0814 forward + 1.0000 healthy - 0.0015 control
print(f"{reward:.4f} = {forward:.4f} forward + {healthy:.4f} healthy {control:.4f} control")
```

---

### Start without noise

# [C#](#tab/csharp)

[!code-csharp[](../../../tests/Gym.Tests/Docs/EnvPageSamples/Walker2dSamples.cs#ResetWithoutNoise)]

# [Python (Gymnasium)](#tab/python)

```python
env = gym.make("Walker2d-v5", reset_noise_scale=0.0)
observation, info = env.reset(seed=42)
# 1.25, True: the torso 1.25 m up, every angle and velocity 0
print(f"{observation[0]}, {bool(np.all(observation[1:] == 0))}")
```

---

### Keep going after a fall

# [C#](#tab/csharp)

[!code-csharp[](../../../tests/Gym.Tests/Docs/EnvPageSamples/Walker2dSamples.cs#KeepGoing)]

# [Python (Gymnasium)](#tab/python)

```python
env = gym.make("Walker2d-v5", terminate_when_unhealthy=False)
walker = env.unwrapped
env.reset(seed=0)
env.action_space.seed(0)
steps, healthy, episode_return = 0, 0, 0.0
while True:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    steps, episode_return = steps + 1, episode_return + reward
    healthy += 1 if walker.is_healthy else 0
    if terminated or truncated:
        break
# 1000 steps, healthy for 60, return 7.5: it falls at step 46
print(f"{steps} steps, healthy for {healthy}, return {episode_return:.1f}")
```

---

### Cameras and depth

# [C#](#tab/csharp)

[!code-csharp[](../../../tests/Gym.Tests/Docs/EnvPageSamples/Walker2dSamples.cs#CamerasAndDepth)]

# [Python (Gymnasium)](#tab/python)

```python
# the model's "track" camera follows the walker; rgbd_tuple draws colour and depth together
env = gym.make("Walker2d-v5", render_mode="rgbd_tuple", camera_name="track")
env.reset(seed=42)
rgb, depth = env.render()
print(rgb.shape, depth.shape)  # (480, 480, 3) (480, 480)
# MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one,
# not metres: the nearest point, the floor at the bottom edge, is 0.97
print(f"{depth.min():.2f}")  # 0.97
```

---

### Many walkers at once

# [C#](#tab/csharp)

[!code-csharp[](../../../tests/Gym.Tests/Docs/EnvPageSamples/Walker2dSamples.cs#ManyWalkers)]

# [Python (Gymnasium)](#tab/python)

```python
# No vector env of its own: make_vec steps 8 envs in turn ("sync")
envs = gym.make_vec("Walker2d-v5", num_envs=8, vectorization_mode="sync")
observations, infos = envs.reset(seed=42)  # 8 x 17 observations
for t in range(100):
    actions = envs.action_space.sample()  # 8 x 6 torques
    # fallen walkers reset themselves
    observations, rewards, terminations, truncations, infos = envs.step(actions)
```

---

Every C# example is a test in `tests/Gym.Tests/Docs/EnvPageSamples`, compiled and run by the unit gate; where a
comment states a number, the test checks it.

## Agents and examples

- **The video's agent** is Farama's SAC expert (Soft Actor-Critic), the agent the Farama Foundation, Gymnasium's
  maintainers, trained on Walker2d-v5 to record the Minari datasets, imported from Hugging Face
  (`farama-minari/Walker2d-v5-SAC-expert`): two dense layers of 256 with ReLU that read the 17 observations and squash
  their output into the motors' range with tanh. Seed 0 returns 6,943.4; over seeds 0 to 9 it averages 6,949.0 in
  Gym.NET, every seed walking all 1,000 steps, and its model card reports 6,956.61 ± 15.95.
- **[examples/Walker2d](https://github.com/SciSharp/Gym.NET/tree/master/examples/Walker2d)**
  ([`Walker2dProgram.cs`](https://github.com/SciSharp/Gym.NET/blob/master/examples/Walker2d/Walker2dProgram.cs))
  plays it: `dotnet run --project examples/Walker2d -c Release` plays seed 0 headless, `--seed N` another seed,
  `--episodes N` that many seeds and their mean, `--render` in MuJoCo's window.
- **This page's video and chart** are the example's own episode: `--record docs/environments/mujoco/media` records it
  with `RecordVideo` (one frame per step, through the `track` camera) and writes the step trace and the figure beside
  it.

## Vectorized environment

Walker2d registers no vector env of its own. `Gymnasium.MakeVec("Walker2d-v5", n)` builds a `SyncVectorEnv` that steps `n`
separate envs in turn, and `VectorizeMode.Async` an `AsyncVectorEnv` that steps them on worker threads; each env resets
itself after its episode ends (next-step autoreset), as Gymnasium's. Each env has its own simulation.

## Variants

<a id="v4"></a>

### Walker2d-v4

The walker of Gymnasium's previous arguments and model: no `xml_file`, `frame_skip` or `default_camera_config`, an
empty info at reset, only `x_position` and `x_velocity` in the step's info, the healthy bonus paid on every step, the
falling one included, and `walker2d.xml`, whose right foot has a friction of 0.9 where v5's has 1.9. From the same seed
and actions the two give the same observations until the right foot slides, and their rewards differ on the step that
ends the episode by a fall (measured: v4 −0.175 against v5 −1.175 on the falling step of seed 3's sampled episode). Gymnasium warns that v4 is out of date, and Gym.NET does the same. The class is
<xref:Gym.Environments.Envs.MuJoCo.Walker2dEnvV4>, its info <xref:Gym.Environments.Envs.MuJoCo.Walker2dInfoV4>.

## Parity with Gymnasium

Gymnasium 1.4.0 runs inside the test process, MuJoCo 4.0.0 under it, and both are compared value by value:

| What is compared | How | Result |
|---|---|---|
| Episodes | sampled episodes of both versions from several seeds, three in a row in one env through their falls, every observation, reward, flag and info value with its NumPy type | ✓ bit for bit |
| Arguments | Gymnasium's keyword variants (the x in the observation, other weights, ranges and noise, another frame skip) | ✓ bit for bit |
| Actions | every form of action NumPy takes or refuses, with Gymnasium's messages | ✓ same values, same messages |
| Frames | `rgb_array` and `depth_array` frames against Gymnasium's on the reference GPU, within a measured baseline | 23,701 of 230,400 pixels differ |
| Spaces, metadata, `make`'s stack, the window | both spaces; the metadata; `make`'s stacks and warnings; the human window's controls and overlay | ✓ |

The tests: [MujocoLiveParityTests](https://github.com/SciSharp/Gym.NET/blob/master/tests/Gym.Tests.Parity/Tests/Envs/MujocoLiveParityTests.cs),
[MujocoFrameLiveParityTests](https://github.com/SciSharp/Gym.NET/blob/master/tests/Gym.Tests.Parity/Tests/Envs/MujocoFrameLiveParityTests.cs),
[MujocoWindowLiveParityTests](https://github.com/SciSharp/Gym.NET/blob/master/tests/Gym.Tests.Parity/Tests/Envs/MujocoWindowLiveParityTests.cs).
Bit for bit holds on the reference machine, an x64 CPU with FMA: the engine calls the platform's math functions, whose
last bits differ on a CPU without FMA. The frames are Gym.NET's own drawing of MuJoCo's renderer: this model's differ in
about 10% of the pixels, mostly by a level or two and almost all inside the floor's checkers, where the texture's
sampling is not yet the GPU's (DR-98).

## Performance

| | Steps a second | Bytes a step |
|---|--:|--:|
| Gym.NET | 8,518 | 16,960 |
| Gymnasium | 5,453 | |

1.6× faster. Gymnasium's own `benchmark_step` on both sides (Gym.NET's `Performance.BenchmarkStep`, DR-36): random actions, a reset at each episode's end, after a 2-second warm-up; .NET 10 in Release against CPython 3.12 with MuJoCo 4.0.0's own bindings; the medians of three interleaved rounds of 5 seconds on one machine (2026-11-24). Gym.NET's engine is managed C#, MuJoCo's is native C; the step is the physics, the observation and the reward, no rendering.

## Reference sources

- [gymnasium/envs/mujoco/walker2d_v5.py](https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/walker2d_v5.py)
  and [walker2d_v4.py](https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/walker2d_v4.py),
  Gymnasium 1.4.0: the environment; [mujoco_env.py](https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/mujoco_env.py),
  the base class; [mujoco_rendering.py](https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/mujoco_rendering.py),
  the cameras and the window.
- [walker2d_v5.xml](https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/assets/walker2d_v5.xml) (and `walker2d.xml` for v4),
  the model, embedded as MuJoCo 4.0.0 compiles it (`src/Gym.Environments/Assets/mujoco`).
- [MuJoCo 4.0.0](https://github.com/google-deepmind/mujoco/tree/4.0.0/src/engine), the engine Gym.NET's `Gym.MuJoCo`
  reproduces, and its classic renderer (`src/render/classic`), which Gym.NET draws in software.

## Version history

| Version | Change |
|---|---|
| v5 | MuJoCo 2.3.3 or later; custom models through `xml_file`; `default_camera_config`, `frame_skip` and `observation_structure`; a non-empty info at reset; both feet's friction 1.9; the healthy bonus only while healthy; the reward's terms and `z_distance_from_origin` in the info |
| v4 | The `mujoco` bindings (MuJoCo 2.1.3 or later) instead of mujoco-py |
| v3 | `make` kwargs such as `xml_file` and `reset_noise_scale`; frames from the tracking camera; now raises `ImportError` in Gymnasium (mujoco-py), and in Gym.NET |
| v2 | mujoco-py 1.50; likewise refused today |
| v1 | Episodes of 1,000 steps; the reward threshold |
| v0 | The first release |
| Gym.NET 1.0.0 | New: Gymnasium 1.4.0's Walker2d-v5 and Walker2d-v4 on Gym.NET's own MuJoCo engine, bit for bit; 0.2.0 had no MuJoCo |

## References

1. T. Erez, Y. Tassa and E. Todorov, "Infinite-Horizon Model Predictive Control for Periodic Tasks with Contacts",
   *Robotics: Science and Systems VII*, 2011: the hopper Walker2d extends with a second leg. [roboticsproceedings.org](https://www.roboticsproceedings.org/rss07/p10.pdf)
2. E. Todorov, T. Erez and Y. Tassa, "MuJoCo: A physics engine for model-based control", *IROS*, 2012: the engine.
   [mujoco.org](https://mujoco.org)
3. T. Haarnoja, A. Zhou, P. Abbeel and S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
   Learning with a Stochastic Actor", *ICML*, 2018: SAC, the expert's algorithm. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

## External links

- [Walker2d on Gymnasium's site](https://gymnasium.farama.org/v1.4.0/environments/mujoco/walker2d/) (1.4.0)
- [Walker2dEnvV5.cs](https://github.com/SciSharp/Gym.NET/blob/master/src/Gym.Environments/Envs/MuJoCo/Walker2dEnvV5.cs) on GitHub
- [examples/Walker2d](https://github.com/SciSharp/Gym.NET/tree/master/examples/Walker2d)
- [farama-minari/Walker2d-v5-SAC-expert](https://huggingface.co/farama-minari/Walker2d-v5-SAC-expert), the expert on Hugging Face

## API reference

<xref:Gym.Environments.Envs.MuJoCo.Walker2dEnvV5> · <xref:Gym.Environments.Envs.MuJoCo.Walker2dEnvV5Options> · <xref:Gym.Environments.Envs.MuJoCo.Walker2dInfoV5> · <xref:Gym.Environments.Envs.MuJoCo.Walker2dEnvV4> · <xref:Gym.Environments.Envs.MuJoCo.MujocoEnv>
