# Gym.NET

[![NuGet](https://img.shields.io/nuget/dt/Gym.NET)](https://www.nuget.org/packages/Gym.NET)<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp_badge.png" width="200" height="200" align="right" /></a>


A port of [openai/gym](https://github.com/openai/gym) to C#.<br>


##### openai/gym
OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. This is the gym open-source library, which gives you access to a standardized set of environments.

## Installation
```sh
### For gym's abstract classes for RL, install:
PM> Install-Package Gym.NET

### For implemented environments, install:
PM> Install-Package Gym.NET.Environments
PM> Install-Package Gym.NET.Rendering.Avalonia
PM> Install-Package Gym.NET.Rendering.WinForm
```

## Example
The following example runs and renders the CartPole-v1 environment with Gymnasium's API: `Reset(seed)` returns
`(observation, info)`, `Step` returns `(observation, reward, terminated, truncated, info)`, and the render mode is chosen
at construction.
```C#
using NumSharp;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering.WinForm;
using Gym.Wrappers;

// "human" renders every reset and step into the window, paced at metadata["render_fps"] (50 fps);
// "rgb_array" would instead make Render() return each frame as a RenderResult.Rgb.
using var cp = new TimeLimit<NDArray, long>(
    new CartPoleEnv(renderMode: "human", viewerFactory: WinFormEnvViewer.Factory), //or AvaloniaEnvViewer.Factory
    maxEpisodeSteps: 500);

var (observation, info) = cp.Reset(seed: 42);
for (int i = 0; i < 100_000; i++)
{
    var (next, reward, terminated, truncated, stepInfo) = cp.Step(i % 2); //we switch between moving left and right
    //do something with the reward and observation.
    if (terminated || truncated)
        (observation, info) = cp.Reset();
    else
        observation = next;
}
```
Code written for the pre-0.26 API (`Reset()` returning the observation, the 4-tuple `Step`, `Seed(s)`) keeps working for
now through `Gym.Legacy.LegacyEnvAdapter`.

## Roadmap
- Implement [Spaces](https://github.com/openai/gym/tree/master/gym/spaces)
  - [X] `Space` (base class)
  - [X] `Box`
  - [X] `Discrete`
  - [ ] `multi.*.py`

- Implement [Env](https://github.com/openai/gym/blob/master/gym/core.py) base classes
  - [X] Env(object)
  - [ ] GoalEnv(Env)

 - Implement environments<br>
    To run an environment, see [Gym.Tests](./tests/Gym.Tests/)
   - [X] Convert Gym.Environments to a net-standard project.
   - [ ] classics
     - [X] CartPole-v1 
       - [ ] Compare visually against python's version
     - [ ] walker2d_v3
     - [ ] acrobot				
     - [ ] continuous_mountain_car
     - [ ] mountain_car		
     - [ ] pendulum			
     - [ ] rendering
   - [ ] Mujco
     - [ ] ant_v3						
     - [ ] half_cheetah_v3			
     - [ ] hopper_v3					
     - [ ] humanoid_v3				
     - [ ] humanoidstandup			
     - [ ] inverted_double_pendulum	
     - [ ] inverted_pendulum			
     - [ ] mujoco_env					
     - [ ] pusher						
     - [ ] reacher					
     - [ ] striker					
     - [ ] swimmer_v3					
     - [ ] thrower				
   - [ ] box2d
     - [ ] bipedal_walker
     - [ ] car_dynamics
     - [ ] car_racing
     - [X] lunar_lander
   - [ ] atari
