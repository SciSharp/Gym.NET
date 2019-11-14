# Gym.NET

[![NuGet](https://img.shields.io/nuget/dt/Gym.NET)](https://www.nuget.org/packages/Gym.NET)<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp_badge.png" width="200" height="200" align="right" /></a>


A complete port of [openai/gym](https://github.com/openai/gym) to C#.<br>
** WORK IN PROGRESS ** 

##### openai/gym
OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. This is the gym open-source library, which gives you access to a standardized set of environments.

## Installation
```sh
### For gym's abstract classes for RL, install:
PM> Install-Package Gym.NET

### For implemented environments, install:
PM> Install-Package Gym.NET.Environments
```

## Example
The following example runs and renders cartpole-v1 environment.
```C#
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;

var cp = new CartPoleEnv(WinFormEnvViewer.Factory); //or AvaloniaEnvViewer.Factory
var done = true;
for (int i = 0; i < 100_000; i++)
{
    if (done)
    {
        var observation = cp.Reset();
        done = false;
    }
    else
    {
        var (observation, reward, _done, information) = cp.Step((i % 2)); //we switch between moving left and right
        done = _done;
        //do something with the reward and observation.
    }

    cp.Render();
    Thread.Sleep(15); //this is to prevent it from finishing instantly !
}
```

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
     - [ ] lunar_lander
   - [ ] atari
