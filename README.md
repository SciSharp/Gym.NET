# Gym.NET
A complete port of [openai/gym](https://github.com/openai/gym) to C#.<br>
** WORK IN PROGRESS ** 

##### openai/gym
OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. This is the gym open-source library, which gives you access to a standardized set of environments.


## TODO
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
   - [ ] Convert Gym.Environments to a net-standard project.
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
