# Gym.NET
A complete [openai/gym](https://github.com/openai/gym) port to C#.<br>
** WORK IN PROGRESS ** 

## TODO
- Implement [Spaces](https://github.com/openai/gym/tree/master/gym/spaces)
  - [X] `Space` (base class)
  - [X] `Box`
  - [X] `Discrete`
  - [ ] `multi.*.py`

- Implement [Env](https://github.com/openai/gym/blob/master/gym/core.py) base classes
  - [X] Env(object)
  - [ ] GoalEnv(Env)

 - Implement environments
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
