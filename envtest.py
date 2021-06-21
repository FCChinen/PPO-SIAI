import gym
from gym import spaces
from function import Funcao
from newmaze import Maze
import numpy as np

env = Maze(8)
a = gym.spaces.Discrete(4)
print(a.n)
env.reset() 
"""


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    env.reset()
    print(env.action_space)
    print(env.observation_space)
    obs, reward, done, _ = env.step(env.action_space.sample())
    print(str(obs))
    
"""

print(env.action_space.n)

#action = env.action_space.sample()
#print("A prox ação é: ", action)
#obs, reward, done, _ = env.step(action)
#print(str(obs))
#print(str(done))