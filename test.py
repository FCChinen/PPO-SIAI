import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
from ppo_torch import ActorNetwork
from newmaze import Maze
if __name__ == '__main__':
    #env = gym.make('FrozenLake-v0', is_slippery=False )
    env = Maze()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=(1,))

    agent.load_models()

    
    done = False
    obs = []
    for i in range (100):
        observation = env.reset()
        obstacles = 0
        num_steps = 0
        tree = 0
        free_colision = []
        
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if reward == -5:
                obstacles += 1
            elif reward == -50:
                tree += 1
            num_steps += 1

        env.render()   
        done = False
        if obstacles == 0 and tree == 0:
            free_colision.append(i)
        obs.append([obstacles, tree, num_steps])
            
    print(obs)     
    #import pdb; breakpoint()
"""
def completness(rotaX, rotaY, ox, oy, value=0.5):
    for rx, ry in zip(rotaX, rotaY):
        for obsx, obsy in zip (ox, oy):
            if dist_euclidiana(rx, ry, obsx, obsy) <= value:
                return True

    return False
"""