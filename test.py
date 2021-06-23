import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
from ppo_torch import ActorNetwork
from function import Funcao
import matplotlib.pyplot as plt


def pos_to_num(x, y):
    """
    Essa função mapeia uma tupla (x,y) dentro do domínio para um valor z
    Como os passos são de 0.2, entre -2 e 2, então irei mapear os 2 primeiros
    digitos para o valor em y e os 2 maiores digitos para o valor de x
    ou seja [-1.8, -1.8]
    se transformara em
    [1,1]
    os pontos -2.2, 2.2 estarão fora da matriz
    porém devem ser levados em conta, pois o agente pode chegar nesse valor
    Assim, dado a matriz NxN
    os pontos:
    [a,b] sendo:
    0<a<N
    N*N<a<N*N+N

    E

    B = N+1

    N*X+1

    """
    new_y = (y+2)
    new_y = new_y/0.2
    new_y = round(new_y)+1
    
    new_x = (x+2)
    new_x = new_x/0.2
    new_x = round(new_x)+1
    return new_x + 22*new_y # Os valores de x vão de 0 a 21. Assim o valor de y deve ser multiplicado por 22, para não haver conflito
#def choose_action(actions, cur_state):

if __name__ == '__main__':
    #env = gym.make('FrozenLake-v0', is_slippery=False )
    #env = Maze()
    env = Funcao()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=(1,))
    eixo_x = []
    eixo_y = []

    acao0 = 0
    acao1 = 0
    acao2 = 0
    acao3 = 0

    agent.load_models()

    
    done = False
    obs = []
    for i in range (100):
        observation_ = env.reset()
        observation = pos_to_num(observation_[0],observation_[1])

        num_steps = 0
        
        while not done and num_steps < 100:
            eixo_x.append(env.px)
            eixo_y.append(env.py)
            action, prob, val = agent.choose_action(observation)
            if action == 0:
                acao0+=1
            elif action == 1:
                acao1+=1
            elif action == 2:
                acao2+=1
            elif action == 3:
                acao3+=1

            observation_, reward, done, info = env.step(action)
            #import pdb;breakpoint()
            #observation = pos_to_num(observation_[0],observation_[1])
            num_steps += 1  
        done = False

print('Acao0: '+str(acao0))
print('Acao1: '+str(acao1))
print('Acao2: '+str(acao2))
print('Acao3: '+str(acao3))

plt.hist2d(eixo_x,eixo_y, bins=[np.arange(-2,2,0.2),np.arange(-2,2,0.2)])

plt.show()
"""
def completness(rotaX, rotaY, ox, oy, value=0.5):
    for rx, ry in zip(rotaX, rotaY):
        for obsx, obsy in zip (ox, oy):
            if dist_euclidiana(rx, ry, obsx, obsy) <= value:
                return True

    return False
"""