import gym
import numpy as np
from ppo_torch import Agent
#from newmaze import Maze
from function import Funcao
from utils import plot_learning_curve
from matplotlib import pyplot as plt
"""
Nessa versão da function, irei discretizar o ambiente, da seguinte forma:
a cada 0.2 na função real, corresponde 1 no ambiente externo.
Além disso, irei utilizar um array, que irá considerar o ambiente:
self.px + (Tamanho da matrix+1) * self.py


"""
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
    #env = gym.make('CartPole-v0')
    #env = gym.make('FrozenLake-v0', is_slippery=False )
    #env = Maze()
    env = Funcao()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0000003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=(1,)) # frozen lake
    #agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape) # cartpole
    #agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=(2,))

    n_games = 2000

    figure_file = 'plots/runningavg.png'

    #best_score = env.reward_range[0]
    best_score = np.float32('-inf')
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation_ = env.reset()
        observation = pos_to_num(observation_[0], observation_[1])
        
        done = False
        score = 0
        max_steps = 0
        
        while not done and max_steps < 100:
            
            action, prob, val = agent.choose_action(observation)
            
            #took_action(action)
            
            observation_, reward, done, info = env.step(action)
            
            observation = pos_to_num(observation_[0], observation_[1])
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            #print("n_steps "+str(n_steps))
            if n_steps % N == 0: 
                agent.learn()
                learn_iters += 1
            observation = observation = pos_to_num(observation_[0], observation_[1])
            max_steps += 1
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])


        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    figure_scores, scores_plt = plt.subplots()
    scores_plt.plot(x, score_history)

    scores_plt.set(xlabel='Number of epochs', ylabel='Scores', title='Scores per epoch')
    figure_scores.savefig("scores_history.png")


