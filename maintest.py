import gym
import numpy as np
from ppo_torch import Agent
#from newmaze import Maze
from function import Funcao
from utils import plot_learning_curve
from matplotlib import pyplot as plt

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

# Função para testar
if __name__ == "__main__":
    f = Funcao()
    ex = 0
    while ex == 0:
        obs = f.reset()
        new_pos = pos_to_num(obs[0], obs[1])
        terminate = f.terminate()
        while (terminate == False):
            action = int(input('Digite uma ação de 0 a 3'))
            obs, reward, done , _ = f.step(action)
            new_pos = pos_to_num(obs[0], obs[1])
            print("Discreto: "+str(new_pos))
            print("Continuo: ",str(obs[0]),str(obs[1]))
            print("Sua recompensa foi: "+ str(reward))
            terminate = done
            if terminate == True:
                print("Fim de jogo: recompensa final = "+str(f.sum_reward))
        ex = int(input("Digite 0 para continuar e qualquer número para sair"))