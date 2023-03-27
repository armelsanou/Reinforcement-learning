import base
import numpy as np

#epsilon permet d'explorer d'autres chemins
#pi: politique associe une probabilité au choix d'action dans un état
def egreedy(q):
    p = np.zeros(q.shape)
    for s in range(p.shape[0]):
        p[s, np.argmax(q[s])] = 1 #dans la politique, l'action qui correspond est celle liée au max des états
    return p

def egreedy(q):
    ns = q.shape[0]
    na = q.shape[0]
    p = np.zeros(q.shape) + epsilon / na
    p = np.zeros(q.shape)
    for s in range(ns):
        p[s, np.argmax(q[s])] += 1 - epsilon #dans la politique, l'action qui correspond est celle liée au max des états
    return p

def monte_car_control(env, gamma, nb_ep_limit):
    na = env.get_nb_actions()
    q = np.zeros((env.get_nb_states, env.get_nb_actions))
    n = np.zeros((env.get_nb_states, env.get_nb_actions))
    epsilon = 1
    pi = egreedy(q, epsilon)
    for k in range(1, nb_ep_limit+1):
        #Faire un épisode
        s = env.reset()
        epsilon = []
        while not env.is_final(s):
            a = np.random.choice(na, p=pi[s])

        #Mettre à jour Q


        #Mettre à jour epsilon et pi
        epsilon = 1/k
        pi = egreedy(q, epsilon)
    return pi


env = base.Maze()

pi = monte_carlo_control(env, gamma, epsilon)
pi = sarsa(env, gamma, alpha, epsilon)
pi = qlearning(env, gamma, alpha, epsilon)

env.observe_epsilon(pi, 10) #100 limite lorsqu'on observe un épisode dans le cas d'une boucle infinie