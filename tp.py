import base
import numpy as np

env = base.Maze()

def executer_politique(env, pi):
    #Boucle d'actions-affichage
    state = env.reset()

    #0 W, 1 S, 2 E, 3 N

    #env.step(1)

    fini = False
    #state prochain etat, r recompense, fini est-ce un état fini ou pas
    while not fini:
        env.render()
        #action = int(input("Action :"))
        action = np.random.choice(4, p=pi[state])
        state, r, fini, _ = env.step(action)
        print("Etat ", state, "- Action", action, "- récompense", r)
    env.render()

    #env.p()


#Fontions valeurs
ns = env.get_nb_states()
na = env.get_nb_actions()

v = np.zeros((ns)) #fonction valeur à chaque état: associe une valeur à chaque état
q = np.zeros((ns, na)) #fonction valeur à chaque couple etat-action: associe une valeur pour chaque couple état action

pi = np.ones((ns, na))/na #politique uniforme: chaque action a la même probalité

executer_politique(env,pi)

va = np.random.rand(ns)

#print(va)

print(env.render_values_img(va))

#l'amélioration de politique est plus facile sur la fonction q que sur v


#Calcul de q en fonction de v

#3 dimensions: 1 etat courant, 2 action effectué, 3 état ou on se trouve après avoir exécuté l'action: représente une transition
#r et p ont les 3 dimensions

#p c'est probabilité qu'une transition se réalise, r c'est la récompose obtenue après avoir fait une transition

#p matrice de probalité, r matrice de récompense


gamma = 0.5
def calculer_q(env, v, gamma):
    return np.sum(env.p()*(env.r() + gamma*v[np.newaxis, np.newaxis, :]), axis=2) #mettre v sur 3 dimensions car, p et r le sont

def ipe(env, pi, gamma, epsilon): #calcule l'estimation à long terme des gains dans chaqu'état
    ns = env.get_nb_states()
    
    v = np.zeros((ns))
    q = calculer_q(env, v, gamma)
    nv = np.sum(pi * q, axis = 1)

    delta = np.sum(np.abs(nv - v))

    while delta > epsilon:
        v = nv
        q = calculer_q(env, v, gamma)
        nv = np.sum(pi * q, axis = 1)
        delta = np.sum(np.abs(nv - v))
    return nv

#v = ipe(env, pi, 0.5, 0.01)
#print(v)

def greedy(v,env,gamma):
    ns = env.get_nb_states()
    na = env.get_nb_actions()
    pi = np.zeros((ns))
    q = calculer_q(env, v, gamma, 0.01)
    #parcourir tous les (états actions) et celui qui correspond à argmax(q) on lui donne la probabilité

    for s, qs in enumerate(q):
        best_action = np.argmax(qs)
        pi[best_action] = 1.

#greedy prend la politique qui mene au meilleur état: extraire une politique à partir d'une fonction v
def policy_iteration(env, gamma, epsilon): #faire évoluer les politiques et non les valeurs
    ns = env.get_nb_states()
    na = env.get_nb_actions()

    pi = np.ones((ns, na))/na
    v = ipe(env, pi, gamma, epsilon)
    npi = greedy(v, env, gamma)
    delta = np.sum(np.abs(npi - pi))

    while delta > epsilon:
        pi = npi
        v = ipe(env, pi, gamma, epsilon)
        npi = greedy(v, env, gamma)
        delta = np.sum(np.abs(npi - pi))
    return npi

#il y a pas de politique intermediare, calcule que les valeurs max
def value_iteration(env, gamma, epsilon):
    ns = env.get_nb_states()

    v = np.zeros((ns))

    q = calculer_q(env, v, gamma)
    nv = np.max(q, axis=1)
    delta = np.sum(np.abs(nv - v))

    while delta > epsilon:
        v = nv
        q = calculer_q(env, v, gamma)
        nv = np.max(q, axis=1)
        delta = np.sum(np.abs(nv - v))

    return greedy(nv, env, gamma)

#MDP{
    #-probalité de transition
    #-récompense
# }


#APPRENTISSAGE PAR RENFORCEMENT

"""

"""