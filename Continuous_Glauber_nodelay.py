#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:15:59 2023

@author: CMoslonka
Just produce a graph to proove that the steady state doesn't depend on epsilon

No pogressive quenching here
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Parameters
N_spins = 16
j = 1/N_spins

list_epsilon = [1.2,1.5,2,3,5]

dt = 0.1 
N_config = 100000

#%%

def Generate_random_config(n):
    l=[]
    for i in range(n):
        l.append(random.choice([1,-1]))
    return(l)

def Canonical_Distribution(n,j) :
    """
    Returns the magnetisation values and the probability distribution 
    
    n = system size
    
    """
    list_M=np.arange(-n,n+1,2)
    Pth=np.zeros(n+1)
    for i in range(n+1):
        Mag=int(list_M[i])
        prout=math.comb(n,(Mag+n)//2)
        logP=((j/2)*((Mag**2))) + math.log(prout)
        Pth[i]=np.exp(logP)
    Pth/=np.sum(Pth)
    
    return(list_M,Pth)

def Generate_canoncial_config(n,j):
    """


    Returns a list of spins generated with a canonical probability
    -------
    None.

    """
    r=random.random()
    Cumul_canonical_probability=np.zeros(n+2) #n+1 values and a 0 (important)
    list_M,Can=Canonical_Distribution(n, j)
    for i in range(1,n+2):
        Cumul_canonical_probability[i]=Can[i-1]+Cumul_canonical_probability[i-1]
    for k in range(n+1):
        if (r>Cumul_canonical_probability[k] and r<Cumul_canonical_probability[k+1]):
            M=list_M[k]
    n_plus=(M+n)//2        
    spin_list=np.ones(n)
    for i in range(n_plus,n):
        spin_list[i]=-1
    random.shuffle(spin_list) #Shake that bottle baby
    return(spin_list)

#%% Glauber algorithm 
"""
We just plot the steady distributions of Glauber algorithms
to show that the are independent from the values of epsilon
even if they vary in time (deterministicly or randomly)
"""

def Flip_probability(config, index_k, j, epsilon, dt) :
    '''
    Parameters
    ----------
    config : list
        list of values of the Ising spins .
    index_k : int
        index of the spin on which the test is made.
    j : float
        interaction constant value.
    epsilon : float
        value of the kinetic parameter.

    Returns the probability of having the spin flipped
    -------

    '''
    sk=config[index_k]
    M_prime=np.sum(config)-sk
    p=(1 - sk*np.tanh(j*M_prime))*dt
    return(p/(2*epsilon))

def Discrete_Glauber(config, j, epsilon, duration, dt):
    """
    Returns one distribution, after a whole glauber process
    of set duration, in units of dt.
    """
    
    for _ in np.arange(0,duration,dt):
        for k in range(len(config)):
            if random.random()<Flip_probability(config, k, j, epsilon, dt):
                config[k]*=-1
    return(config)

def Static_distribution_Glauber(N_spins, j, epsilon, duration, dt, N_config):
    
    M_count = np.zeros(N_spins+1) 
    for _ in tqdm(range(N_config)):
        config = Generate_random_config(N_spins) # First config is random
        # Glauber algorithm on the random configuration, reach the steady state
        config = Discrete_Glauber(config, j, epsilon, duration, dt)
        #Compute the magnetisation and count the values : 
        M = np.sum(config)
        M_count[int(M+N_spins)//2]+=1
    return(M_count/N_config)

def Plot_of_steady_state(N_spins, j, list_epsilon, dt, N_config) :
    mag, can = Canonical_Distribution(N_spins, j)
    for epsilon in list_epsilon :
        duration = 10 * epsilon
        dist = Static_distribution_Glauber(N_spins, j, epsilon, duration, dt, N_config)
        f = open(f'Steady_state_from_random_config_with_epsilon_{epsilon}.txt','w')
        f.write(f'Parameters are : \n N_spins = {N_spins} \n j = {j} \
                 \n dt = {dt} \n N_config = {N_config}')
        f.write('\n ----------- \n')
        f.write(str(dist))
        f.close()
        plt.plot(mag, dist,'-.', label = f'$\\epsilon = $ {epsilon}')
    plt.plot(mag, can, 'b--', label = 'Canonincal Distribution')
    plt.xlabel('Magnetisation Values')
    plt.ylabel('Steady state probability distribution')
    plt.legend()
    plt.grid()
    plt.title('Plot of the steady state distributions of a Glauber algorithm \n from random configurations over $ 10 \\epsilon $')
    plt.show()

def Steady_State_variating_epsilon(N_spins, j, dt, N_config) :
    mag, can = Canonical_Distribution(N_spins, j)
    list_t = np.arange(0, 10*3, dt)
    M_count = np.zeros(N_spins+1) 
    for _ in tqdm(range(N_config)):
        config = Generate_random_config(N_spins)
        for t in list_t :
            for k in range(len(config)):
                if random.random()<Flip_probability(config, k, j, 3 + np.cos(t) , dt):
                    config[k]*=-1
        M = np.sum(config)
        M_count[int(M+N_spins)//2]+=1
    dist = M_count/N_config
    f = open ('Steady_state_random_start_variating_epsilon.txt', 'w')
    f.write(str(dist))
    f.close()
    plt.plot(mag,can,label='canon')
    plt.plot(mag, dist, label ='$ \\epsilon = 3+ \\cos (t) $')
    plt.xlabel('Magnetisation Values')
    plt.ylabel('Steady state probability distribution')
    plt.legend()
    #plt.grid()
    plt.title('Plot of the steady state distributions of a Glauber algorithm \n from random configurations over $ 10 < \\epsilon > $')
    plt.show()
    
def Steady_State_random_epsilon(N_spins, j, dt, N_config) :
    mag, can = Canonical_Distribution(N_spins, j)
    list_t = np.arange(0, 10*2, dt)
    M_count = np.zeros(N_spins+1) 
    for _ in tqdm(range(N_config)):
        config = Generate_random_config(N_spins)
        for t in list_t :
            for k in range(len(config)):
                if random.random()<Flip_probability(config, k, j, 1+2*random.random() , dt):
                    config[k]*=-1
        M = np.sum(config)
        M_count[int(M+N_spins)//2]+=1
    dist = M_count/N_config
    f = open (f'Steady_state_random_start_random_epsilon.txt_{N_config}', 'w')
    f.write(str(dist))
    f.close()
    plt.plot(mag,can,label='canon')
    plt.plot(mag, dist, label ='$ \\epsilon = 1 + 2 \\mathcal{U} [0,1] $')
    plt.xlabel('Magnetisation Values')
    plt.ylabel('Steady state probability distribution')
    plt.legend()
    #plt.grid()
    plt.title('Steady state distribution of a Glauber algorithm \n from random configurations over $ 10 < \\epsilon > $')
    plt.show()
    
#%% plotting the data 
list_epsilon = [1.2,1.5,2,3,5]
mag, can = Canonical_Distribution(N_spins, j)

esp1_2 = np.array([0.01547, 0.03947, 0.0577,  0.06872, 0.07287, 0.07129, 0.07076, 0.07047, 0.06883,
 0.06888, 0.07044, 0.07223, 0.0709 , 0.06839 ,0.05771 ,0.0396 ,
  0.01627])
eps1_5 = np.array([0.01616 ,0.04    ,0.05808 ,0.06753 ,0.07184 ,0.07251 ,0.07029 ,0.07055 ,0.06824
 ,0.06797 ,0.07128 ,0.07153 ,0.07349 ,0.06973 ,0.05698 ,0.03821 ,0.01561])
eps1_2 = esp1_2
eps2 = np.array([0.01532 ,0.03963 ,0.05709 ,0.06768 ,0.07184 ,0.07213 ,0.07228 ,0.07038 ,0.06896
 ,0.07074 ,0.07043 ,0.07118 ,0.07159 ,0.06821 ,0.05768 ,0.03919 ,0.01567])
eps3 = np.array([0.01564 ,0.03887 ,0.05716 ,0.06853 ,0.07255 ,0.07264 ,0.07031 ,0.06879 ,0.06778
 ,0.06977 ,0.07108 ,0.07261 ,0.07308 ,0.06864 ,0.05895 ,0.03884 ,0.01476])
eps5 = np.array([0.01551 ,0.03954 ,0.05704 ,0.06694 ,0.07251 ,0.07309 ,0.07131 ,0.07004 ,0.0684
 ,0.06912 ,0.07262 ,0.07206 ,0.07192 ,0.06871 ,0.05727 ,0.03832 ,0.0156 ])

eps_var = np.array([0.01582 ,0.03844 ,0.05724 ,0.06918 ,0.07146 ,0.07418 ,0.07158 ,0.06952 ,0.06917
 ,0.06965 ,0.0722  ,0.07063 ,0.07273 ,0.0667  ,0.05687 ,0.03895 ,0.01568])

eps_rand = np.array([0.01487 ,0.04009 ,0.05702 ,0.06771 ,0.07238 ,0.07124 ,0.07195 ,0.07029 ,0.06911
 ,0.06915 ,0.07144 ,0.07218 ,0.07166 ,0.06895 ,0.05597 ,0.03995 ,0.01604])



#plt.plot(mag, eps1_2 , ls = '-', marker = '1', label = '$\\varepsilon = 1.2 $')
plt.plot(mag, eps1_5 , ls = '-', marker = '1', label = '$\\varepsilon = 1.5 $')
plt.plot(mag, eps2 , ls = '-', marker = '2', label = '$\\varepsilon = 2 $')
plt.plot(mag, eps3 , ls = '-', marker = '3', label = '$\\varepsilon = 3 $')
plt.plot(mag, eps5 , ls = '-', marker = '4', label = '$\\varepsilon = 5 $')

plt.plot(mag, eps_var , ls = '-', marker = '+', label = 'Variating case:\n $\\varepsilon(t) = 2 + \\cos(t) $')
plt.plot(mag, eps_rand , ls = '-', marker = 'x', label = 'Random case:\n $\\varepsilon = 1 + 2\, \\mathcal{U} \, [0,1] $')

plt.plot(mag,can , ls='--', color = 'k' , label = 'Canonical\nDistribution')

plt.xlabel('Magnetisation Values')
plt.ylabel('Steady state probability distribution')
plt.legend(fontsize = 11)
#plt.grid()
#plt.title('Steady state distribution of a Glauber algorithm \n from random configurations over $ 10 < \\epsilon > $')
#plt.savefig('Steady_State_Glauber_multiple_epsilon2.pdf')
plt.show()