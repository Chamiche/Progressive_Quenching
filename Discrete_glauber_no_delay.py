#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:46:39 2022

@author: CMoslonka

Discrete-Time Glauber dynamics, with and without delay (separated programs)
"""

# Importing modules

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


#%% Base functions

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
    

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    j : TYPE
        DESCRIPTION.

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

def Flip_probability_Discrete_Glauber(config,index_k,j,epsilon):
    sk=config[index_k]
    M_prime=np.sum(config)-sk #The sum of all the other spins
    p=1 - sk*np.tanh((j)*M_prime)
    return(p/(2*epsilon))

def Discrete_Glauber_no_Delay(start_config,PQ_index,j,epsilon,nsteps):
    """
    

    Parameters
    ----------
    start_config : list
        DESCRIPTION.
    j : TYPE
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.

    Returns the thermalized configuration
    -------

    """
    s=start_config
    for i in range(nsteps):
        for k in range(PQ_index,len(s)):
            if random.random()<Flip_probability_Discrete_Glauber(s, k, j, epsilon) :
                s[k]*=-1
    return(s)

def PQ_on_Glauber(n,j,epsilon,nsteps):
    
    spin_list=Generate_canoncial_config(n, j)
    # spin_list=Generate_random_config(n)  # For testing 
    for PQ_index in range(n):
        spin_list=Discrete_Glauber_no_Delay(spin_list, PQ_index, j, epsilon, nsteps)
        
    return(spin_list)

def Polarised_canonical_distribution(n,j,T,M):
    list_mu=np.arange(-(n-T),n-T+1,2)
    Pth=np.zeros(n-T+1)
    for i in range(n+1-T):
        mu=int(list_mu[i])
        prout=math.comb(n-T,(mu+n-T)//2)
        logP=((j/2)*(mu+M)**2) + math.log(prout)
        Pth[i]=np.exp(logP)
    Pth/=np.sum(Pth)
    
    return(list_mu,Pth)

def Generate_polarised_canonical_distribution(n,j,T,M):
    nplus=(M+T)//2
    quenched=np.ones(T)
    for i in range(nplus,T):
        quenched[i]=-1
    random.shuffle(quenched)
    r=random.random()
    N1=n-T
    list_mu,Can=Polarised_canonical_distribution(n, j, T, M)
    Cumul_canonical_probability=np.zeros(N1+2) #n+1 values and a 0 (important)
    for i in range(1,N1+2):
        Cumul_canonical_probability[i]=Can[i-1]+Cumul_canonical_probability[i-1]
    for k in range(N1+1):
        if (r>Cumul_canonical_probability[k] and r<Cumul_canonical_probability[k+1]):
            mu=list_mu[k]
    n_plus=(mu+N1)//2        
    spin_list=np.ones(N1)
    for i in range(n_plus,N1):
        spin_list[i]=-1
    random.shuffle(spin_list) #Shake that bottle baby
    return(np.concatenate((quenched,spin_list)))

def Check_meq_values(n,j,T,M,epsilon,nsteps,simulsize):
    increment=0
    for k in tqdm(range(simulsize)):
        spins=Generate_polarised_canonical_distribution(n, j, T, M)
        spins=Discrete_Glauber_no_Delay(spins, T, j, epsilon, nsteps) #get Glauber'ed
        increment+=spins[T] #The mean of this value should be meq
    meq=increment/simulsize
    return('Estimated value for meq(%i,%i) (%i spins) based on %i increments, eps=%1.1f, %i Glauber steps : ' %(T,M,n,simulsize, epsilon, nsteps), meq)
    


    
#%% Actual simulation : 
n=4
j=1/4
simulsize=100000
nsteps=5
epsilon=5
    
M_count=np.zeros(n+1)    
for i in tqdm(range(simulsize)):
    M=np.sum(PQ_on_Glauber(n, j, epsilon, nsteps))
    M_count[int(M+n)//2]+=1

Canon=Canonical_Distribution(n, j)
plt.plot(Canon[0],M_count/simulsize,'.')
plt.plot(Canon[0],Canon[1],'--')
plt.show()