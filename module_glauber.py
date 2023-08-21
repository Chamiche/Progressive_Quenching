#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 19:43:55 2022

@author: CMoslonka
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def Flip_probability_Delay(config_now, delayed_config, index_k, j, epsilon) :
    sk=config_now[index_k]
    M_prime=np.sum(delayed_config)-delayed_config[index_k]
    p=1 - sk*np.tanh(j*M_prime)
    return(p/(2*epsilon))

def Discrete_Glauber_with_Delay(start_config, memory, PQ_index, j, epsilon, nsteps):
    s=start_config
    
    for i in range(nsteps):
        stored_config=np.copy(s)
        for k in range(PQ_index, len(s)):
            if random.random()<Flip_probability_Delay(s, memory[0], k, j, epsilon):
                s[k]*=-1
        memory=np.delete(memory, 0, 0) #we remove the oldest just used
        memory=np.concatenate((memory,np.array([stored_config])),axis=0)
    return(s,memory)

def PQ_on_Glauber_with_Delay(n,j,tau,epsilon,nsteps,init_memory_type='canon'):
    actual_spin_list=Generate_canoncial_config(n, j)
    memory=np.zeros((tau,n))
    if init_memory_type=='canon':
        for i in range(tau):
            memory[i]=Generate_canoncial_config(n, j)
    elif init_memory_type=='rand':
        for i in range(tau):
            memory[i]=Generate_random_config(n)
    else :
        return('WRONG INITIAL CONDITION PARAMETER. Please choose "rand" or "canon" ')
    for PQ_index in range(1,n):
        actual_spin_list, memory=Discrete_Glauber_with_Delay(actual_spin_list, memory, PQ_index, j, epsilon, nsteps)
        
    return(actual_spin_list)
        

def Check_meq_values_delayed(n,j,tau,T,M,epsilon,nsteps,simulsize,init_memory_type='canon') :
    increment=0
    for k in range(simulsize):
        actual_spin_list=Generate_polarised_canonical_distribution(n, j, T, M)
        memory=np.zeros((tau,n))
        if init_memory_type=='canon':
            for i in range(tau):
                memory[i]=Generate_polarised_canonical_distribution(n, j, T, M)
        elif init_memory_type=='rand':
            for i in range(tau):
                memory[i]=Generate_random_config(n)
        else :
            return('WRONG INITIAL CONDITION PARAMETER. Please choose "rand" or "canon" ')
        actual_spin_list=Discrete_Glauber_with_Delay(actual_spin_list, memory, T, j, epsilon, nsteps)[0]
        increment+=actual_spin_list[T] #The mean of this value should be meq
    meq=increment/simulsize
    return('Estimated value for meq(%i,%i) (%i spins with delay tau=%i) based on %i increments, eps=%1.1f, %i Glauber steps : ' %(T,M,n,tau,simulsize, epsilon, nsteps), meq)


def Flip_probability_Continuous_Delay(config_now, delayed_config, index_k, j, epsilon, dt):
    sk=config_now[index_k]
    M_prime=np.sum(delayed_config)-delayed_config[index_k]
    p=(1 - sk*np.tanh(j*M_prime))*dt
    return(p/(2*epsilon))

def Continuous_Glauber_with_Delay(start_config, memory, PQ_index, j, epsilon, duration, dt):
    '''
    PLEASE MAKE MEMORY AS A PILE

    Returns the simulated glauber distribution
    -------

    '''
    s=start_config
    n=len(s)
    for i in np.arange(0,duration,dt):
        stored_config=np.copy(s)
        for k in range(PQ_index, n):
            if random.random()<Flip_probability_Continuous_Delay(s, memory[0], k, j, epsilon, dt):
                s[k]*=-1
        memory=np.delete(memory, 0, 0) #we remove the oldest just used
        memory=np.concatenate((memory,np.array([stored_config])),axis=0)
        
    return(s,memory)
        
def PQ_on_Continuous_Glauber_with_Delay(n,j,tau,duration,dt,epsilon,init_memory_type='canon'):
    """
    

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    j : TYPE
        DESCRIPTION.
    tau : TYPE
        In T units !!.
    duration : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.
    init_memory_type : TYPE, optional
        DESCRIPTION. The default is 'canon'.

    Returns
    -------
    None.

    """
    actual_spin_list=Generate_canoncial_config(n, j)
    memsize=int(tau/dt)
    memory=np.zeros((memsize,n))
    if init_memory_type=='canon':
        for i in range(memsize):
            memory[i]=Generate_canoncial_config(n, j)
    elif init_memory_type=='rand':
        for i in range(memsize):
            memory[i]=Generate_random_config(n)
    else :
        return('WRONG INITIAL CONDITION PARAMETER. Please choose "rand" or "canon" ')
    for PQ_index in range(1,n):
        actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, PQ_index, j, epsilon, duration, dt)
        
    return(actual_spin_list)



def Generate_static_configuration(n,j,tau,duration,dt,epsilon,init_memory_type='canon'):
    memsize=int(tau/dt)
    memory=np.zeros((memsize,n))
    if init_memory_type=='canon':
        actual_spin_list=Generate_canoncial_config(n, j)
        for i in range(memsize):
            memory[i]=Generate_canoncial_config(n, j)
    elif init_memory_type=='rand':
        actual_spin_list=Generate_random_config(n)
        for i in range(memsize):
            memory[i]=Generate_random_config(n)
    else :
        return('WRONG INITIAL CONDITION PARAMETER. Please choose "rand" or "canon" ')
    actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, 0 , j, epsilon, duration, dt)
    return(np.sum(actual_spin_list))


def Generate_static_distribution(n,j,tau,duration,simulsize,dt,epsilon,init_memory_type='canon') :
    M_count=np.zeros(n+1)    
    for i in tqdm(range(simulsize)):
        M=Generate_static_configuration(n, j, tau, duration, dt, epsilon)
        M_count[int(M+n)//2]+=1

    Canon=Canonical_Distribution(n, j)
    plt.plot(Canon[0],M_count/simulsize,'.')
    plt.plot(Canon[0],Canon[1],'--')
    plt.show()
    asym=0
    for i in range(n+1):
        asym+=np.abs(M_count[i]-M_count[-i-1])
    return(M_count/simulsize,'asymetry score : %1.6f' %(asym/(simulsize*2*(n+1))))

def Generate_static_distribution_parallel(simulsize) :
    n, j, tau, duration, dt, epsilon = (16, 1.25/16, 0.8, 10, 0.02, 1.5)
    M_count=np.zeros(n+1)    
    for i in range(simulsize):
        M=Generate_static_configuration(n, j, tau, duration, dt, epsilon)
        M_count[int(M+n)//2]+=1
        
    return(M_count)

