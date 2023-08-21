#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:32:47 2022

@author: CMoslonka
"""

import numpy as np
import random
import math
import concurrent
import time
import ast

import matplotlib.pyplot as plt

from tqdm import tqdm
from os import cpu_count

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
    p=(1 - (sk * np.tanh(j * M_prime))) * dt
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
        for k in range(PQ_index, n): #We act of those spins
            if random.random()<Flip_probability_Continuous_Delay(s, memory[0], k, j, epsilon, dt):
                s[k]*=-1
        # k = random.randint(PQ_index, n-1)
        # if random.random()<Flip_probability_Continuous_Delay(s, memory[0], k, j, epsilon, dt):
        #     s[k]*=-1
        memory = np.delete(memory, 0, 0) #we remove the oldest just used
        memory = np.concatenate((memory,np.array([stored_config])),axis=0)
        
    return(s, memory)



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


def Static_config_16():
    """
    

    Returns a list of spins drawn from the static3 distribution
    -------
    None.

    """
    n = 16

    static = np.array([0.03751333, 0.07288667, 0.08647333, 0.07759333, 0.06618,
                       0.05444, 0.04554667, 0.04090667, 0.03840667, 0.04094667,
                       0.04478667, 0.05531333, 0.06650667, 0.07772, 0.08466,
                       0.07268667, 0.03743333])

    r = random.random()
    Cumul_probability = np.zeros(n+2)  # n+1 values and a 0 (important)
    list_M = Canonical_Distribution(16, 1.25/16)[0]
    for i in range(1, n+1):
        Cumul_probability[i] = static[i-1]+Cumul_probability[i-1]
    Cumul_probability[-1] = 1
    for k in range(n+1):
        if (r > Cumul_probability[k] and r < Cumul_probability[k+1]):
            M = list_M[k]
    n_plus = (M+n)//2
    spin_list = np.ones(n)
    for i in range(n_plus, n):
        spin_list[i] = -1
    random.shuffle(spin_list)  # Shake that bottle baby
    return(spin_list)

def Generate_static_configuration_v2(n,j,tau,duration,dt,epsilon):
    memsize=int(tau/dt)
    memory=np.zeros((memsize,n))
    
    actual_spin_list=Static_config_16()
    for i in range(memsize):
        memory[i]=Static_config_16()
    
    actual_spin_list, memory = Continuous_Glauber_with_Delay(actual_spin_list, 
        memory, 0 , j, epsilon, duration, dt)
    
    return(np.sum(actual_spin_list))

def Generate_static_configuration_v3(n,j,tau,duration,dt,epsilon):
    memsize=int(tau/dt)
    memory=np.zeros((memsize,n))
    conf = Static_config_16()
    actual_spin_list = conf
    for i in range(memsize):
        memory[i] = conf
    
    actual_spin_list, memory = Continuous_Glauber_with_Delay(actual_spin_list, 
        memory, 0 , j, epsilon, duration, dt)
    
    return(np.sum(actual_spin_list))

#%%


def Generate_static_configuration_v4(n,j,tau,duration,dt,epsilon):
    
    r = random.randint(0,l-1)
    
    actual_spin_list = possible_history[r][0]
    memory = possible_history[r][1]
    
    actual_spin_list, memory = Continuous_Glauber_with_Delay(actual_spin_list, 
        memory, 0 , j, epsilon, duration, dt)
    
    return(np.sum(actual_spin_list))

# s = np.zeros(n+1)
# for i in tqdm(range(Size)):
#     M = Generate_static_configuration_v4(
#         n, j, tau, duration, dt, epsilon)
#     s[int(M+n)//2] += 1
# mag, can = Canonical_Distribution(16, 1.25/16)
 
# plt.plot(mag, s/np.sum(s), 'b.', label = 'Statio %i' %(Size))
# plt.plot(mag, static, 'r*', label = 'Starting distrubution')
 
# plt.plot(mag, can, 'r-', label = 'Canonical')
# plt.title(f"Stationary distribution with $T = 500 + {T} $ and $dt = {dtt} $")
# plt.legend()
# plt.savefig(f'Dist16spinsFromRealHistory_T=500+{T}_dt={dtt}_Size={Size}.pdf')
# plt.show()


#%%
def Write_Static_Configuration(n, j, tau, duration, dt, epsilon):
    memsize = int(tau/dt)
    memory = np.zeros((memsize, n))
    config = Static_config_16()
    actual_spin_list = config
    for i in range(memsize):
        memory[i] = config

    actual_spin_list, memory = Continuous_Glauber_with_Delay(actual_spin_list,
                                                             memory, 0, j, epsilon, duration, dt)

    return(actual_spin_list, memory)


def Generate_static_distributionv3(n, j, tau, duration, simulsize, dt, epsilon):
    M_count = np.zeros(n+1)
    for i in tqdm(range(simulsize)):
        M = Generate_static_configuration_v3(
            n, j, tau, duration, dt, epsilon)
        M_count[int(M+n)//2] += 1

    mag, can = Canonical_Distribution(n, j)
    plt.plot(mag, M_count/simulsize, '.')
    plt.plot(mag, can, '--')
    plt.show()
    asym = 0
    for i in range(n+1):
        asym += np.abs(M_count[i]-M_count[-i-1])
    proba = M_count/np.sum(M_count)
    
    file = open(f'Stationnary_distribution_T={duration}size={simulsize}.txt', 'w')
    file.write(str(proba))
    file.close()
    return(M_count/simulsize,
           'asymetry score : %1.6f' % (asym/(simulsize*2*(n+1))))

def Generate_static_distribution_parallel(simulsize) :
    n, j, tau, duration, dt, epsilon = (16, 1.25/16, 0.8, T, dtt, 1.5) #duration has to be quite large also.
    M_count=np.zeros(n+1)
    stt = time.perf_counter()    
    for i in range(simulsize):
        M=Generate_static_configuration_v3(n, j, tau, duration, dt, epsilon)
        M_count[int(M+n)//2]+=1
        if i == 9 :
            print('Estimated remaining time : ', (time.perf_counter() - stt)*simulsize/10, 'seconds')
        
        
    return(M_count)



#%%

static = np.array([0.03751333, 0.07288667, 0.08647333, 0.07759333, 0.06618   ,
        0.05444   , 0.04554667, 0.04090667, 0.03840667, 0.04094667,
        0.04478667, 0.05531333, 0.06650667, 0.07772   , 0.08466   ,
        0.07268667, 0.03743333])


cpu = cpu_count() - 3
# T = 1
# dtt = 0.05
# Size = 40000
# s = np.zeros(17)
# if __name__ == '__main__' :
#     __spec__ = None
#     with concurrent.futures.ProcessPoolExecutor() as executor :
#         results = [executor.submit(Generate_static_distribution_parallel, Size) for _ in range(cpu)]
    
#         for f in concurrent.futures.as_completed(results) : 
#             s += f.result()
#         print(s)
            
#     print(time.perf_counter() - stt)

#     mag, can = Canonical_Distribution(16, 1.25/16)
    
#     plt.plot(mag, s/np.sum(s), 'b.', label = 'Statio %i' %(Size*6))
#     plt.plot(mag, static, 'r*', label = 'Starting distrubution')
#     plt.plot(mag, can, 'r-', label = 'Canonical')
#     plt.title(f"Stationary distribution with $T = 500 + {T} $ and $dt = {dtt} $")
#     plt.legend()
#     plt.savefig(f'Dist16spinsFromHistory_T=500+{T}_dt={dtt}_Size={Size * cpu}.pdf')
#     plt.show()

#%% Writting module
ind = 0
T = 50
dtt = 0.1
Size = 200000
n, j, tau, duration, dt, epsilon = (16, 1.25/16, 0.8, T, dtt, 1.5)

f = open(f'List_of_static_history_dt={dtt}_size={Size}__num{ind}.txt', 'w')
f.write('[')
for i in tqdm(range(Size)) : 
    config = Write_Static_Configuration(n, j, tau, duration, dt, epsilon)
    f.write('[') 
    f.write(str(config[0]))
    f.write(',[')
    for k in range(int(tau/dtt) - 1) : 
        f.write(str(config[1][k]))
        f.write(',')
    f.write(str(config[1][-1]))
    f.write(']],')
config = Write_Static_Configuration(n, j, tau, duration, dt, epsilon)
f.write('[') 
f.write(str(config[0]))
f.write(',[')
for k in range(int(tau/dtt) - 1) : 
    f.write(str(config[1][k]))
    f.write(',')
f.write(str(config[1][-1]))
f.write(']]]')
f.close()

ind +=0

#%% Module pour la lecture

# f = open('lebonfichier.txt', 'r')
# raw = f.read()
# raw = raw.replace('. ',',')

# possible_history = ast.literal_eval(raw)

# l = len(possible_history)

# del(raw)
# f.close()
#%% PQ module with a good memory state
def PQ_on_Continuous_Glauber_with_Delay_from_Static() :
    memsize = int(tau/dt)
    memory = np.zeros((memsize, n))
    config = Static_config_16()
    actual_spin_list = config
    for i in range(memsize):
        memory[i] = config

    for PQ_index in range(1,16):
        actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, PQ_index, 1.25/16, 1.5, T, dtt)
        
    return(actual_spin_list)


def PQ_on_Continuous_Glauber_with_Delay_from_History() :
    r = random.randint(0,l-1)
    
    actual_spin_list = possible_history[r][0]
    memory = possible_history[r][1]

    for PQ_index in range(1,16):
        actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, PQ_index, 1.25/16, 1.5, T, dtt)
        
    return(actual_spin_list)

def Dist_PQ_from_static(size) :
    
    M_count=np.zeros(17)
    stt = time.perf_counter()    
    for i in range(size):
        M=np.sum(PQ_on_Continuous_Glauber_with_Delay_from_Static())
        M_count[int(M+16)//2]+=1
        if i == 19 :
            print('Estimated remaining time : ', (time.perf_counter() - stt)*size/20, 'seconds')
        
    return(M_count)

def Dist_PQ_from_history(size) :
    
    M_count=np.zeros(17)    
    for i in range(size):
        M=np.sum(PQ_on_Continuous_Glauber_with_Delay_from_History())
        M_count[int(M+16)//2]+=1
        
    return(M_count)
# T = 1

# size = 20000
# s = Dist_PQ_from_static(size)
# s = s/np.sum(s)

# mag, can  = Canonical_Distribution(16,1.25/16)

# plt.plot(mag, can, label = 'Canonical Distribution')
# plt.plot(mag, static,'r*', label = 'Stationnary Distribution ')
# plt.plot(mag, s/np.sum(s), 'b.', label = f' PQ $T = {T} $')

# plt.legend()
# plt.title(f'PQ with dt = {dtt} and T = {T} over {size} samples')
# plt.savefig(f'PQfrom_history_T={T}_dt={dtt}_{size}.pdf')
# plt.show()


def Partial_PQ_dist_from_static(size):
    return None 

#%%

T = 100
dtt = 0.2
Size = 8000
n, j, tau, duration, dt, epsilon = (16, 1.25/16, 0.8, T, dtt, 1.5)


s = np.zeros(17)
if __name__ == '__main__' :
    __spec__ = None
    with concurrent.futures.ProcessPoolExecutor() as executor :
        results = [executor.submit(Dist_PQ_from_static, Size) for _ in range(cpu)]
    
        for f in concurrent.futures.as_completed(results) : 
            s += f.result()
        print(s)
    prout = s/np.sum(s)
    file = open(f'PQ_from_fake_history_T={T}_size={Size*cpu}_dt={dtt}.txt', 'w')
    file.write(str(prout))
    file.close() 
