#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:01:10 2022

@author: CMoslonka
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# from threading import Thread, RLock

# nope = RLock()

#%% Generating configurations

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

#%%

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
    for k in tqdm(range(simulsize)):
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
#%% Continuous Glauber 


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
        # for k in range(PQ_index, n): #We act of those spins
        #     if random.random()<Flip_probability_Continuous_Delay(s, memory[0], k, j, epsilon, dt):
        #         s[k]*=-1
        k = random.randint(PQ_index, n-1)
        if random.random()<Flip_probability_Continuous_Delay(s, memory[0], k, j, epsilon, dt):
            s[k]*=-1
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


#%% Generation of the static distribution


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

#%% Multi-threading goes brrrrr

# class Afficheur(Thread) :
    
#     def __init__(self, n, j):
#         Thread.__init__(self)
#         self.n = n
#         self.j = j
        
#     def run(self):
#         M = 0
#         for i in range(10000):
#             M += np.sum(Generate_canoncial_config(self.n, self.j))
#         with nope : 
# #             print('M vaut', M)
            
# import multiprocessing
# import time

# def f(n):
#     M=0
#     j=n/16
#     for i in range(10000):
#         M+= np.sum(Generate_canoncial_config(n, j))
#     return(M)

# def it(param):
#     with multiprocessing.Pool() as pool :
#         pool.map(f, param)

# # if __name__ == '__main__' :
param=[16]*10
#     it(param)
#%% Another test :
# import os
   
# if __name__ == '__main__' :
#     stt=time.time()
#     pool = multiprocessing.Pool(os.cpu_count())
#     print(pool.map(f, param)) # That is the way
#     print(time.time()-stt)
    
#%%
# if __name__ == '__main__' :
#     p1 = multiprocessing.Process(target=(f), args=[16])
#     p2 = multiprocessing.Process(target=(f), args=[16])
#     p3 = multiprocessing.Process(target=(f), args=[16])
    
#     p1.start()
#     p2.start()
#     p3.start()
    
#     p1.join()
#     p2.join()
#     p3.join()
    
#%% Implementation 

# from module_glauber import Generate_static_distribution_parallel # should solve ? 

# if __name__ == '__main__' :
#     stt=time.time()
#     pool = multiprocessing.Pool(os.cpu_count()-2) # should be 8
#     print(pool.map(Generate_static_distribution_parallel, [16]*8)) # That is the way
#     print(time.time()-stt)


#%% Approx_static_distribution :
    
static=[0.03153, 0.06268, 0.07599, 0.07532, 0.06663, 0.06064, 0.05197,
        0.04948, 0.04778, 0.0496 , 0.05194, 0.05887, 0.06907, 0.07731,
        0.07668, 0.06378, 0.03073] # made with less config but smaller st

static2=[0.03081667, 0.06279   , 0.07785333, 0.07544   , 0.06780333,
        0.05966333, 0.05301   , 0.04876667, 0.04830333, 0.05008667,
        0.05275333, 0.06033667, 0.06703   , 0.07566667, 0.07585   ,
        0.06289   , 0.03094   ] # 3 times More config but larger dt 

static10 = [0.03141071, 0.0639119,  0.07648452, 0.0759619,  0.06774643, 0.05890714,
 0.05287024, 0.0490131,  0.04797738, 0.04919286, 0.05260119, 0.059725,
 0.06756905, 0.07529762, 0.0769119,  0.06298214, 0.0314369 ] #best (860000 and 0.01, T = 10)

static_T100 = np.array([1826., 3600., 4054., 3737., 3102., 2578., 2152., 1962., 1896.,
       1902., 2151., 2645., 3123., 3823., 4097., 3610., 1742.]) / 48000

static1000 = np.array([0.0395    , 0.07525   , 0.08316667, 0.07725   , 0.06733333,
       0.05591667, 0.04675   , 0.03866667, 0.04041667, 0.04166667,
       0.0455    , 0.05541667, 0.06808333, 0.0745    , 0.08608333,
       0.0715    , 0.033     ])
static1000_v2 = np.array([0.04      , 0.0725    , 0.08191667, 0.08      , 0.06458333,
       0.05633333, 0.045     , 0.04408333, 0.04241667, 0.03891667,
       0.04425   , 0.0545    , 0.06508333, 0.08133333, 0.08041667,
       0.07591667, 0.03275   ]) #12000

static500 = np.array([0.03838571, 0.0755    , 0.08577143, 0.0792    , 0.06487143,
       0.05374286, 0.04354286, 0.03968571, 0.03902857, 0.04037143,
       0.04465714, 0.05401429, 0.06518571, 0.0786    , 0.08528571,
       0.07384286, 0.03831429]) #70000

afterPQ  = [0.0553 ,    0.08617692, 0.08638077, 0.07393846, 0.0587,     0.04715769,
 0.03876538, 0.03545 ,   0.03427692, 0.03556538, 0.03973462, 0.04750769,
 0.05910769, 0.07402308, 0.08680385, 0.08581923, 0.05529231] #260000 iter, same parameters as static3 (SO T=10 !!!!)

afterPQ_stat500 = np.array([0.03838571, 0.0755    , 0.08577143, 0.0792    , 0.06487143,
       0.05374286, 0.04354286, 0.03968571, 0.03902857, 0.04037143,
       0.04465714, 0.05401429, 0.06518571, 0.0786    , 0.08528571,
       0.07384286, 0.03831429]) #?????

afterPQ_stat500_2 = np.array([0.02816429, 0.05821429, 0.07197143, 0.07252143, 0.06935714,
       0.06200714, 0.05722143, 0.05470714, 0.05372143, 0.05337857,
       0.05642857, 0.06307143, 0.06865   , 0.07302143, 0.07135714,
       0.05735714, 0.02885   ])

static100_v2 = np.array([0.04835   , 0.083     , 0.08682143, 0.07588571, 0.06152143,
       0.0504    , 0.04187857, 0.03584286, 0.03469286, 0.03705   ,
       0.04077143, 0.04961429, 0.05971429, 0.07642857, 0.08701429,
       0.08258571, 0.04842857])

static1000_v2 = np.array([0.05618571, 0.08836429, 0.08741429, 0.07265   , 0.05847143,
       0.04666429, 0.03932143, 0.03586429, 0.03432143, 0.03516429,
       0.0393    , 0.04682143, 0.05748571, 0.07241429, 0.08703571,
       0.08662857, 0.05589286])

static1000_best = np.array([0.03751333, 0.07288667, 0.08647333, 0.07759333, 0.06618   ,
       0.05444   , 0.04554667, 0.04090667, 0.03840667, 0.04094667,
       0.04478667, 0.05531333, 0.06650667, 0.07772   , 0.08466   ,
       0.07268667, 0.03743333])


# There is a problem bc we planned to do T = 1
# But I did T = 10
# The funny thing is that we recovered the canonical distribution with that
# I think it is probably because the static distribution was not quite the good one


def Static_config_16():
    """
    

    Returns a list of spins drawn from the static3 distribution
    -------
    None.

    """
    n = 16
    
    static = np.array([0.03751333, 0.07288667, 0.08647333, 0.07759333, 0.06618   ,
           0.05444   , 0.04554667, 0.04090667, 0.03840667, 0.04094667,
           0.04478667, 0.05531333, 0.06650667, 0.07772   , 0.08466   ,
           0.07268667, 0.03743333])
    
    r=random.random()
    Cumul_probability=np.zeros(n+2) #n+1 values and a 0 (important)
    list_M = Canonical_Distribution(16, 1.25/16)[0]
    for i in range(1,n+1):
        Cumul_probability[i]=static[i-1]+Cumul_probability[i-1]
    Cumul_probability[-1] = 1
    for k in range(n+1):
        if (r>Cumul_probability[k] and r<Cumul_probability[k+1]):
            M = list_M[k]
    n_plus=(M+n)//2        
    spin_list=np.ones(n)
    for i in range(n_plus,n):
        spin_list[i]=-1
    random.shuffle(spin_list) #Shake that bottle baby
    return(spin_list)

T = 15
ddt = 0.1

def PQ_on_Continuous_Glauber_with_Delay_from_static_16() :
    config = Static_config_16()
    actual_spin_list = config
    memsize=int(0.8/ddt)
    memory=np.zeros((memsize,16))
    for i in range(memsize):
        memory[i]=config

    for PQ_index in range(1,16):
        actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, PQ_index, 1.25/16, 1.5, T, ddt)
        
    return(actual_spin_list)

def Dist_PQ_from_static(size) :
    
    M_count=np.zeros(17)    
    for i in range(size):
        M=np.sum(PQ_on_Continuous_Glauber_with_Delay_from_static_16())
        M_count[int(M+16)//2]+=1
        
    return(M_count)

def Partial_PQ_dist_from_static(size):
    return None 

#%%
import concurrent
import time 
from os import cpu_count

cpu = cpu_count() - 2
size = 10000
s = np.zeros(17)
if __name__ == '__main__' :
    __spec__ = None
    stt = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor :
        results = [executor.submit(Dist_PQ_from_static, size) for _ in range(cpu)]
    
        for f in concurrent.futures.as_completed(results) : 
            s += f.result()
        print(s)
    print(time.time() - stt)
   
if __name__ == '__main__' :
    mag, can  = Canonical_Distribution(16,1.25/16)
    
    plt.plot(mag, can, label = 'Canon')
    plt.plot(mag, static1000_best,'r*', label = 'Starting distribution')
    plt.plot(mag, s/np.sum(s), 'b.', label = f'After PQ $T = {T} $')
    
    plt.legend()
    plt.title(f'PQ with dt = {ddt} and T = {T} over {size * cpu} iterations')
    plt.savefig(f'PQ_from_historyT={T}_dt={ddt}_size={size*cpu}.pdf')
    plt.show()

# #%% Test pour voir si ya un j effectif 

# def Distance_Distrib(n,j) :
#     static = static10 = [0.03141071, 0.0639119,  0.07648452, 0.0759619,  0.06774643, 0.05890714,
#        0.05287024, 0.0490131,  0.04797738, 0.04919286, 0.05260119, 0.059725,
#        0.06756905, 0.07529762, 0.0769119,  0.06298214, 0.0314369 ]
#     can = Canonical_Distribution(n, j/n)[1]
#     mini = 0
#     for i in range(3,13) :
#         mini += np.abs(static[i] - can[i])
#     return mini

# dist_list = []
# for j in np.linspace(1.18, 1.22, 100000) : 
#     dist_list.append((Distance_Distrib(16, j), j))
# print(min(dist_list), 'for static10')

        




