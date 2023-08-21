#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:37:31 2023

@author: CMoslonka

Please load data file to recover variables and stuff
"""
import numpy as np
import random
import math


import matplotlib.pyplot as plt

from tqdm import tqdm

# Setting parameters

n = 8
j = 1/n
dt = 0.1
mem_length = 16
epsilon = 1.5
eps = epsilon
duration = 10 * epsilon
tau = mem_length * dt



plt.style.use('bmh')

#%% Generation of configurations

def Generate_random_config(n):
    """
    Generate a random spin configuration of size n

    Parameters
    ----------
    n : round
        
    """
    l=[]
    for i in range(n):
        l.append(random.choice([1,-1]))
    return(l)

def Canonical_Distribution(n,j) :
    """
    Returns the magnetisation values and the probability distribution 
    of a spin system at equilibrium temperature 1 and coupling constant j.
    
    n = system size
    
    """
    list_M=np.arange(-n,n+1,2)
    Pth=np.zeros(n+1)
    for i in range(n+1):
        Mag=round(list_M[i])
        prout=math.comb(n,(Mag+n)//2)
        logP=((j/2)*((Mag**2))) + math.log(prout)
        Pth[i]=np.exp(logP)
    Pth/=np.sum(Pth)
    
    return(list_M,Pth)

def Generate_canonical_config(n,j):
    """


    Returns a list of n spins generated with a canonical probability at
    temperature 1 and coupling j 
    -------

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

mag, can = Canonical_Distribution(n,j)

#%% Symetrise a curve

def Make_symetric(l):
    N = len(l)
    new_l = np.zeros(N)
    if N%2 == 0: #If there is a pair number of points
        for i in range(N//2):
            new_l[i] = (l[i] + l[-(i+1)])/2
            new_l[-i-1] = new_l[i]
        return(new_l)
    else :
        for i in range(N//2):
            new_l[i] = (l[i] + l[-(i+1)])/2
            new_l[-i-1] = new_l[i]
        new_l[N//2] = l[N//2]
        return(new_l)



#%% Generate a starting configuration and a matching history

def Generate_fixed_random_history(n,mem_length) :
    actual_config = Generate_random_config(n)
    memory  = np.array([actual_config]*mem_length)
    return(actual_config, memory)

def Generate_fixed_canon_history(n,j,mem_length) :
    actual_config = Generate_canonical_config(n,j)
    memory  = np.array([actual_config]*mem_length)
    return(actual_config, memory)

def Generate_alternating_random_history(n,mem_length) :
    actual_config = Generate_random_config(n)
    memory  = []
    for _ in range(mem_length):
        memory.append(Generate_random_config(n))
    return(actual_config, np.array(memory))

def Generate_alternating_canon_history(n,j,mem_length) :
    actual_config = Generate_canonical_config(n,j)
    memory  = []
    for _ in range(mem_length):
        memory.append(Generate_canonical_config(n,j))
    return(actual_config, np.array(memory))


#%% Choi-Hubberman algorithm : flip probability with delay

def Flip_probability_Continuous_Delay(config_now, delayed_config, index_k, j, epsilon, dt):
    """
    Choi-Hubberman model.
    Returns the flip probability of the k-th spin of the actual configuration,
    that depends on the latest configuration of the memory.
    The coupling is set to j and the kinetic constant is set to epsilon.
    """
    sk=config_now[index_k]
    M_prime=np.sum(delayed_config)-delayed_config[index_k]
    p=(1 - (sk * np.tanh(j * M_prime))) * dt
    return(p/(2*epsilon))

def Continuous_Glauber_with_Delay(start_config, memory, PQ_index, j, epsilon, duration, dt):
    '''
    Returns a modified configuration and the memory state
    according to the CH-Glauber algorithm
    applied nsteps times with parameters j and epsilon.
    The PQ index prevents the update of spins (effectively frozen) number 0 to 
    PQ_index-1.
    The memory list needs to have the oldest configuration on index 0

    '''
    s=start_config
    n=len(s)
    for _ in np.arange(0,duration,dt):
        stored_config=np.copy(s)
        for k in range(PQ_index, n): #The quenched spins are not modified
            if random.random()<Flip_probability_Continuous_Delay(s, memory[0], k, j, epsilon, dt):
                s[k]*=-1
        memory = np.delete(memory, 0, 0) #we remove the oldest just used
        # We add the previous configuration (before it was modified) 
        # to the LAST position of the pile
        memory = np.concatenate((memory,np.array([stored_config])),axis=0)     
    return(s, memory)

#%% Compute a stationnary distribution

def Stationnary_distribution_CH_fixed_random_history(n,j,epsilon,dt,mem_length, duration, N_config) :
    M_count = np.zeros(n+1) 
    for _ in tqdm(range(N_config)) :
        # Change the Generating function for other starting conditions
        start_config, memory = Generate_fixed_random_history(n, mem_length)
        end_config = Continuous_Glauber_with_Delay(start_config, memory, 0, j, epsilon, duration, dt)[0]
        M = np.sum(end_config)
        M_count[int(M+n)//2]+=1
    return(M_count/N_config)

def Stationnary_distribution_CH_fixed_canonical_history(n,j,epsilon,dt,mem_length, duration, N_config) :
    M_count = np.zeros(n+1) 
    for _ in tqdm(range(N_config)) :
        # Change the Generating function for other starting conditions
        start_config, memory = Generate_fixed_canon_history(n,j, mem_length)
        end_config = Continuous_Glauber_with_Delay(start_config, memory, 0, j, epsilon, duration, dt)[0]
        M = np.sum(end_config)
        M_count[int(M+n)//2]+=1
    return(M_count/N_config)
#%% Plotting the stationnary distributions
Statio_canon_100000 = []
for mem_length in [2,4,8,16,32] : 
    Statio_canon_100000.append(Stationnary_distribution_CH_fixed_canonical_history(n,j,epsilon,dt,mem_length, duration, 100000) )
Statio_canon_100000 = np.array(Statio_canon_100000)


#%% Plotting
plt.plot(mag, Statio_canon_100000[3] , ls = '-', marker = 'o', label = f'$a = {16*dt / epsilon :.2f} $')
plt.plot(mag, Statio_canon_100000[2] , ls = '-', marker = 'o', label = f'$a = {8*dt / epsilon :.2f} $')
plt.plot(mag, Statio_canon_100000[1] , ls = '-', marker = 'o', label = f'$a = {4*dt / epsilon :.2f} $')
plt.plot(mag, Statio_canon_100000[0] , ls = '-', marker = 'o', label = f'$a = {2*dt / epsilon :.2f} $')

#plt.plot(mag, Statio_canon_100000[4] , ls = '-', marker = 'o', label = f'$a = {32*dt / epsilon :.2f} $')

plt.plot(mag,can,color = 'red', ls = '--', marker = 'D', label = 'Canonical distribution\n($a=0$)')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
#Look at the syntax for limited decimals in f strings
#plt.title(f'Steady State of Choi-Hubberman algorithm for different\n values of $a$ with $\\varepsilon = {epsilon} $ and over $10^5$ configurations')
plt.legend(fontsize = 11)
#plt.tight_layout()
#plt.savefig('Steady_State_CH_with_a_notitle.pdf')
plt.plot()

#%% Check to see if dt influences the curve
# We put 32 as mem_length because dt is divided by 2
# We compare with St_dist_dt_01 = np.array([0.04079, 0.09331, 0.13365, 0.15349, 0.16119, 0.15257, 0.13256,
#      0.0936 , 0.03884])

St_dist_dt_005 = Stationnary_distribution_CH_fixed_canonical_history(n,j,epsilon,0.05,32, 10*epsilon, 100000)

#%% Plotting the two curves

St_dist_dt_01 = np.array([0.04079, 0.09331, 0.13365, 0.15349, 0.16119, 0.15257, 0.13256, 0.0936 , 0.03884])

plt.plot(mag, St_dist_dt_01,ls = '-', marker = 'o', label =' $dt = 0.1$')
plt.plot(mag, St_dist_dt_005, ls = '-', marker = 'o', label = '$dt = 0.05$')
plt.plot(mag, can, ls = '--', marker = '.', label = 'canon')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
plt.title(f'Comparing the effect of $dt$ with $a = {1.6 /1.5 :.2f} $')
plt.legend()
plt.show()

#%% Convergence speed for different initial history

# We need to compute the second moment at every time step and plot
# how the distribution converges

def Compute_Second_Moment(mag, proba):
    mean = np.dot(mag, proba)
    cor = (mag - mean)**2
    return np.dot(cor, proba)

def Convergence_speed_random_fixed(n,j,epsilon,dt,mem_length, duration, N_config,N_points) :
    M_count = np.zeros((N_points,n+1)) 
    for _ in tqdm(range(N_config)):
        start_config, memory = Generate_fixed_random_history(n, mem_length)
        M = np.sum(start_config)
        M_count[0, int(M+n)//2]+=1
        inter = start_config
        for i in range(1,N_points):
            inter, memory = Continuous_Glauber_with_Delay(start_config, memory, 0, j, epsilon, duration/N_points, dt)
            M = np.sum(inter)
            M_count[i, int(M+n)//2]+=1
    return(M_count/N_config)

def Convergence_speed_canon_fixed(n,j,epsilon,dt,mem_length, duration, N_config,N_points) :
    M_count = np.zeros((N_points,n+1)) 
    for _ in tqdm(range(N_config)):
        start_config, memory = Generate_fixed_canon_history(n,j, mem_length)
        M = np.sum(start_config)
        M_count[0, int(M+n)//2]+=1
        inter = start_config
        for i in range(1,N_points):
            inter, memory = Continuous_Glauber_with_Delay(start_config, memory, 0, j, epsilon, duration/N_points, dt)
            M = np.sum(inter)
            M_count[i, int(M+n)//2]+=1
    return(M_count/N_config)

#%% Plotting the convergence speed : data
N_points = 20
N_config = 100000
t = np.linspace(0, duration, N_points)

dist_random  = Convergence_speed_random_fixed(n, j, epsilon, dt, mem_length, duration, N_config, N_points)
dist_canon  = Convergence_speed_canon_fixed(n, j, epsilon, dt, mem_length, duration, N_config, N_points)

mom_can = Compute_Second_Moment(mag, can)

mom_random = np.zeros(N_points)
mom_canon = np.zeros(N_points)
for i in range(N_points):
    mom_random[i] = Compute_Second_Moment(mag, dist_random[i])
    mom_canon[i] = Compute_Second_Moment(mag, dist_canon[i])
    
#%% Plotting the convergence speed : actual plot
t =  np.linspace(0, duration/epsilon, 20) 
plt.plot(t, mom_random, ls = '-', marker = 'o', label = 'Random initial history')
plt.plot(t, mom_canon, ls = '-', marker = 'o', label = 'Canonical initial history')
#plt.axvline(x=eps, ls = ':', color = 'g', label = f'epsilon value $\\varepsilon = {eps :.1f} $')
plt.axhline(y=mom_can, ls = '--', color='k', label = 'Canonical (memoryless $a=0$) distribution')
plt.xlabel('Time (in $\\varepsilon$ units)')
plt.ylabel('Second moment\n of magnetization distribution')
#Look at the syntax for limited decimals in f strings
#plt.title(f'Convergence to Steady State with different initial conditions\n $a =$ {mem_length * dt/epsilon :.2f} over $10^{np.log10(N_config) :.0f}$ configurations')
plt.legend(fontsize = 12)
#plt.tight_layout()
#plt.savefig('Convergence_to_Steady_State.pdf')
plt.plot()

#%% Progressive Quenching on Choi-Hubberman

# We need a starting distribution. We SET : eps = 1.5 and mem_length = 16
epsilon = 1.5
mem_length = 16

Steady_distribution = np.array([0.0381225, 0.092735 , 0.133525 , 0.154125 , 0.161385 , 0.1544375,
       0.1339075, 0.0933275, 0.038435 ])

# We generate configurations drawn from this 


def Static_config_8spins(mem_length):
    """
    Returns a list of spins drawn from a pseudo static distribution
    Please make sure that the distribution is rightly chosen.
    Parameters here :
        n=8
        mem_length = 16
        epsilon = 1.5

    """
    n = 8
    if mem_length == 2 :
        static = np.array([0.07489, 0.11667, 0.12477, 0.12233, 0.12204, 0.12344, 0.12508,
               0.11661, 0.07417])
    elif mem_length == 4 :
        static = np.array([0.06734, 0.11182, 0.12686, 0.12874, 0.12844, 0.13013, 0.1272 ,
               0.11213, 0.06734])
    elif mem_length == 8 :
        static = np.array([0.05379, 0.10509, 0.1285 , 0.14074, 0.14234, 0.14034, 0.13007,
               0.10516, 0.05397])
    elif mem_length == 16 :
        static = np.array([0.0381225, 0.092735 , 0.133525 , 0.154125 , 0.161385 , 0.1544375,
               0.1339075, 0.0933275, 0.038435 ])
    else :
        return('mem_length not correct')
    r = random.random()
    Cumul_probability = np.zeros(n+2)  # n+1 values and a 0 (important)
    list_M = np.arange(-n,n+1,2)
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

def PQ_on_Choi_Hubberman_8spins(n,j,epsilon, dt, DeltaT, mem_length, equil_duration, N_configs) :
    '''
    This generates a list of size N_config, with the values of M at the 
    beginning, at equilibrium, and after each quench.
    This should be sufficient to make statistics.
    '''
    if n != 8 :
        print('N is not right')
        return()
    list_M_every_config = np.zeros((N_configs,n+1))
    for i in tqdm(range(N_configs)) :
        evolution_M = []
        config = Static_config_8spins(mem_length)
        evolution_M.append(np.sum(config))
        
        # Here we use non-fixed memory (I think it's faster for convergence)
        # Or is it ?
        memory = []
        for _ in range(mem_length):
            memory.append(Static_config_8spins(mem_length))
        memory = np.array(memory)
        # We make sure we reach equilibrium
        config, memory = Continuous_Glauber_with_Delay(config, memory, 0, j, epsilon, equil_duration, dt)
        evolution_M.append(np.sum(config))
        #Now we quench
        for PQ_index in range(1,n):
            config, memory = Continuous_Glauber_with_Delay(config, memory, PQ_index, j, epsilon, DeltaT, dt)
            evolution_M.append(np.sum(config))
        list_M_every_config[i] = np.array(evolution_M)
    
    return(list_M_every_config)
            
def PQ_on_Choi_Hubberman_fixed_history_8spins(n,j,epsilon, dt, DeltaT, mem_length, equil_duration, N_configs) :
    '''
    This generates a list of size N_config, with the values of M at the 
    beginning, at equilibrium, and after each quench.
    This should be sufficient to make statistics.
    '''
    if n != 8 :
        print('N is not right')
        return()
    list_M_every_config = np.zeros((N_configs,n+1))
    for i in tqdm(range(N_configs)) :
        evolution_M = []
        config = Static_config_8spins(mem_length)
        evolution_M.append(np.sum(config))
        
        # Here we use fixed memory 
        memory = np.array([config]*mem_length)
        # We make sure we reach equilibrium
        config, memory = Continuous_Glauber_with_Delay(config, memory, 0, j, epsilon, equil_duration, dt)
        evolution_M.append(np.sum(config))
        #Now we quench
        for PQ_index in range(1,n):
            config, memory = Continuous_Glauber_with_Delay(config, memory, PQ_index, j, epsilon, DeltaT, dt)
            evolution_M.append(np.sum(config))
        list_M_every_config[i] = np.array(evolution_M)
    
    return(list_M_every_config)
            

#%% Generate data

list_M_after_PQ_8spins_16dt_DeltaT_10_100000 = PQ_on_Choi_Hubberman_8spins(8, 1/8, 1.5, 0.1, 10, 16, 10, 100000)

list_M_after_PQ_8spins_16dt_DeltaT_1_100000 = PQ_on_Choi_Hubberman_8spins(8, 1/8, 1.5, 0.1, 1, 16, 3, 100000)

#list_M_after_PQ_8spins_16dt_DeltaT_15e_100000 = PQ_on_Choi_Hubberman(8, 1/8, 1.5, 0.1, 15*1.5, 16, 15*1.5, 100000)

#%%plotting the data

dist_before_equil = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_1_100000[:,0], bins=8+1)
dist_after_equil = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_1_100000[:,1], bins=8+1)
dist_before_PQ = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_1_100000[:,-1], bins=8+1)

#%%Evolution of the distribution with the number of quenched spins
dist_0 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,0], bins=8+1)[0]/100000
dist_1 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,1], bins=8+1)[0]/100000
dist_2 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,2], bins=8+1)[0]/100000
dist_3 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,3], bins=8+1)[0]/100000
dist_4 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,4], bins=8+1)[0]/100000
dist_5 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,5], bins=8+1)[0]/100000
dist_6 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,6], bins=8+1)[0]/100000
dist_7 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,7], bins=8+1)[0]/100000
dist_8 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_15e_100000[:,8], bins=8+1)[0]/100000

#%%

plt.plot(mag, dist_0, ls = '-', marker = 'o', label = 'Initial steady distribution')
#plt.plot(mag, dist_1, ls = '-', marker = 'o', label = 'After equilibration')
plt.plot(mag, dist_2, ls = '-', marker = 'o', label = '$S_1$ quenched')
plt.plot(mag, dist_3, ls = '-', marker = 'o', label = '$S_2$ quenched')
plt.plot(mag, dist_4, ls = '-', marker = 'o', label = '$S_3$ quenched')
#plt.plot(mag, dist_5, ls = '-', marker = 'o', label = '$S_4$ quenched')
plt.plot(mag, dist_6, ls = '-', marker = 'o', label = '$S_5$ quenched')
#plt.plot(mag, dist_7, ls = '-', marker = 'o', label = '$S_6$ quenched')
plt.plot(mag, dist_8, ls = '-', marker = 'o', label = 'End of PQ')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')

plt.legend(fontsize = 11)

#plt.title(f'Evolution of the magnetization distribution during PQ,\n with $\\Delta T = 10$ and $a = {1.6/1.5 :.2f}$.')
#plt.savefig('CH_Mag_evolution_PQ_8spins.pdf')

plt.show()

#%%



#%%

dist_before_equil_10 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_10_100000[:,0], bins=8+1)
dist_after_equil_10 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_10_100000[:,1], bins=8+1)
dist_before_PQ_10 = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_10_100000[:,-1], bins=8+1)


#%%
plt.plot(mag, dist_before_equil_10[0]/100000,ls = '-', marker = 'o', label = 'Before equilibration')
plt.plot(mag, dist_after_equil_10[0]/100000,ls = '-', marker = 'o',label = 'After $(T=10)$ equilibration\n with CH algorithm')
plt.plot(mag, dist_before_PQ_10[0]/100000, ls = '-', marker = 'o',label = 'After PQ $\\Delta T =10$')
plt.plot(mag, Steady_distribution,ls = '--', marker = '.', label = 'Steady_CH')
plt.plot(mag,can, ls=':', label = '$a=0$ distribution' )

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')


plt.legend()
plt.show()

# Very weird because it looks like we reach a purely random configuration...
# Not at all, just compute 1/2^N and compare to the smallest one
#%% Test of thermalization with fixed hisotry 
# To check if we still get the weird convergence

list_M_after_PQ_8spins_16dt_DeltaT_1_100000_fixed_history = PQ_on_Choi_Hubberman_fixed_history_8spins(8,j,epsilon, 0.1, 1, 16, 3, 100000)

#%%
dist_before_eq_FH = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_1_100000_fixed_history[:,0], bins = 9)[0]/100000
dist_after_eq_FH = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_1_100000_fixed_history[:,1], bins = 9)[0]/100000
dist_after_PQ_FH = np.histogram(list_M_after_PQ_8spins_16dt_DeltaT_1_100000_fixed_history[:,-1], bins = 9)[0]/100000

plt.plot(mag, dist_before_eq_FH,ls = '-', marker = 'o', label = 'Before equilibration')
plt.plot(mag,dist_after_eq_FH ,ls = '-', marker = 'o',label = 'After $(T=3)$ equilibration\n with CH algorithm')
plt.plot(mag,dist_after_PQ_FH , ls = '-', marker = 'o',label = 'After PQ $\\Delta T =1$')
plt.plot(mag, Steady_distribution,ls = '--', marker = '.', label = 'Steady_CH')
plt.plot(mag,can, ls=':', label = '$a=0$ distribution' )

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')


plt.legend()
plt.show()

# Conclusion : Fixed history does not exhibits weird things during thermaliaztion


#%% Second moment plots
#%% Let us plot the magnetizations with respect to DeltaT 

big_list_of_dist = np.zeros((11, 100000, 9))
i=0
for DeltaT in [1,2,3,4,5,8,10,12,15,18,25] :
    print(f'Delta T = {DeltaT}')
    big_list_of_dist[i] = PQ_on_Choi_Hubberman_fixed_history_8spins(n, j, epsilon, dt, DeltaT, 4, 5, 100000)
    i+=1
#%% More points

dist_DT_25 = PQ_on_Choi_Hubberman_fixed_history_8spins(8, 1/8, 1.5, 0.1, 25, 16, 5, 1000)

#%%



#%% Plotting the final distributions

plt.plot(mag, Make_symetric(Steady_distribution),ls = '-', marker = 'o', label = 'Steady State $\\Delta T / \\varepsilon  =0$')

#list_DeltaT = [1,2,3,10,15,18]

plt.plot(mag, np.histogram(big_list_of_dist[0,:,-1], bins = 9)[0]/100000, 
              ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={1/epsilon :.2f}$')

plt.plot(mag, np.histogram(big_list_of_dist[1,:,-1], bins = 9)[0]/100000, 
              ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={2/epsilon :.2f}$')
plt.plot(mag, np.histogram(big_list_of_dist[2,:,-1], bins = 9)[0]/100000, 
              ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={3/epsilon :.2f}$')
plt.plot(mag, np.histogram(big_list_of_dist[6,:,-1], bins = 9)[0]/100000, 
              ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={10/epsilon :.2f}$')
# plt.plot(mag, Make_symetric(np.histogram(big_list_of_dist[8,:,-1], bins = 9)[0]/100000), 
#              ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={15/epsilon :.2f}$')

plt.plot(mag, np.histogram(big_list_of_dist[9,:,-1], bins = 9)[0]/100000, 
              ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={18/epsilon :.2f}$')
plt.plot(mag,can, ls='--', marker ='D' ,color = 'red', label = 'Canonical distribution\n$(a=0)$' )

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')


plt.legend(fontsize = 9.5)
#plt.savefig('CH_Final_dist_PQ_with_DeltaT_colorOK.pdf')
plt.show()

#%%
static4 = np.array([0.06734, 0.11182, 0.12686, 0.12874, 0.12844, 0.13013, 0.1272 ,
       0.11213, 0.06734])


#%% Plotting the moments
mag, can = Canonical_Distribution(8, 1/8)
EM = Compute_Second_Moment(mag, can)
list_DeltaT = np.array([0,1,2,3,4,5,8,10,12,15,18,25])/eps
l = len(list_DeltaT)
list_moments_PQ_CH = np.zeros(l)
list_moments_PQ_CH[0] = Compute_Second_Moment(mag, Steady_distribution)/EM
for i in range(l-2):
    list_moments_PQ_CH[i+1] = Compute_Second_Moment(mag, np.histogram(big_list_of_dist[i,:,-1], bins = 9)[0]/100000)/EM

#list_moments_PQ_CH[6]=24.612250671599995 #Unique for mem_length = 4
list_moments_PQ_CH[-1] = Compute_Second_Moment(mag, np.histogram(dist_DT_25[:,-1],bins = 9)[0]/100000)/EM
plt.plot(list_DeltaT, list_moments_PQ_CH, ls = '-', marker = 'o',color = 'C7', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, Steady_distribution)/EM, ls=':', label = 'Steady Distribution', color='C0')
plt.axhline(1, ls = '--', label = 'Canonical distribution $(a=0)$', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2] / \\mathbf{E}_{can}[M^2]$')
plt.legend(fontsize = 12)

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_colorOK.pdf')

plt.plot()

#%% Compute the transfer matrix for the 2-2 discrete time case:

 
def Transition_proba(b,a,eps, eta):
    '''
    Parameters
    ----------
    b : list
        value of the 4 spins.
        In order : 00,01,10,11
    a : list
        same for the previous configuration.

    Returns
    -------
    The transition probability value.
    '''
    
    c1 = (1+b[1]*a[3])
    c2 = (1+b[0]*a[2])
    b3 = (1-a[2]*a[1]*eta)/eps
    c3 = 1 + b[2]*a[2]*(1-b3)
    b4 = (1-a[3]*a[0]*eta)/eps
    c4 = 1 + b[3]*a[3]*(1-b4)
    return(c1*c2*c3*c4/16)

#%% eigenvalue analysis: 
j = 1/4
eta = np.tanh(j)
    
from scipy.sparse.linalg import eigs

spins = [[-1,-1,-1,-1],[-1,-1,-1,1],[-1,-1,1,-1],[-1,-1,1,1],[-1,1,-1,-1],
         [-1,1,-1,1],[-1,1,1,-1],[-1,1,1,1],[1,-1,-1,-1],[1,-1,-1,1],[1,-1,1,-1],
         [1,-1,1,1],[1,1,-1,-1],[1,1,-1,1],[1,1,1,-1],[1,1,1,1]]

TM = np.zeros((16,16))
for i in range(16):
    for j in range(16):
        TM[i][j] = Transition_proba(spins[i],spins[j],2,eta)
        
steady_state = eigs(TM, k=1, which = 'LM')[1]

steady_b = np.zeros(4)

for k in range(4):
    steady_b[k] = np.real(steady_state[k][0]+steady_state[4+k][0]+steady_state[8+k][0]+steady_state[12+k][0])
    
steady_b /= np.sum(steady_b)

print('j = ', j,'\n++ or --  probability =', steady_b[0], '\n+- or -+  probability =', steady_b[1])

#%% plot of ++ probability vs epsilon 
# Eigenvalue analysis
N = 1000  
list_eps = np.linspace(1, 100, N)
plusplusproba = np.ones(N)
plusminusproba = np.ones(N)
q = 0
for eps in list_eps:
    
    TM = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            TM[i][j] = Transition_proba(spins[i],spins[j],eps,eta)
            
    steady_state = eigs(TM, k=1, which = 'LM')[1]

    steady_b = np.zeros(4)

    for k in range(4):
        steady_b[k] = np.real(steady_state[k][0]+steady_state[4+k][0]+steady_state[8+k][0]+steady_state[12+k][0])
        
    steady_b /= np.sum(steady_b)
    plusplusproba[q] = steady_b[0]
    plusminusproba[q] = steady_b[1]
    
    q+=1
#%% plotting module   
plt.plot(list_eps, plusplusproba, label = '$(++)$ Configuration\nProbability')
plt.axhline(y=(eta+1)/4, ls ='--', color = 'b', label = 'Canonical value $(++)$')
plt.plot(list_eps, plusminusproba, label = '$(+-)$ Configuration\nProbability')
plt.axhline(y=(1-eta)/4, ls ='--', color = 'r', label = 'Canonical value $(+-)$')
# Trying to fit a simple exponential
#plt.plot(list_eps, plusplusproba[0] + (plusplusproba[-1]-plusplusproba[0])*(1-np.exp(-(list_eps-1)/4)))
#


plt.xlabel('$\\varepsilon$ value')
plt.ylabel('Probability value')
plt.legend()
#plt.savefig('CH22_epsilon.pdf')
plt.show()

#%%

for eta in np.tanh(np.array([1,1/2,1/4, 1/6])) :
    N = 100
    list_eps = np.linspace(1, 100, N)
    plusplusproba = np.ones(N)
    plusminusproba = np.ones(N)
    q = 0
    for eps in list_eps:
        
        TM = np.zeros((16,16))
        for i in range(16):
            for j in range(16):
                TM[i][j] = Transition_proba(spins[i],spins[j],eps,eta)
                
        steady_state = eigs(TM, k=1, which = 'LM')[1]
    
        steady_b = np.zeros(4)
    
        for k in range(4):
            steady_b[k] = np.real(steady_state[k][0]+steady_state[4+k][0]+steady_state[8+k][0]+steady_state[12+k][0])
            
        steady_b /= np.sum(steady_b)
        plusplusproba[q] = steady_b[0]
        plusminusproba[q] = steady_b[1]
        
        q+=1
  
    plt.plot(list_eps, plusplusproba, label = f'$\\eta = {eta :.2f}$')
    plt.axhline(y=(eta+1)/4, ls ='--')
    # plt.plot(list_eps, plusminusproba, label = '$(+-)$ Configuration\nProbability')
    # plt.axhline(y=(1-eta)/4, ls ='--', color = 'r', label = 'Canonical value $(+-)$')
    # Trying to fit a simple exponential
    #plt.plot(list_eps, plusplusproba[0] + (plusplusproba[-1]-plusplusproba[0])*(1-np.exp(-(list_eps-1)/4)))
    #


plt.xlabel('$\\varepsilon$ value')
plt.ylabel('Probability value')
plt.legend()
#plt.savefig('CH22_epsilon.pdf')
plt.show()

#%% Plot of the crossing with canonical distributions.

# We want small systems, so N = 4 to reduce computation times. 
# Do we have the steady state configurations ? idk but we need those 
# for all of the possible a's we want to look at. 
# Let us do that :
    
n = 4
j = 1/n
dt = 0.05 # double precision
mem_length = 16 # In units of dt ! 
epsilon = 1.5
eps = epsilon
duration = 10 * epsilon
tau = mem_length * dt

list_mem = np.array([6,9,10,12]) # just for now

list_steady_states = np.zeros((len(list_mem),n+1))
i = 0
for mem_length in list_mem :
    list_steady_states[i] = Stationnary_distribution_CH_fixed_canonical_history(n,j,epsilon,dt,mem_length, duration, 200000)
    i +=1
  #%%  
steady4spins_mem6 = Make_symetric(list_steady_states[0])
steady4spins_mem9 = Make_symetric(list_steady_states[1])
steady4spins_mem10 = Make_symetric(list_steady_states[2])
steady4spins_mem12 = Make_symetric(list_steady_states[3])

#%%%

steady4spins_mem2 = Make_symetric(list_steady_states[0])
steady4spins_mem4 = Make_symetric(list_steady_states[1])
steady4spins_mem8 = Make_symetric(list_steady_states[2])
steady4spins_mem16 = Make_symetric(list_steady_states[3])
#%% Plotting those

#plt.plot(mag, steady4spins_mem2, label = '2')
#plt.plot(mag, steady4spins_mem4, label = '4')
#plt.plot(mag, steady4spins_mem6, label = '6')
plt.plot(mag, steady4spins_mem8, label = '8')
plt.plot(mag, steady4spins_mem9, label = '9')
plt.plot(mag, steady4spins_mem10, label = '10')
plt.plot(mag, steady4spins_mem12, label = '12')
#plt.plot(mag, steady4spins_mem16, label = '16')
plt.plot(mag,can,'--' ,label = 'canon')
plt.legend()
plt.show()


#%% Steadu State configuration generator for 4 spins

def Static_config_4spins(mem_length):
    """
    Returns a list of spins drawn from a pseudo static distribution
    Please make sure that the distribution is rightly chosen.
    Parameters here :
        n=4
        mem_length = 16
        epsilon = 1.5

    """
    n = 4
    if mem_length == 2 :
        static = steady4spins_mem2
    elif mem_length == 4 :
        static = steady4spins_mem4
    elif mem_length == 8 :
        static = steady4spins_mem8
    elif mem_length == 12 :
        static = np.array([0.175357, 0.215265 ,0.218755,0.215265,0.175357])
    elif mem_length == 16 :
        static = steady4spins_mem16
    else :
        return('mem_length not correct')
    r = random.random()
    Cumul_probability = np.zeros(n+2)  # n+1 values and a 0 (important)
    list_M = np.arange(-n,n+1,2)
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

def PQ_on_Choi_Hubberman_fixed_history_4spins(n,j,epsilon, dt, DeltaT, mem_length, equil_duration, N_configs) :
    '''
    This generates a list of size N_config, with the values of M at the 
    beginning, at equilibrium, and after each quench.
    This should be sufficient to make statistics.
    '''
    if n != 4 :
        print('N is not right')
        return()
    list_M_every_config = np.zeros((N_configs,n+1))
    for i in tqdm(range(N_configs)) :
        evolution_M = []
        config = Static_config_4spins(mem_length)
        evolution_M.append(np.sum(config)) #store the first magnetisation
        
        # Here we use fixed memory 
        memory = np.array([config]*mem_length)
        # We make sure we reach equilibrium 
        # to have "plausible" history variations in the configuration dynamics 
        config, memory = Continuous_Glauber_with_Delay(config, memory, 0, j, epsilon, equil_duration, dt)
        evolution_M.append(np.sum(config)) #magnetisation after equilibration
        #Now we quench
        for PQ_index in range(1,n):
            config, memory = Continuous_Glauber_with_Delay(config, memory, PQ_index, j, epsilon, DeltaT, dt)
            evolution_M.append(np.sum(config))
        list_M_every_config[i] = np.array(evolution_M)
    
    return(list_M_every_config)
#%% To make more precise steady state distribution

def Generate_fixed_static_history(mem_length) :
    actual_config = Static_config_4spins(mem_length)
    memory  = np.array([actual_config]*mem_length)
    return(actual_config, memory)


def Stationnary_distribution_CH_fixed_steady_history(n,j,epsilon,dt,mem_length, duration, N_config) :
    M_count = np.zeros(n+1) 
    for _ in tqdm(range(N_config)) :
        # Change the Generating function for other starting conditions
        start_config, memory = Generate_fixed_static_history(mem_length)
        end_config = Continuous_Glauber_with_Delay(start_config, memory, 0, j, epsilon, duration, dt)[0]
        M = np.sum(end_config)
        M_count[int(M+n)//2]+=1
    return(M_count/N_config)



#%% Make the histograms
list_DeltaT = np.array([0,5,10,15])/eps
mem_length = 16
N_config = 200000

big_list_of_dist_4spins = np.zeros((len(list_DeltaT), N_config, n+1))
i=0
for DeltaT in [5,10,15] :
    print(f'Delta T = {DeltaT}')
    big_list_of_dist_4spins[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 5, N_config)
    i+=1
    
    
#%%
mag, can = Canonical_Distribution(n, j)

l = len(list_DeltaT)
list_moments_PQ_CH = np.zeros(l)
list_moments_PQ_CH[0] = Compute_Second_Moment(mag, steady4spins_mem16)
for i in range(l-1):
    list_moments_PQ_CH[i+1] = Compute_Second_Moment(mag, np.histogram(big_list_of_dist_4spins[i,:,-1], bins = n+1)[0]/N_config)
#list_moments_PQ_CH[-1] = Compute_Second_Moment(mag, np.histogram(dist_DT_25[:,-1],bins = 9)[0]/1000)    
plt.plot(list_DeltaT, list_moments_PQ_CH, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem16), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution $(a=0)$', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=8/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a2.pdf')

#%% Redo PQ to make things faster

#%% Complete curve (plot of all things) with all DT for mem = 4

list_DeltaT = np.array([0,1,2,3,4,5,6,7])/eps
mem_length = 4
N_config = 300000

big_list_of_dist_4spins_mem4 = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in [1,2,3,4,5,6,7] :
    print(f'Delta T = {DeltaT}')
    #big_list_of_dist_4spins_mem4[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    i+=1
#%%
plt.plot(mag, steady4spins_mem4, label = 'Steady State')
for i in range(len(list_DeltaT)-1):
    plt.plot(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem4[i,:,-1], bins = n+1)[0]/N_config), 
                  ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={list_DeltaT[i+1] :.2f}$')
plt.plot(mag, can, label = 'Canonical')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
plt.title('$N_0 = 4$ and $ a = 2/15$')

plt.legend(fontsize = 9.5)
#plt.savefig('CH_Final_dist_PQ_with_DeltaT_4mem4.pdf')
plt.show()   

#%%

l = len(np.array([0,1,2,3,4,5,6,7])/eps)
list_moments_PQ_CH = np.zeros(l)
list_moments_PQ_CH[0] = Compute_Second_Moment(mag, steady4spins_mem4)
for i in range(l-1):
    list_moments_PQ_CH[i+1] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem4[i,:,-1], bins = n+1)[0]/N_config))
    
plt.plot(np.array([0,1,2,3,4,5,6,7])/eps, list_moments_PQ_CH, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem4), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=2/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a2_new.pdf')

plt.show()   

#%% mem_length = 2
list_DeltaT = np.array([0,0.5,1,1.5,2,3,4,5,6])/eps
mem_length = 2
N_config = 200000

big_list_of_dist_4spins_mem2 = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in [0.5,1,1.5,2,3,4,5,6] :
    print(f'Delta T = {DeltaT}')
    #big_list_of_dist_4spins_mem2[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    i+=1

plt.plot(mag, steady4spins_mem2, label = 'Steady State')
for i in range(len(list_DeltaT)-1):
    plt.plot(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem2[i,:,-1], bins = n+1)[0]/N_config), 
                  ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={list_DeltaT[i+1] :.2f}$')
plt.plot(mag, can,'--', label = 'Canonical')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
plt.title('$N_0 = 4$ and $ a = 1/15$')

plt.legend(fontsize = 9.5)
plt.savefig('CH_Final_dist_PQ_with_DeltaT_4mem2.pdf')
plt.show()   

#%% Adding more terms

list_DeltaT = np.array([0,7,8])/eps
mem_length = 2
N_config = 200000

big_list_of_dist_4spins_mem2bis = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in [7,8] :
    print(f'Delta T = {DeltaT}')
    big_list_of_dist_4spins_mem2bis[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    i+=1

plt.plot(mag, steady4spins_mem2, label = 'Steady State')
for i in range(len(list_DeltaT)-1):
    plt.plot(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem2bis[i,:,-1], bins = n+1)[0]/N_config), 
                  ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={list_DeltaT[i+1] :.2f}$')

plt.plot(mag, can,'--', label = 'Canonical')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
plt.title('$N_0 = 4$ and $ a = 1/15$')

plt.legend(fontsize = 9.5)
plt.savefig('CH_Final_dist_PQ_with_DeltaT_4mem2bis.pdf')
plt.show()  
#%% Plot with more terms



#%%

l = len(np.array([0,0.5,1,1.5,2,3,4,5,6])/eps)
list_moments_PQ_CH = np.zeros(l+2)
list_moments_PQ_CH[0] = Compute_Second_Moment(mag, steady4spins_mem2)
for i in range(l-1):
    list_moments_PQ_CH[i+1] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem2[i,:,-1], bins = n+1)[0]/N_config))

for i in [0,1]:
    list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem2bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(np.array([0,0.5,1,1.5,2,3,4,5,6,7,8])/eps, list_moments_PQ_CH, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem2), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=1/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a1.pdf')

plt.show()   


#%% memlength = 16

list_DeltaT = np.arange(0,11)*eps
mem_length = 16
N_config = 200000

big_list_of_dist_4spins_mem16 = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in np.arange(1,11)*eps :
    print(f'Delta T = {DeltaT}')
    big_list_of_dist_4spins_mem16[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    dss = Make_symetric(np.histogram(big_list_of_dist_4spins_mem16[i,:,-1], bins = n+1)[0]/N_config)
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, a = 16/15$ and $\Delta T / \epsilon $ = {DeltaT / eps :.2f}')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1
#%%
plt.plot(mag, steady4spins_mem16, label = 'Steady State')
for i in range(len(list_DeltaT)-1):
    plt.plot(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem16[i,:,-1], bins = n+1)[0]/N_config), 
                  ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={list_DeltaT[i+1] :.2f}$')
plt.plot(mag, can,'--', label = 'Canonical')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
plt.title('$N_0 = 4$ and $ a = 8/15$')

#plt.legend(fontsize = 9.5)
#plt.savefig('CH_Final_dist_PQ_with_DeltaT_4mem16.pdf')
plt.show()   

#%%

l = 11
list_moments_PQ_CH = np.zeros(l)
list_moments_PQ_CH[0] = Compute_Second_Moment(mag, steady4spins_mem16)
for i in range(l-1):
    list_moments_PQ_CH[i+1] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem16[i,:,-1], bins = n+1)[0]/N_config))

# for i in [0,1]:
#     list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem2bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(np.arange(0,11), list_moments_PQ_CH, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem16), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=8/15$")

#@plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a16.pdf')

plt.show()   

#%% More points memlength = 16

list_DeltaT = np.arange(11,15)*eps
mem_length = 16
N_config = 200000

big_list_of_dist_4spins_mem16bis = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in np.arange(11,15)*eps :
    print(f'Delta T = {DeltaT}')
    big_list_of_dist_4spins_mem16bis[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    dss = Make_symetric(np.histogram(big_list_of_dist_4spins_mem16bis[i,:,-1], bins = n+1)[0]/N_config)
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, a = 16/15$ and $\Delta T / \epsilon $ = {DeltaT / eps :.2f}')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1
#%%
plt.plot(mag, steady4spins_mem16, label = 'Steady State')
for i in range(len(list_DeltaT)-1):
    plt.plot(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem16bis[i,:,-1], bins = n+1)[0]/N_config), 
                  ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={list_DeltaT[i+1] :.2f}$')
plt.plot(mag, can,'--', label = 'Canonical')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
plt.title('$N_0 = 4$ and $ a = 8/15$')

#plt.legend(fontsize = 9.5)
#plt.savefig('CH_Final_dist_PQ_with_DeltaT_4mem16.pdf')
plt.show()   

#%%

l = 11
list_moments_PQ_CH = np.zeros(l+3)
list_moments_PQ_CH[0] = Compute_Second_Moment(mag, steady4spins_mem16)
for i in range(l-1):
    list_moments_PQ_CH[i+1] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem16[i,:,-1], bins = n+1)[0]/N_config))

for i in range(0,3):
    list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem16bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(np.arange(0,14), list_moments_PQ_CH, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem16), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=8/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a16_new.pdf')

plt.show() 


#%% mem_length 8
list_DeltaT = np.array([8,10,15])*eps
mem_length = 8
N_config = 300000

big_list_of_dist_4spins_mem8bis = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in list_DeltaT :
    print(f'Delta T = {DeltaT}')
    big_list_of_dist_4spins_mem8bis[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    dss = Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config)
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, a = 8/15$ and $\Delta T / \epsilon $ = {DeltaT / eps :.2f}')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1
#%%

plt.plot(mag, steady4spins_mem8, label = 'Steady State')
for i in range(len(list_DeltaT)-1):
    plt.plot(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config), 
                  ls = '-', marker = 'o',label = f'After PQ $\\Delta T / \\varepsilon  ={list_DeltaT[i+1] :.2f}$')
plt.plot(mag, can,'--', label = 'Canonical')

plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
plt.title('$N_0 = 4$ and $ a = 4/15$')

plt.legend(fontsize = 9.5)
#plt.savefig('CH_Final_dist_PQ_with_DeltaT_4mem8bis.pdf')
plt.show() 

#%%

l = 5
list_moments_PQ_CH = np.zeros(l+2)
list_moments_PQ_CH[0] = Compute_Second_Moment(mag, steady4spins_mem8)
for i in range(l-1):
    list_moments_PQ_CH[i+1] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8[i,:,-1], bins = n+1)[0]/N_config))

for i in [0,1]:
    list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config))

list_DeltaTfull = np.array([0,4,5,6,7,8,10])*eps

plt.plot(list_DeltaTfull/eps, list_moments_PQ_CH, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem8), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=4/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a8ter.pdf')

plt.show()   

#%% mem_length 32

n = 4
j = 1/n
dt = 0.1 # single precision
mem_length = 16 # In units of dt ! 
epsilon = 1.5
eps = epsilon
duration = 10 * epsilon
tau = mem_length * dt
#%% Do not touch

steady4spins_mem32 = #Stationnary_distribution_CH_fixed_canonical_history(n,j,epsilon,dt,mem_length, duration, 400000)

#%%
steady4spins_mem64 = #Stationnary_distribution_CH_fixed_canonical_history(n,j,epsilon,0.1,32, duration, 400000)

#%%

steady4spins_mem24 = np.array([0.14816625, 0.2281975 , 0.2472725 , 0.2281975 , 0.14816625]) 
#Stationnary_distribution_CH_fixed_canonical_history(n,j,epsilon,0.1,12, duration, 400000)

steady4spins_mem24 = Make_symetric(steady4spins_mem24)
#%%

steady4spins_mem32 = Make_symetric(steady4spins_mem32)

#%%

def Static_config_4spins_32(mem_length):
    """
    Returns a list of spins drawn from a pseudo static distribution
    Please make sure that the distribution is rightly chosen.
    Parameters here :
        n=4
        dt = 0.1
        epsilon = 1.5

    """
    n = 4
    # if mem_length == 2 :
    #     static = steady4spins_mem2
    # elif mem_length == 4 :
    #     static = steady4spins_mem4
    # elif mem_length == 8 :
    #     static = steady4spins_mem8
    if mem_length == 16 :
        static = steady4spins_mem32 #WATCH OUT 
    elif mem_length == 32 :    
        static = steady4spins_mem64
    elif mem_length == 12 :
        static = np.array([0.14816625, 0.2281975 , 0.2472725 , 0.2281975 , 0.14816625])
    else :
        return('mem_length not correct')
    r = random.random()
    Cumul_probability = np.zeros(n+2)  # n+1 values and a 0 (important)
    list_M = np.arange(-n,n+1,2)
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

def PQ_on_Choi_Hubberman_fixed_history_4spins32(n,j,epsilon, dt, DeltaT, mem_length, equil_duration, N_configs) :
    '''
    This generates a list of size N_config, with the values of M at the 
    beginning, at equilibrium, and after each quench.
    This should be sufficient to make statistics.
    '''
    if (n != 4 and dt != 0.1) :
        print('N or a or dt is not right')
        return()
    list_M_every_config = np.zeros((N_configs,n+1))
    for i in tqdm(range(N_configs)) :
        evolution_M = []
        config = Static_config_4spins_32(mem_length)
        evolution_M.append(np.sum(config)) #store the first magnetisation
        
        # Here we use fixed memory 
        memory = np.array([config]*mem_length)
        # We make sure we reach equilibrium 
        # to have "plausible" history variations in the configuration dynamics 
        config, memory = Continuous_Glauber_with_Delay(config, memory, 0, j, epsilon, equil_duration, dt)
        evolution_M.append(np.sum(config)) #magnetisation after equilibration
        #Now we quench
        for PQ_index in range(1,n):
            config, memory = Continuous_Glauber_with_Delay(config, memory, PQ_index, j, epsilon, DeltaT, dt)
            evolution_M.append(np.sum(config))
        list_M_every_config[i] = np.array(evolution_M)
    
    return(list_M_every_config)


#%%

list_DeltaT = np.arange(16)*eps
#mem_length = 16
N_config = 300000

big_list_of_dist_4spins_mem32 = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in np.delete(list_DeltaT, 0) :
    print(f'Delta T = {DeltaT/eps}')
    big_list_of_dist_4spins_mem32[i] = PQ_on_Choi_Hubberman_fixed_history_4spins32(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    dss = Make_symetric(np.histogram(big_list_of_dist_4spins_mem32[i,:,-1], bins = n+1)[0]/N_config)
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, a = 16/15$ and $\Delta T / \epsilon $ = {DeltaT / eps :.2f}')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1

#%%

l = len(list_DeltaT)-1
list_moments_PQ_CH = np.zeros(l+1)
list_moments_PQ_CH[0] = Compute_Second_Moment(mag, steady4spins_mem32)
for i in range(1,l+1):
    list_moments_PQ_CH[i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem32[i-1,:,-1], bins = n+1)[0]/N_config))

# for i in [0,1]:
#     list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(list_DeltaT/eps, list_moments_PQ_CH, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem32), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=16/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a16.pdf')

plt.show() 

#%% Delta T constant, make $a$ variate.
N_config = 300000

big_list_4spins_DeltaT10eps = np.zeros((6, N_config, n+1)) # 6 points plus zero
i = 0
for mem_length in [2,4,8,16] :
    big_list_4spins_DeltaT10eps[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, 0.05, 10*eps, mem_length, 2, N_config)
    dss = np.histogram(big_list_4spins_DeltaT10eps[i,:,-1], bins = n+1)[0]/N_config
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, \Delta T / \epsilon = 10 $ and $ a = {mem_length*0.5 :.2f}/15$')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1

for mem_length in [16,32] :
    big_list_4spins_DeltaT10eps[i] = PQ_on_Choi_Hubberman_fixed_history_4spins32(n, j, epsilon, 0.1, 10*eps, mem_length, 2, N_config)
    dss = np.histogram(big_list_4spins_DeltaT10eps[i,:,-1], bins = n+1)[0]/N_config
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, \Delta T / \epsilon = 10 $ and $ a = {mem_length :.1f}/15$')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1

#%% test only for 32

#test_dss = PQ_on_Choi_Hubberman_fixed_history_4spins32(n, j, epsilon, 0.1, 10, 32, 2, N_config)
    
#%%

list_moments_PQ_CHDT10 = np.zeros(7)
list_moments_PQ_CHDT10[0] = Compute_Second_Moment(mag, can)
for i in range(1,7):
    list_moments_PQ_CHDT10[i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_4spins_DeltaT10eps[i-1,:,-1], bins = n+1)[0]/N_config))

# for i in [0,1]:
#     list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(np.array([0,0.1,0.2,0.4,0.8,1.6,3.2])/eps, list_moments_PQ_CHDT10, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$a$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title(f"$N_0 = 4$ and $\Delta T / \epsilon = {10 :.1f} $")

#plt.savefig('CH_Second_moments_PQ_with_a_DeltaT10.pdf')

plt.show() 

#%% mem_length 64

list_DeltaT = np.arange(16)*eps
mem_length = 32
N_config = 300000

big_list_of_dist_4spins_mem64 = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in np.delete(list_DeltaT, 0) :
    print(f'Delta T = {DeltaT/eps}')
    big_list_of_dist_4spins_mem64[i] = PQ_on_Choi_Hubberman_fixed_history_4spins32(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    dss = Make_symetric(np.histogram(big_list_of_dist_4spins_mem64[i,:,-1], bins = n+1)[0]/N_config)
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, a = 16/15$ and $\Delta T / \epsilon $ = {DeltaT / eps :.2f}')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1


l = len(list_DeltaT)-1
list_moments_PQ_CH64 = np.zeros(l+1)
list_moments_PQ_CH64[0] = Compute_Second_Moment(mag, steady4spins_mem64)
for i in range(1,l+1):
    list_moments_PQ_CH64[i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem64[i-1,:,-1], bins = n+1)[0]/N_config))

# for i in [0,1]:
#     list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(list_DeltaT/eps, list_moments_PQ_CH64, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem64), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=32/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a64.pdf')

plt.show() 

#%% Redo mem_length = 2 
# First, recompute the stationary distribution with many points and starting 
# from the previous steady state,

# Then for Delta T range(0,16) : Compute with 400000 points.




#%% Confirm the supercanonical case 

n = 8
j = 1/n
dt = 0.1
mem_length = 16
epsilon = 1.5
eps = epsilon
duration = 10 * epsilon
tau = mem_length * dt

mag, can = Canonical_Distribution(n, j)

plt.style.use('bmh')


#%% First redo the static curve

def Generate_fixed_statio_history(n,j,mem_length) :
    actual_config = Static_config_8spins(mem_length)
    memory  = np.array([actual_config]*mem_length)
    return(actual_config, memory)

def Stationnary_distribution_CH_fixed_statio_history(n,j,epsilon,dt,mem_length, duration, N_config) :
    M_count = np.zeros(n+1) 
    for _ in tqdm(range(N_config)) :
        # Change the Generating function for other starting conditions
        start_config, memory = Generate_fixed_statio_history(n,j, mem_length)
        end_config = Continuous_Glauber_with_Delay(start_config, memory, 0, j, epsilon, duration, dt)[0]
        M = np.sum(end_config)
        M_count[int(M+n)//2]+=1
    return(M_count/N_config)

#%%
#steady_state_8_16 = Stationnary_distribution_CH_fixed_statio_history(8,1/8,epsilon,0.1,16, duration, 400000)

#%%
N_config = 500000
supercanon = PQ_on_Choi_Hubberman_fixed_history_8spins(n, j, epsilon, 0.1, 10*1.5, 16, 2, N_config)

#%%
N_config = 500000
dss = np.histogram(supercanon[:,-1], bins = n+1)[0]/N_config

plt.plot(mag,can)
plt.plot(mag, dss, '-.')
plt.show()

print(f'E[M^2] = {Compute_Second_Moment(mag, dss) :.3f} and expected over {Compute_Second_Moment(mag, can) :.2f}')
#%%

N_config = 400000
supercanon2 = PQ_on_Choi_Hubberman_fixed_history_8spins(n, j, epsilon, 0.1, 15*1.5, 16, 2, N_config)

#%%
N_config = 400000
dss2 = np.histogram(supercanon2[:,-1], bins = n+1)[0]/N_config

plt.plot(mag,can)
plt.plot(mag, dss2, '-.')
plt.show()

print(f'E[M^2] = {Compute_Second_Moment(mag, dss2) :.3f} and expected over {Compute_Second_Moment(mag, can) :.2f}')

#%%3D plot 
#plt.style.use('_mpl-gallery')

xs = np.array([0]*16 + [1/15]*11 + [2/15] * 8 + [4/15] * 7 + [8/15] * 14  +[16/15] * 16 + [32/15]*16) # values of $a$
# For $a$ = 0, we have a canonical so we can completely fill up this row
ys = np.concatenate((np.arange(0,16),np.array([0,0.5,1,1.5,2,3,4,5,6,7,8])/eps, np.arange(0,8)/eps, np.array([0,4,5,6,7,8,10]), np.arange(0,14), np.arange(0,16),np.arange(0,16)))

xsbis = np.array([0]*16 + [1/15]*11 + [2/15] * 8 + [4/15] * 13 + [8/15] * 14 +[12/15]*4 + [16/15] * 16 + [32/15]*16) # values of $a$
ysbis = np.concatenate((np.arange(0,16),np.array([0,0.5,1,1.5,2,3,4,5,6,7,8])/eps, np.arange(0,8)/eps, np.arange(0,13), np.arange(0,14), np.array([0,9,10,11]), np.arange(0,16),np.arange(0,16)))
# For every $a$, we put the values of Delta T where we have computed some points.
  
SecMom0 = [Compute_Second_Moment(mag, can)]*16 # a = 0 It is only canonical
SecMom2 = np.array([8.24758, 8.30878, 8.35392, 8.41712, 8.43794, 8.49584, 8.497  ,
       8.49886, 8.4865 , 8.50686, 8.52274])
SecMom4 = np.array([8.01566, 8.26624, 8.364  , 8.4289 , 8.4486 , 8.4537 , 8.48902,
       8.5046 ])
SecMom8 = np.array([7.67896   , 8.41657333, 8.4682    , 8.49906667, 8.48704   ,
       8.50172   , 8.5278    ])

#SecMom8bis = np.array([7.67896 , 8.141, 8.353, 8.437, 8.478, 8.502, 8.521, 8.524, 8.525, 8.519, 8.550, 8.522, 8.526])
SecMom8bis = np.array([7.67896 , 8.141, 8.353, 8.437, 8.478, 8.502, 8.521, (3/13*8.524 + 10/13*8.515), (3/13*8.525 +8.504*10/13), (3/13*8.519 + 10/13*8.530), (3/13*8.550 + 10/13*8.529), (3/13*8.522+ 8.512*10/13), 8.526])

SecMom16 = np.array([7.03632, 7.80488, 8.15062, 8.32676, 8.41422, 8.45164, 8.48232,
       8.51768, 8.53062, 8.52176, 8.50552, 8.55022, 8.51368, 8.53172])
SecMom32 = np.array([6.19442   , 7.11716   , 7.77546667, 8.10808   , 8.24250667,
       8.34116   , 8.43748   , 8.47144   , 8.48761333, 8.52330667,
       8.51222667, 8.53012   , 8.53701333, 8.55005333, 8.55356   ,
       8.53265333])
SecMom64 = np.array([5.49534992, 6.28506667, 6.97528   , 7.6382    , 7.90666667,
       8.07996   , 8.22681333, 8.33222667, 8.36012   , 8.39417333,
       8.45970667, 8.46728   , 8.49334667, 8.49238667, 8.50229333,
       8.50698667])

SecMom24 = np.array([Compute_Second_Moment(mag,steady4spins_mem24 ),8.510, 8.513, 8.525 ])
  
zs = np.concatenate((SecMom0,SecMom2,SecMom4,SecMom8,SecMom16,SecMom32,SecMom64))

zsbis  = np.concatenate((SecMom0,SecMom2,SecMom4,SecMom8bis,SecMom16,SecMom24,SecMom32,SecMom64))

EM2 = Compute_Second_Moment(mag, can)
zsbis = zsbis / EM2
#%%


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(xsbis, ysbis, zsbis/EM2)
ax.set(xlabel='$a$ value',
       ylabel='$\Delta T / \epsilon $ value',
       zlabel='Second moment $\\mathbf{E}[M^2]$')

plt.show()

#%%
from matplotlib import cm

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_trisurf(xs, ys, zs, vmin=zs.min() , cmap=cm.Blues)

ax.set(xlabel='$a$ value',
       ylabel='$\Delta T / \epsilon $ value',
       zlabel='Second moment $\\mathbf{E}[M^2]$')

plt.show()

#%%redo memlength 8
n = 4
j = 1/n
dt = 0.05
list_DeltaT = np.arange(16)*eps
mem_length = 8
N_config = 1000000

big_list_of_dist_4spins_mem8 = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in np.delete(list_DeltaT, 0) :
    print(f'Delta T = {DeltaT/eps}')
    big_list_of_dist_4spins_mem8[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    dss = Make_symetric(np.histogram(big_list_of_dist_4spins_mem8[i,:,-1], bins = n+1)[0]/N_config)
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, a = 4/15$ and $\Delta T / \epsilon $ = {DeltaT / eps :.2f}')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1


l = len(list_DeltaT)-1
list_moments_PQ_CH8 = np.zeros(l+1)
list_moments_PQ_CH8[0] = Compute_Second_Moment(mag, steady4spins_mem8)
for i in range(1,l+1):
    list_moments_PQ_CH8[i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8[i-1,:,-1], bins = n+1)[0]/N_config))

# for i in [0,1]:
#     list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(list_DeltaT/eps, list_moments_PQ_CH8, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem8), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=4/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a8_bis.pdf')

plt.show() 

#%% do some points with mem 12

n = 4
j = 1/n
dt = 0.1
list_DeltaT = np.array([0,9,10,11])*eps
mem_length = 12
N_config = 300000

big_list_of_dist_4spins_mem24 = np.zeros((len(list_DeltaT)-1, N_config, n+1))
i=0
for DeltaT in np.delete(list_DeltaT, 0) :
    print(f'Delta T / e = {DeltaT/eps}')
    big_list_of_dist_4spins_mem24[i] = PQ_on_Choi_Hubberman_fixed_history_4spins32(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    dss = Make_symetric(np.histogram(big_list_of_dist_4spins_mem24[i,:,-1], bins = n+1)[0]/N_config)
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, a = 12/15$ and $\Delta T / \epsilon $ = {DeltaT / eps :.2f}')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1


l = len(list_DeltaT)-1
list_moments_PQ_CH24 = np.zeros(l+1)
list_moments_PQ_CH24[0] = Compute_Second_Moment(mag, steady4spins_mem24)
for i in range(1,l+1):
    list_moments_PQ_CH24[i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem24[i-1,:,-1], bins = n+1)[0]/N_config))

# for i in [0,1]:
#     list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(list_DeltaT/eps, list_moments_PQ_CH24, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem24), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 12)
plt.title("$N_0 = 4$ and $a=12/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a24.pdf')

plt.show() 

#%% contour plot
c = 3 #To be kept constant
atilde1 = 0.2
atilde2 = 0.05
plt.tricontour(xsbis, ysbis, zsbis, levels = np.array([zsbis.min(),0.90,0.95, 0.999, 1.001,1.003, zsbis.max()]), colors = ['C5','C0', 'C1', 'red', 'C3'], linewidths = 1.3)
plt.plot(xsbis, ysbis, 'k1', label = 'Computed data \npoint')
#plt.plot(xsbis[40:], (1/(c*atilde1))*(xsbis[40:]-atilde1),ls = ':', color = 'C1', label = '$\\tilde{a} = 0.2$ and $ c = 3.5$')
#plt.plot(xsbis[30:], (1/(c*atilde2))*(xsbis[30:]-atilde2),ls = ':', color = 'C2')



plt.text(1.8, 1.85, '$0.9 $',fontsize = '12' ,bbox=dict(boxstyle="round", fc="0.9", alpha = 0.8, color = 'C0'), color='C0')
plt.text(1.8, 3.6, '$0.95 $',fontsize = '12' , bbox=dict(boxstyle="round", fc="0.9", alpha = 0.8, color = 'C1'), color = 'C1')
plt.text(1.8, 10.6, '$1$',fontsize = '12' , bbox=dict(boxstyle="round", fc="0.9", alpha = 0.8, color = 'red'), color = 'red')
plt.text(0.3, 13.8, '$1.001 $',fontsize = '12' , bbox=dict(boxstyle="round", fc="0.9", alpha = 0.8, color = 'C3'), color = 'C3')
plt.text(0.8, 13.5, '$1.003 $',fontsize = '12' , bbox=dict(boxstyle="round", fc="0.9", alpha = 0.8, color = 'C5'), color = 'C5')
plt.xlabel('$a$', fontsize = '15')
plt.xlim(-0.05,2.2)
plt.ylim(0,15)
plt.ylabel('$ \\Delta T/ \\varepsilon$')
plt.legend(loc ="upper right", fontsize = 10)
#plt.savefig('Contours.pdf')
plt.show()

#%% Again mem 8 with large things


n = 4
j = 1/n
dt = 0.05
list_DeltaT = np.arange(7,12)*eps
mem_length = 8
N_config = 1000000

big_list_of_dist_4spins_mem8_2 = np.zeros((len(list_DeltaT), N_config, n+1))
i=0
for DeltaT in list_DeltaT :
    print(f'Delta T = {DeltaT/eps}')
    big_list_of_dist_4spins_mem8_2[i] = PQ_on_Choi_Hubberman_fixed_history_4spins(n, j, epsilon, dt, DeltaT, mem_length, 2, N_config)
    dss = Make_symetric(np.histogram(big_list_of_dist_4spins_mem8_2[i,:,-1], bins = n+1)[0]/N_config)
    plt.plot(mag, dss)
    plt.title(f'$N_0 = 4, a = 4/15$ and $\Delta T / \epsilon $ = {DeltaT / eps :.2f}')
    plt.show()
    print(f'E value = {Compute_Second_Moment(mag, dss) :.3f}')
    i+=1


l = len(list_DeltaT)
list_moments_PQ_CH8_2 = np.zeros(l)
#list_moments_PQ_CH8_2[0] = Compute_Second_Moment(mag, steady4spins_mem8)
for i in range(l):
    list_moments_PQ_CH8_2[i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8_2[i,:,-1], bins = n+1)[0]/N_config))

# for i in [0,1]:
#     list_moments_PQ_CH[l+i] = Compute_Second_Moment(mag, Make_symetric(np.histogram(big_list_of_dist_4spins_mem8bis[i,:,-1], bins = n+1)[0]/N_config))

plt.plot(list_DeltaT/eps, list_moments_PQ_CH8_2, ls = '-', marker = 'o', label = 'PQ distribution')

plt.axhline(Compute_Second_Moment(mag, steady4spins_mem8), ls='--', label = 'Steady Distribution', color='green')
plt.axhline(Compute_Second_Moment(mag, can), ls = '--', label = 'Canonical distribution', color='red')
plt.xlabel('$\\Delta T / \\varepsilon$ value')
plt.ylabel('Second moment $\\mathbf{E}[M^2]$')
plt.legend(fontsize = 11)
plt.title("$N_0 = 4$ and $a=4/15$")

#plt.savefig('CH_Second_moments_PQ_with_DeltaT_4a8_bis.pdf')

plt.show() 


#%% Contour plot model
k1 = 0.05
k2 = 0.01
a, b = np.meshgrid(np.linspace(0,20,200), np.linspace(0,15,200))
c = np.exp(a * ( k2*b - k1 * a))
levels = np.linspace(c.min(), c.max(), 10)

fig, ax = plt.subplots()
cs =  ax.contourf(a,b,c, levels = levels)
cbar = fig.colorbar(cs)
plt.show()

#%%

k = 8
a, b = np.meshgrid(np.linspace(0,2,50), np.linspace(0,15,50))
c = np.exp(a * ( k2*b - k1 * a))
plt.pcolormesh(a,b,c)
plt.show()