#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:46:44 2022

@author: CMoslonka

Goal : Make rapid simulations of Choi-hubberman systems for 4 spins
"""
import numpy as np
import random
import math
import time
import ast

import matplotlib.pyplot as plt

from tqdm import tqdm

n = 3
j = 1/n
tau = 0.8
T = 100
dt = 0.05
eps = 1.5

param = n, j, tau, T, dt, eps
epsilon = eps
duration = T


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

def Generate_canoncial_config(n,j):
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

def Polarised_canonical_distribution(n,j,T,M):
    """
    Returns the equilibrium distribution of n spins with coupling j
    with condition on the magnetisation M of T of the spins.

    """
    list_mu=np.arange(-(n-T),n-T+1,2)
    Pth=np.zeros(n-T+1)
    for i in range(n+1-T):
        mu=round(list_mu[i])
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
    for i in np.arange(0,duration,dt):
        stored_config=np.copy(s)
        for k in range(PQ_index, n): #We act of those spins
            if random.random()<Flip_probability_Continuous_Delay(s, memory[0], k, j, epsilon, dt):
                s[k]*=-1
        # k = random.randround(PQ_index, n-1)
        # if random.random()<Flip_probability_Continuous_Delay(s, memory[0], k, j, epsilon, dt):
        #     s[k]*=-1
        memory = np.delete(memory, 0, 0) #we remove the oldest just used
        memory = np.concatenate((memory,np.array([stored_config])),axis=0)
        
    return(s, memory)



def PQ_on_Continuous_Glauber_with_Delay(n,j,tau,duration,dt,epsilon,memory):
    """
    This function does a complete Progresive Quenching simulation on a 
    Choi-Hubberman system.
    
    The memory needs to be specified explicitely.
    
    """
    actual_spin_list=Generate_canoncial_config(n, j)
    for PQ_index in range(1,n):
        actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, PQ_index, j, epsilon, duration, dt)
        
    return(actual_spin_list)




def Static_config():
    """
    Returns a list of spins drawn from a pseudo static distribution
    Please make sure that the distribution is rightly chosen.
    Parameters here :
        n=3
        tau = 0.8
        dt = 0.05
        epsilon = 1.5

    """
    n = 3

    static = np.array([0.22775667, 0.27315   , 0.27278667, 0.22630667])

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

def Static_history_generator(n, tau, dt):
    history = np.zeros((round(tau/dt),n))
    for i in range(round(tau/dt)) :
        history[i] = Static_config()
        
    return(history)

def Generate_static_configuration(n,j,tau,duration,dt,epsilon):
    
    memory=Static_history_generator(n, tau, dt)
    actual_spin_list = Static_config()
    actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, 0 , j, epsilon, duration, dt)
    return(np.sum(actual_spin_list))

def Generate_partial_PQ_static_configuration(n,j,tau,duration,dt,epsilon):
    
    memory=Static_history_generator(n, tau, dt)
    actual_spin_list = Static_config()
    for PQ_index in range(0,4): #You can modify things here
        actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, PQ_index , j, epsilon, duration, dt)
    return(np.sum(actual_spin_list))


def Generate_PQ_static_configuration(n,j,tau,duration,dt,epsilon):
    
    memory=Static_history_generator(n, tau, dt)
    actual_spin_list = Static_config()
    for PQ_index in range(n):
        actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, PQ_index , j, epsilon, duration, dt)
    return(np.sum(actual_spin_list))

def Static_Distribution(n, j, tau, duration, dt, epsilon, size) :
    
    list_M = np.arange(-n,n+1,2)
    M_count = np.zeros(n+1)
    for i in tqdm(range(size)) :
        M=Generate_static_configuration(n, j, tau, duration, dt, epsilon)
        M_count[round(M+n)//2] += 1
    M_count = M_count/size
    return(list_M, M_count)

def Static_partial_PQ_Distribution(n, j, tau, duration, dt, epsilon, size) :
    
    list_M = np.arange(-n,n+1,2)
    M_count = np.zeros(n+1)
    for i in tqdm(range(size)) :
        M=Generate_partial_PQ_static_configuration(n, j, tau, duration, dt, epsilon)
        M_count[round(M+n)//2] += 1
    M_count = M_count/size
    return(list_M, M_count)

def Generate_static_configuration_to_write_in_file(n,j,tau,duration,dt,epsilon):
    
    memory=Static_history_generator(n, tau, dt)
    actual_spin_list = Static_config()
    actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, 0 , j, epsilon, duration, dt)
    return(actual_spin_list, memory)

def Write_history_in_file(n,j,tau,duration,dt,epsilon, size) :
    
    file = open(f'History_file_n={n}T={duration}size={size}.txt', 'w')
    file.write('[')
    for i in tqdm(range(size)):
        actual_spin_list, memory = Generate_static_configuration_to_write_in_file(n, j, tau, duration, dt, epsilon)
        file.write('[')
        file.write(str(memory).replace('\n',','))
        file.write(',')
        file.write(str(actual_spin_list))
        file.write(']')
        if i!= size-1 : file.write(', \n')
    file.write(']')
    file.close()
    
def random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile, 2):
        if random.randrange(num):
            continue
        line = aline
    return line

#%% Non-Markovian detailed balance simulation, with PQ
'''
Everything here is for 3 + 3 spins config only

'''

def Hidden_flip_proba_3spins(spins, flip_index, J1, J2, dt, epsilon) :
    sk = spins[flip_index]
    s1, s2, s3, s4, s5, s6 = spins
    if  flip_index == 0:
        Ekprime = J1*(s2+s3) + J2*(s4+s5)
    elif flip_index == 1:
        Ekprime = J1*(s1+s3) + J2*(s4+s6)
    elif flip_index == 2:
        Ekprime = J1*(s2+s1) + J2*(s6+s5)
    elif flip_index == 3:
        Ekprime = J2*(s1+s2)
    elif flip_index == 4:
        Ekprime = J2*(s1+s3)
    elif flip_index == 5:
        Ekprime = J2*(s3+s2)
    p=(1 - (sk * np.tanh(Ekprime))) * dt
    
    return(p/(2*epsilon))

def Hidden_Glauber_step_3spins(spins, J1, J2, dt, epsilon, PQ_index=0) :
    for k in range(PQ_index,6):
        r = random.random()
        if r < Hidden_flip_proba_3spins(spins, k, J1, J2, dt, epsilon) :
            spins[k] *= -1
    return(spins)
def Hidden_canonical_simulation(J1,J2,dt,epsilon,T,size):
    mag = np.arange(-3,4,2)
    count_1 = np.zeros(4)
    count_2 = np.zeros(4)
    
    for _ in tqdm(range(size)) :
        spins = Generate_random_config(6)
        for t in np.arange(0, T, dt) :
            spins = Hidden_Glauber_step_3spins(spins, J1, J2, dt, epsilon)
        visible_mag = np.sum(spins[0:3])
        hidden_mag = np.sum(spins[3:6])
        #total_mag = np.sum(spins)
        count_1[round((visible_mag+3)/2)] +=1
        count_2[round((hidden_mag+3)/2)] +=1
    count_1 /= size
    count_2 /= size
    return(mag, count_1, count_2)

def Effective_coupling(J1,J2):
    return J1 + 0.5*np.log(np.cosh(2*J2))

def Hidden_PQ_protocol(J1,J2,dt,epsil,deltaT,size) :
    """
    deltaT is the time between 2 quenches
    """
    mag = np.arange(-3,4,2)
    count_1 = np.zeros(4)
    count_2 = np.zeros(4)
    #count_3 = np.zeros(4)
    for _ in tqdm(range(size)) :
        spins = Generate_random_config(6)
        eps = epsil
        effJ1 = J1
        effJ2 = J2
        effJ = Effective_coupling(J1, J2) #Should be constant
        for PQ_index in range(0, 3):
            if PQ_index == 0 : 
                T = 20 #ensure we reach a canonical state first ...
            else : 
                T = deltaT
                #spins[3] = 1
                #spins[4] = 1
                #spins[5] = 1
                # eps = 10
                effJ1 *= 0.5
                effJ2 = 0.5*np.arccosh(np.exp(2*(effJ - effJ1)))
            for t in np.arange(0, T, dt) :
                spins = Hidden_Glauber_step_3spins(spins, effJ1, effJ2, dt, eps, PQ_index)
        visible_mag = np.sum(spins[0:3])
        hidden_mag = np.sum(spins[3:6])
        #total_mag = np.sum(spins)
        count_1[round((visible_mag+3)/2)] +=1
        count_2[round((hidden_mag+3)/2)] +=1
    count_1 /= size
    count_2 /= size
    return(mag, count_1, count_2)

def Hidden_PQ_protocol_sigma_check(J1,J2,dt,epsil,deltaT,size) :
    """
    deltaT is the time between 2 quenches
    """
    mag = np.arange(-3,4,2)
    count = np.zeros(4)
    for _ in tqdm(range(size)) :
        spins = Generate_random_config(6)
        eps = epsil
        effJ1 = J1
        effJ2 = J2 
        effJ = Effective_coupling(J1, J2) #Should be constant
        for PQ_index in range(0, 3):
            if PQ_index == 0 : 
                T = 10 #ensure we reach a canonical state first ...
            else : 
                T = deltaT
                #spins[3] = 1
                #spins[4] = 1
                #spins[5] = 1
                #eps *=2
                effJ1 *= 0.5
                effJ2 = 0.5*np.arccosh(np.exp(2*(effJ - effJ1)))
            for t in np.arange(0, T, dt) :
                spins = Hidden_Glauber_step_3spins(spins, effJ1, effJ2, dt, eps, PQ_index)
        visible_mag = np.sum(spins[3:6])
        #total_mag = np.sum(spins)
        count[round((visible_mag+3)/2)] +=1
    count /= np.sum(size)
    return(mag, count)


def Histogram_spins(J1,J2,dt,epsil,deltaT,size):
    mag = np.arange(-3,4,2)
    count = np.zeros(4)
    s1_history = np.zeros(round((10+2*deltaT)/dt))
    s2_history = np.zeros(round((10+2*deltaT)/dt))
    s3_history = np.zeros(round((10+2*deltaT)/dt))
    sigma1_history = np.zeros(round((10+2*deltaT)/dt))
    sigma2_history = np.zeros(round((10+2*deltaT)/dt))
    sigma3_history = np.zeros(round((10+2*deltaT)/dt))
    for _ in tqdm(range(size)) :
        spins = Generate_random_config(6)
        eps = epsil
        effJ1 = J1
        effJ2 = J2 
        effJ = Effective_coupling(J1, J2) #Should be constant
        for PQ_index in range(0, 3):
            if PQ_index == 0 : 
                T = 10 #ensure we reach a canonical state first ...
            else : 
                T = deltaT
                #spins[3] = 1
                #spins[4] = 1
                #spins[5] = 1
                #eps *=2
                #effJ1 *= 0.5
                #effJ2 = 0.5*np.arccosh(np.exp(2*(effJ - effJ1)))
            for i in range(round(T/dt)) :
                spins = Hidden_Glauber_step_3spins(spins, effJ1, effJ2, dt, eps, PQ_index)
                s1, s2, s3, sigma1, sigma2, sigma3 = spins
                if PQ_index == 0 :
                    s1_history[i] += s1
                    s2_history[i] += s2
                    s3_history[i] += s3
                    sigma1_history[i] += sigma1
                    sigma2_history[i] += sigma2
                    sigma3_history[i] += sigma3
                else :
                    s1_history[round(10/dt)+((PQ_index-1)*round(T/dt))+i] += s1
                    s2_history[round(10/dt)+((PQ_index-1)*round(T/dt))+i] += s2
                    s3_history[round(10/dt)+((PQ_index-1)*round(T/dt))+i] += s3
                    sigma1_history[round(10/dt)+((PQ_index-1)*round(T/dt))+i] += sigma1
                    sigma2_history[round(10/dt)+((PQ_index-1)*round(T/dt))+i] += sigma2
                    sigma3_history[round(10/dt)+((PQ_index-1)*round(T/dt))+i] += sigma3
                    
        #visible_mag = np.sum(spins[0:3])
        #total_mag = np.sum(spins)
        #count[round((visible_mag+3)/2)] +=1
    #count /= np.sum(size)
    s1_history/= size
    s2_history/= size
    s3_history/= size
    sigma1_history/= size
    sigma2_history/= size
    sigma3_history/= size
    return(mag, count, s1_history,s2_history,s3_history,sigma1_history,sigma2_history,sigma3_history)

def History_magnetisation_PQ(J1,J2,dt,epsil,deltaT,size):
    mag = np.arange(-3,4,2)
    Mcount = np.zeros((round((10+2*deltaT)/dt),4))
    Msigma_count = np.zeros((round((10+2*deltaT)/dt),4))
    Mtotal_count = np.zeros((round((10+2*deltaT)/dt),7))
    for _ in tqdm(range(size)) :
        spins = Generate_random_config(6)
        eps = epsil
        effJ1 = J1
        effJ2 = J2 
        effJ = Effective_coupling(J1, J2) #Should be constant
        for PQ_index in range(0, 3):
            if PQ_index == 0 : 
                T = 10 #ensure we reach a canonical state first ...
            else : 
                T = deltaT
                # HERE YOU CAN INPUT SOME PARAMETERS TO MODIFY THINGS
                #spins[3] = 1
                #spins[4] = 1
                #spins[5] = 1
                #eps *=2
                effJ1 *= 0.5
                effJ2 = 0.5*np.arccosh(np.exp(2*(effJ - effJ1)))
            for i in range(round(T/dt)) :
                spins = Hidden_Glauber_step_3spins(spins, effJ1, effJ2, dt, eps, PQ_index)
                visible_mag = np.sum(spins[0:3])
                hidden_mag = np.sum(spins[3:6])
                total_mag = visible_mag + hidden_mag
                Mcount[round(10/dt)+((PQ_index-1)*round(T/dt))+i][round((visible_mag+3)/2)] +=1
                Msigma_count[round(10/dt)+((PQ_index-1)*round(T/dt))+i][round((hidden_mag+3)/2)] +=1
                Mtotal_count[round(10/dt)+((PQ_index-1)*round(T/dt))+i][round((total_mag+6)/2)] +=1
    Mcount /= size
    Msigma_count /= size
    Mtotal_count /= size
        
    return(mag, Mcount, Msigma_count, Mtotal_count)

def Compute_Second_Moment(mag, proba):
    mean = np.dot(mag, proba)
    cor = (mag - mean)**2
    return np.dot(cor, proba)

def Classic_Glauber(spins,j,dt,epsilon):
    n = len(spins)
    for k in range(n):
        sk = spins[k]
        if random.random() < (dt/(2*epsilon))*(1 - sk*np.tanh(j * (np.sum(spins) - sk))) :
            spins[k] *= -1
    return(spins)

def History_canoncial_equivalent_3spins(j,T,dt,epsilon,size) :
    mag = np.arange(-3,4,2)
    effJ = Effective_coupling(j, j)
    Mcount = np.zeros((round(T/dt),4))
    Moments = np.zeros(round(T/dt))
    for _ in tqdm(range(size)) :
        spins = Generate_random_config(3)
        for i in range(round(T/dt)) :
            spins = Classic_Glauber(spins, effJ, dt, epsilon)
            M = np.sum(spins)
            Mcount[i][round((M+3)/2)] += 1
    Mcount /= size
    for i in range(round((T/dt))) :
        Moments[i] = Compute_Second_Moment(mag, Mcount[i])
    return(Moments)
            
    


#%% Results

stationnary_dist_T10_dt005_eps1dot5_size300000 = np.array([0.22775667, 0.27315   , 0.27278667, 0.22630667])
stationnary_dist_T100_dt01_eps1dot5_size100000 = np.array([0.22496, 0.27354, 0.27871, 0.22279]) #not symetric....
stationnary_dist_T100_dt01_eps10_size100000 = np.array([0.26986, 0.23129, 0.23088, 0.26797])

test = Static_Distribution(3, 1/3, 0.8, 100, 0.05, 1.5, 1000000)

#%% Plot the evolution of the second moment for the test
l = len(Mcount_test)
mag_tot = np.arange(-6,7,2)
visible_moments = np.zeros(l)
hidden_moments = np.zeros(l)
total_moments = np.zeros(l)

for i in range(l):
    visible_moments[i] = Compute_Second_Moment(mag, Mcount_test[i])
    hidden_moments[i] = Compute_Second_Moment(mag, Msigma_test[i])
    total_moments[i] = Compute_Second_Moment(mag_tot, Mtotal_test[i])
    
#%% PLotting module test

plt.plot(np.arange(0,12,dt), total_moments)
plt.vlines(10,np.min(total_moments),np.max(total_moments), colors='red', linestyles='dotted', label = 'First quench')
plt.vlines(11,np.min(total_moments),np.max(total_moments), colors='green', linestyles='dotted', label = 'Second quench')
plt.legend()

plt.grid()
plt.show()

#%% Second moment evolution for j modification

l = len(Mcount_PQ_jmodT20)
mag_tot = np.arange(-6,7,2)
visible_moments = np.zeros(l)
hidden_moments = np.zeros(l)
total_moments = np.zeros(l)

for i in range(l):
    visible_moments[i] = Compute_Second_Moment(mag, Mcount_PQ_jmodT20[i])
    hidden_moments[i] = Compute_Second_Moment(mag, Msigma_PQ_jmodT20[i])
    total_moments[i] = Compute_Second_Moment(mag_tot, Mtotal_PQ_jmodT20[i])
#%% Plotting module j mod

data = hidden_moments
label_data = 'Hidden ($\\sigma$) magnetisation distributions'

plt.plot(np.arange(0,50,dt), data, label = label_data)
#plt.hlines(data[round(10/dt -1)], 0, 50, 'black', 'dotted')
plt.vlines(10,np.min(data),np.max(data), colors='red', linestyles='-', label = 'First quench ($S_1$)')
plt.vlines(30,np.min(data),np.max(data), colors='green', linestyles='-', label = 'Second quench ($S_2$)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Second moment value')
plt.grid()
plt.title("Time evolution of the Second moment of the magnetisation \n starting from a random configuration $ \\Delta T = 20$ ")
#plt.savefig('Hidden_Dynamics_PQ_with_J1_decrease_T20.pdf')
plt.show()

#%%Plotting module comp with canonical dynamics

data = visible_moments
canon_moments_dynamicsT10 = History_canoncial_equivalent_3spins(j,10,0.05,1.5,100000)

#%%

label_data = 'Visible ($S$) magnetisation distributions'
t = np.arange(0,10,dt)
plt.plot(t, data[0:round(10/dt)],'.', label = label_data)
plt.plot(t, model(t,res_0), '--', label = f'exponential fit : $\\tau = {1/res_0[0]} $')
plt.plot(t, canon_moments_dynamicsT10,'.', label = 'Effective 3 spins system')
plt.plot(t, model(t,res_1), '--', label = f'exponential fit : $\\tau = {1/res_1[0]} $')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Second moment value')
plt.grid()
plt.title("Time evolution of the Second moment of the magnetisation \n starting from a random configuration $\\epsilon = 1.5 , dt = 0.05$")
plt.savefig('Glauber_convergence_speed_difference_33_and_effective3.pdf')
plt.show()


#%% Epsilon dependency

mag, can = Canonical_Distribution(n, j)
T=20
for tau in [0.1,0.2,0.3,0.4,0.5,1,2] : 
    mag, count = Static_Distribution(n,j,tau,T,0.1,1.5,50000)
    plt.plot(mag,count,'-o', label = f'tau = {tau}')
plt.plot(mag, can, 'r--', label = 'canon')
plt.title(f'Test for tau dependency on CH, for $T = {T}$, $\\epsilon = 1.5$ ')
plt.xlabel('Magnetisation value')
plt.ylabel('Stationnary probability')
plt.legend()
plt.savefig('Tau_variation_3spins.pdf')
plt.show()

#%% tau dependency

mag, can = Canonical_Distribution(n, j)

for dt in [0.2,0.1,0.05,0.02,0.01,0.005] : 
    mag, count = Static_Distribution(n,j,tau,2,dt,1.5,50000)
    plt.plot(mag,count,'-o', label = f'dt = {dt}')
plt.plot(mag, can, 'r--', label = 'canon')
plt.title('Test for dt dependency')
plt.legend()
plt.show()
#%%

plt.plot(mag, countjj, 'b-o', label = 'Visible magnetisation $j_1 = j_2 = 1/3$')
plt.plot(mag, PQ_hidden_eps_change, 'g-o', label = 'hidden mag, eps increse')
#plt.plot(mag, countj05, 'g-o', label = 'Visible magnetisation $j_1 = 1/3, j_2 = 1/6$')
#plt.plot(mag, countj1_23, '-o', color = 'purple', label = 'Visible magnetisation $j_1 = 1/3, j_2 = 2/3$')
#plt.plot(mag, can_eff, '--', color ='purple', label = 'Canon effective coupling')
#plt.plot(mag, countj1_0, '-o',color='orange', label = 'Visible magnetisation $j_1 = 0$')
#plt.plot(mag, can, 'r--', label = 'Visible magnetisation $j_1 = 1/3,  j_2 = 0$')
plt.xlabel('Magnetisation value')
plt.ylabel('Probability')
plt.title('Probability distribution for the visible magnetisation \n for the 3+3 system with $T=10$ and $dt=0.05$ and $\\epsilon = 1.5$')
plt.legend()
plt.grid()
#plt.savefig('proba33_diffj(noPQ)_T=10_size=1mil.pdf')
plt.show()


#%% Other plotting module
#plt.plot(mag, can_eff, '--', color ='purple', label = 'Canon effective coupling')

#plt.plot(mag, visible_can, '--o', label = 'Canonical $S$ magnetisation ')

#plt.plot(mag, PQ_visible_j_change_T1, '-o', label = '$S$ magnetisation after $j$ change, T=1')

#plt.plot(mag, PQ_visible_j_change_T10, '-o', label = '$S $ magnetisation after $j$ change, T=10')

#plt.plot(mag, PQ_visible_j_change_T50, '-o', label = '$S $ magnetisation after $j$ change, T=50')


plt.plot(mag, hidden_can, '--*', label = 'Canonical $\\sigma$ magnetisation')


plt.plot(mag, PQ_hidden_j_change_T1, '-*', label = '$\\sigma$ magnetisation after $j$ change, T=1')

plt.plot(mag, PQ_hidden_j_change_T10, '-*', label = '$\\sigma$ magnetisation after $j$ change, T=10')

plt.plot(mag, PQ_hidden_j_change_T50, '-*', label = '$\\sigma$ magnetisation after $j$ change, T=50')



# plt.plot(mag, PQ_sigma1, '-o', label = 'PQ with $ \\sigma_1 = 1$ after quench')
# plt.plot(mag, PQ_sigma2, '-o', label = 'PQ with $ \\sigma_2 = 1$ after quench')
# plt.plot(mag, PQ_sigma3, '-o', label = 'PQ with $ \\sigma_3 = 1$ after quench')

#plt.plot(mag, PQ_10, '-o', label = 'PQ with $ \\Delta T = 10$ and $J= K = 1/3$')

#plt.plot(mag, PQ_normaljj_1, '-o', label = 'PQ with changing $J$ and $K$, constant $\\tilde{J}$ \n with $ \\Delta T = 10$')


# plt.plot(mag, PQ_test_eps_1, '-o', label = 'PQ with eps increase to 10 with DT=10')
# plt.plot(mag, PQ_test_eps_2, '-o', label = 'PQ with eps increase to 10 with DT=2')
# plt.plot(mag, PQ_test_eps_j_inc_3, '-o', label = 'PQ with eps increase by 300% with DT=10')
#plt.plot(mag, countj1_23, '-o', color = 'purple', label = 'Visible magnetisation $j_1 = 1/3, j_2 = 2/3$')

#plt.plot(mag, countj1_0, '-o',color='orange', label = 'Visible magnetisation $j_1 = 0$')
#plt.plot(mag, can, 'r--', label = 'Visible magnetisation $j_1 = 1/3,  j_2 = 0$')
plt.xlabel('Magnetisation value')
plt.ylabel('Probability')
plt.title('Probability distribution for the visible magnetisation \n for the 3+3 system with $T=10$ and $dt=0.05$ and $\\epsilon = 1.5$')
plt.legend()
plt.grid()
#plt.savefig('proba33_PQ_sigma_up_T=10_size=50k.pdf')
plt.show()

#%% fitting module for Glazuber convergence

from scipy.optimize import least_squares

def model(x, t):
    return x[0] + x[1]*(1-np.exp(-t*x[2]))

def fun(x,t,y):
    return model(x,t) - y

def jac(x,t,y) :
    J = np.empty((len(t),len(x)))
    J[:,0] = 1
    J[:,1] = 1 - np.exp(-t * x[2])
    J[:,2] = x[1] * t * np.exp(-t*x[2])
    return J

t = np.arange(0, 10, 0.05)

y = test

x0 = np.array([3.2,3.2,1.0/3])

res = least_squares(fun, x0, jac= jac, bounds=(0,10), args=(t,y), verbose=1)

#%% More precise, regression only on eps

yep = visible_moments[0:round(10/dt)]

def model(x, t):
    return yep[-1] - (yep[-1]-yep[0])*(np.exp(-t*x))

def fun(x,t,y): # x has to be first argument
    return model(x,t) - y

def jac(x,t,y):
    J = np.empty((len(t),1))
    J[:,0] = (yep[-1]-yep[0])*t*np.exp(-t*x)
    return J

t = np.arange(0, 10, 0.05)

y = yep

x0 = np.array([1.0/1.5]) # according to doc : never a scalar

res = least_squares(fun, x0, jac= jac, bounds=(0,10), args=(t,y), verbose=1)
#%%
plt.plot(t,y,'b.')
plt.plot(t, model(res.x, t), 'r-')
plt.plot()

#%%

def model2(x,l_esp):
    return x[0]*l_esp + x[1]

def f(x,l_eps,y):
    return model2(x,l_eps) - y

def jac2(x,l_eps,y):
    J = np.empty((len(l_eps), 2))
    J[:,0] = l_eps
    J[:,1] = 1
    return J
    
l_eps = np.array([1.2, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9])

y = l_tau

x0 = np.array([0,0.8])

result = least_squares(f, x0, jac= jac2, bounds=(-1,5), args=(l_eps,y), verbose=1)

print(result.x)

# this gives a slope and origin of value : [ 0.83326555 -0.18596141]
#%% Ouin ouin j'ai pas les noms de variable

plt.plot(mag, visible_canon, ls = '--', marker = '.', label = 'Visible ($S$) canonical distribution\nwith $J=K=1/3$ ')
plt.plot(mag,visible_PQ_DeltaT1,ls = '-', marker = '1', label = 'Visible part after PQ $\\Delta T = 1 \\varepsilon$\nwith $J=K=1/3$', linewidth = '1')
plt.plot(mag, visible_PQ, ls = '-', marker = '2', label = 'Visible part after PQ $\\Delta T = 15 \\varepsilon$\nwith $J=K=1/3$', linewidth = '1')
plt.plot(mag, visible_PQ_Jtilde, ls = ':', marker = 'o', label = 'Visible part after PQ with $\\Delta T = 15 \\varepsilon$\nand changing $J$ and $K$, constant $\\tilde{J}$')
plt.legend(fontsize='9')
plt.xlabel('Magnetization')
plt.ylabel('Probability distribution')
#plt.savefig('33_spins_steady_states.pdf')
plt.show()