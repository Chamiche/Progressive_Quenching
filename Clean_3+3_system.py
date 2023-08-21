#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:58:23 2023

@author: CMoslonka
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
mem_length = 8
epsilon = 1.5
eps = epsilon
duration = 15 * epsilon

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
#%% Non-Markovian detailed balance simulation, with PQ
'''
Everything here is for 3 + 3 spins config only

'''

def Hidden_flip_proba_3spins(spins, flip_index, J1, J2, dt, epsilon) :
    
    # This is the probability of spin flip, depending on which spin 
    # we consider. (Glauber algorithm)
    
    sk = spins[flip_index]
    s1, s2, s3, s4, s5, s6 = spins
    if flip_index == 0:
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
    p = (1 - (sk * np.tanh(Ekprime))) * dt

    return(p/(2*epsilon))

def Hidden_Glauber_step_3spins(spins, J1, J2, dt, epsilon, PQ_index=0) :
    # A single step of glauber, to check flip every spin once
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

def Hidden_PQ_protocol(J1,J2,dt,epsilon,deltaT,size) :
    """
    CHNAGE THIS PROGRAM IF YOU WANT TO CHANGE THE PQ PROTOCOL.
    deltaT is the time between 2 quenches
    """
    mag = np.arange(-3,4,2)
    count_1 = np.zeros(4)
    count_2 = np.zeros(4)
    #count_3 = np.zeros(4)
    for _ in tqdm(range(size)) :
        # We start from random spins. We could
        # also start from canonical spins 
        spins = Generate_random_config(6)
        eps = epsilon
        effJ1 = J1
        effJ2 = J2
        effJ = Effective_coupling(J1, J2) #Should be constant
        for PQ_index in range(0, 3):
            if PQ_index == 0 : 
                T = 10*epsilon #ensure we reach a canonical state first ...
            else : 
                T = deltaT
                # Uncomment which protocol you want to be applied
                # after EACH quench.
                
                # Protocol where we set the hidden spins to certain values
                
                #spins[3] = 1
                #spins[4] = 1
                #spins[5] = 1
                
                # Protocol where we set the kinetic constant to certain values
                
                #eps = 10
                
                # Protocol where we modifiy both interaction constants 
                # while keeping the effective coupling constant
                
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
    Mcount = np.zeros((round((10*epsil + 2*deltaT)/dt),4))
    Msigma_count = np.zeros((round((10*epsil + 2*deltaT)/dt),4))
    Mtotal_count = np.zeros((round((10*epsil + 2*deltaT)/dt),7))
    for _ in tqdm(range(size)) :
        spins = Generate_random_config(6)
        eps = epsil
        effJ1 = J1
        effJ2 = J2 
        effJ = Effective_coupling(J1, J2) #Should be constant
        for PQ_index in range(0, 3):
            if PQ_index == 0 : 
                T = 10*epsil #ensure we reach a canonical state first ...
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
                Mcount[round(15/dt)+((PQ_index-1)*round(T/dt))+i][round((visible_mag+3)/2)] +=1 #Change 15 if necessary !!!
                Msigma_count[round(15/dt)+((PQ_index-1)*round(T/dt))+i][round((hidden_mag+3)/2)] +=1
                Mtotal_count[round(15/dt)+((PQ_index-1)*round(T/dt))+i][round((total_mag+6)/2)] +=1
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
            
#%%
#mag, Mcount_PQ_jmodT20, Msigma_PQ_jmodT20, Mtotal_PQ_jmodT20 = History_magnetisation_PQ(1/3,1/3,0.1,1.5,15*epsilon,5*10**5)

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

data = visible_moments
label_data = 'Visible ($S$) magnetisation distributions'

plt.plot(np.linspace(0,30,451), np.log(np.abs(data[149:] - np.max(data[149:]))), label = label_data)
#plt.hlines(data[round(10/dt -1)], 0, 50, 'black', 'dotted')
#plt.axvline(0, color='red', linestyle=':', label = 'First quench ($S_1$)', linewidth=1.5)
#plt.axvline(15, color='green', ls=':', label = 'Second quench ($S_2$)', linewidth=1.5)
#plt.ylim(5.9,6.8)
#plt.axhline(data[149], ls = '--', color = 'k', linewidth=1, label = '$\\mathbb{E}[M^2]$ value before PQ')
plt.legend(fontsize = 11)
plt.xlabel('Time (in $\\varepsilon$ units)')
plt.ylabel('Second moment $\\mathbb{E}[M^2]$')
#plt.grid()
#plt.title("Time evolution of the Second moment of the magnetisation \n starting from a random configuration $ \\Delta T = 20$ ")
#plt.savefig('Visible_Dynamics_PQ_with_J1_decrease_T15.pdf')
plt.show()

#%% 