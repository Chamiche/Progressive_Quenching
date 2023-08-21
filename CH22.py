#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:41:26 2022

@author: CMoslonka

Discrete-time delayed Glauber Dynamics
Specific for N=2 and one time delay.

Reference is in main-III.tex
"""
#%% Importing modules 

import numpy as np
import random 
import matplotlib.pyplot as plt

from tqdm import tqdm

#%% Parameters

j=1/8 # Value of the interaction constant, we set Beta == 1
eps = 2 # Value of the kinetic parameter || must be >1
T = 10000000 # duration of the simulation 
eta=np.tanh(j)

#%%Initialisation and simulation

present_spins=[random.choice([-1,1]),random.choice([-1,1])]
past_spins=[random.choice([-1,1]),random.choice([-1,1])]

spin1_history=np.zeros(T+2)
spin2_history=np.zeros(T+2) #will allow us to write the history
spin1_history[0]=past_spins[0]
spin1_history[1]=present_spins[0]
spin2_history[0]=past_spins[1]
spin2_history[1]=present_spins[1]

for i in tqdm(range(T)):
    s0=present_spins[0]
    s1=present_spins[1]
    t0=past_spins[0]
    t1=past_spins[1]
    past_spins[0]=s0 #We change now because we stored the values elsewehere
    past_spins[1]=s1
    if (random.random()<(1/(2*eps))*(1-s0*t1*eta)) : 
        present_spins[0]=-s0
    if (random.random()<(1/(2*eps))*(1-s1*t0*eta)):
        present_spins[1]=-s1
    
    spin1_history[i+2]=present_spins[0]
    spin2_history[i+2]=present_spins[1]

occ11=0
occ1m=0
occm1=0
occmm=0
for i in range(T+2):
    if spin1_history[i]==1 :
        if spin2_history[i]==1:
            occ11+=1
        else :
            occ1m+=1
    else :
        if spin2_history[i]==1:
            occm1+=1
        else :
            occmm+=1
print(''' 
      ***
 Probability results for j = %1.2f AND epsilon = %1.2f
 Made over %i counts
      ***
      '''%(j,eps,T))
print('P(++) = ',occ11/(T+2))
print('P(--) = ', occmm/(T+2))
print( 'P(+-) = ',occ1m/(T+2))
print('P(-+) = ',occm1/(T+2))

#%% Data treatment

n_1=np.count_nonzero(spin1_history==1) #number of +1 spins
n_2=np.count_nonzero(spin2_history==1)

print(n_1,T+2-n_1,n_2)

#%% Computing P(++,+-,-+ and --)
