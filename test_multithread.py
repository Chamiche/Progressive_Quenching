#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:18:19 2022

@author: CMoslonka

Implementation of multitheading for Gmauber simulations.
"""
import numpy as np
import random
import time
import sys
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from threading import Thread, RLock

#nope = RLock()

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

class Afficheur(Thread) :
    
    def __init__(self, n, j):
        Thread.__init__(self)
        self.n = n
        self.j = j
        
    def run(self):
        M = 0
        for i in range(10000):
            M += np.sum(Generate_canoncial_config(self.n, self.j))
        with nope : 
            return(M)
        
        
from multiprocessing import Process

def f(n,j):
    M=0
    for i in range(10000):
        M+= np.sum(Generate_canoncial_config(n, j))
    return(M)

if __name__ == '__main__' :
    p=Process(target = f, args=(16,1.25/16))
    p.start()
    p.join()   
            
#%%
thread1 = Afficheur(16,1.25/16)
thread2 = Afficheur(16,1.25/16)
thread3 = Afficheur(16,1.25/16)

thread1.start()
thread2.start()

thread1.join()
thread2.join()