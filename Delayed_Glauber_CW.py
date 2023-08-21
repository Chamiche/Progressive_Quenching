#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:19:32 2021

@author: CMoslonka

This code is made to simulate delayed Glauber dynamics.
Our source is the article 
"Collective excitations and retarded interactions"
By M.Y. Choi and B.A. Huberman 

My choice here is a Curie-Weiss model

"""
#%% Importing modules 

import numpy as np
import random 
import matplotlib.pyplot as plt
import math

from tqdm import tqdm #for completion bars

#%% Parameters

n=10 # Number of spins 
j=0.2 # Value of the interaction constant, we set Beta == 1
eps = 2 # Value of the relaxation time of a single spin
dt = 0.05 # Value of the simulation time step #dt=1 set for the discrete time 
#simulation
tau = 1 # Value of the delay IN dt UNITS
#You have to set tau = dt for NO DELAY SIMULATIONS
T = 10000 # duration of the simulation 

#%% Simulation 
"""
Our plan here is to compute at each time step the probability of each 
spins to switch to its inverse value, which is given by the transition 
rate.
We need a "present config" a past config which must be equal to the present
configuration at time t-tau.
Between t=0 and t=tau, we make something with a random past, then
we take those configurations and hopefully it will converge.


"""

#%% Modules and functions

def Generate_random_config(n):
    l=[]
    for i in range(n):
        l.append(random.choice([1,-1]))
    return(l)


def Transition_proba(i,present_sc, past_sc,j,eps):
    """
    
    Parameters
    ----------
    i : TYPE : Int
        Spin number.
    past_sc : list
        Past configuration 

    Returns the transition probability between t and t+dt
    -------

    """
    E=(j)*(np.sum(past_sc)-past_sc[i]) #Energy E_i prime
    return( (1/(2*eps)) * (1 - (present_sc[i]*np.tanh(E)) ) )




# #%% First try with writing down (not optimal at all)
# present_sc=[] # Present spin configuration
# past_sc=[] # Past spin configuration
# for i in range(n):
#     present_sc.append(random.choice([-1,1]))
#     past_sc.append(random.choice([-1,1]))
#     # The first ones are random


# history_config=[]
# history_config.append(present_sc)
# f = open('History j=%1.3f tau=%1.3f test.txt' %(j,tau),'w')
# f.write("[")
    
# for t in np.arange(0,tau,dt) :
#     past_sc=Generate_random_config(n)
#     for i in range(n):
#         p=random.random()
#         if (p<Transition_proba(i, present_sc, past_sc)*dt) :
#             present_sc[i]*=-1 # Inversion
#     history_config.append(present_sc)
#     f.write("[")
#     f.write(str(t))
#     f.write(",")
#     f.write(str(present_sc))
#     f.write("],")
#     # f.write("\n") #line break

# count=0 # Introducing a counter 

# for t in np.arange(tau,T,dt) :
    
#     past_sc=history_config[count%tau] #important bc we only need to retain 
#     # until tau 
#     for i in range(n):
#         p=random.random()
#         if (p<Transition_proba(i, present_sc, past_sc)*dt) :
#             present_sc[i]*=-1 # Inversion
#     history_config[count%tau]=present_sc
#     count+=1 #Update the counter
#     f.write("[")
#     f.write(str(t))
#     f.write(",")
#     f.write(str(present_sc))
#     if (t==T) : 
#         f.write("]")
#     else :
#         f.write('],')
#     # f.write("\n") #line break
    
# f.write(']')
    

# #%% Read the whole story.

# f = open('History j=%1.3f tau=%1.3f test.txt' %(j,tau),'r')
# data_string=f.read()
# #We can probably use np.loadtext instead *
# data = ast.literal_eval(data_string)

# #%%
# N=len(data)
# M=np.zeros(N)
# for i in range(N):
#     M[i]=np.sum(data[i][1])
# bins=[i for i in range(-(n+1),(n+2),2)]    
# plt.plot(M)
# plt.title('n= %i, j=%1.3f, N= %i' %(n,j,N))
# plt.show()
# plt.hist(M,bins,density=True)
# plt.title('n= %i, j=%1.3f, N= %i' %(n,j,N))
# plt.show()
#%% Canonical value



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
        logP=(-(j/2)*((Mag**2))) + math.log(prout)
        Pth[i]=np.exp(logP)
    Pth/=np.sum(Pth)
    
    return(list_M,Pth)


#%% Setting things up for large scale computing

"""
The idea is to make things bit by bit and store everything.
For exemple : cut the process in chunks of size 50000, store the last
memory, compute everything, count and store the data, delete those and 
continue with the memory first stored. 

We can then change the size N (multiply by 2) with the same parameters and
look at things. Then, for a given size, change the size of the memory or 
the cut-off epsilon. 

We will not write things in a file, 
it takes to long to read and will probably overload the RAM.

"""
def Run_Simulation(n,T,dt,j,tau,eps) :

    present_sc=Generate_random_config(n) # Present spin configuration
    
    M_count=[np.arange(-n,n+1,2),np.zeros(n+1)] # setting the counter for hist
    
    history_config=[] #the list that contains all of the history : size tau*dt 
    history_config.append(present_sc) # The history that will be used after 
    # generating the first random pasts

    
    for t in np.arange(dt,tau,dt) : #(size : dt*tau -1 )
        
        past_sc=Generate_random_config(n) #random past 
        for i in range(n): #We check for all the spins
            if(random.random()\
            <Transition_proba(i, present_sc, past_sc,j,eps)*dt) :
                present_sc[i]*=-1 # Inversion
        history_config.append(present_sc)
        
    
    count=0 # Introducing a counter 
    
    
    for t in tqdm(np.arange(tau,T,dt)) :
        
        past_sc=history_config[count%tau] #important bc we only need to retain 
        # until tau 
        for i in range(n):
            if (random.random() \
            <Transition_proba(i, present_sc, past_sc,j,eps)*dt):
                present_sc[i]*=-1 # Inversion
        M_count[1][(np.sum(present_sc)+n)//2]+=1 #Update the count for magn
        history_config[count%tau]=present_sc
        count+=1 #Update the counter
    
    print(M_count[1]/(np.sum(M_count[1])))

    # Plotting things properly
    Pth = Canonical_Distribution(n,j)
    plt.plot(Pth[0],Pth[1],'bo--',label='Canonical Distribution')
    plt.plot(Pth[0],M_count[1]/(np.sum(M_count[1])),'ro',label='Empirical distribution' )
    plt.xlabel('Magnetisation values')
    plt.ylabel('Probability')
    plt.legend()
    plt.title("Simulation results for n= %i, j=%1.2f, T= %i, dt=%1.2f , $ \\epsilon $ = %1.1f , $\\tau = %1.1f $ "  %(n,j,T,dt,eps,tau))
    plt.show()
    
    #return(np.max(np.abs(Pth[1]-(M_count[1]/(np.sum(M_count[1]))))))
    

    
#%% Special Simulation for the n=2 case (debug)

def Run_Simulation2(n,T,dt,j,tau,eps) :

    present_sc=Generate_random_config(n) # Present spin configuration
    
    M_count=[np.arange(-n,n+1,2),np.zeros(n+1)] # setting the counter for hist
    
    history_config=[] #the list that contains all of the history : size tau*dt 
    history_config.append(present_sc) # The history that will be used after 
    # generating the first random pasts

    
    for t in np.arange(dt,tau,dt) : #(size : dt*tau -1 )
        
        past_sc=Generate_random_config(n) #random past 
        for i in range(n): #We check for all the spins
            if(random.random()<Transition_proba(i, present_sc, past_sc,j,eps)*dt) :
                present_sc[i]*=-1 # Inversion
        history_config.append(present_sc)
        
    
    count=0 # Introducing a counter 
    
    occ11=0
    occ1m=0
    occm1=0
    occmm=0
    
    
    for t in np.arange(tau,T,dt) :
        
        past_sc=history_config[count%tau] #important bc we only need to retain 
        # until tau 
        for i in range(n):
            if(random.random()<Transition_proba(i, present_sc, past_sc,j,eps)*dt):
                present_sc[i]*=-1 # Inversion
        M_count[1][(np.sum(present_sc)+n)//2]+=1 #Update the count for magn
        
        # Little counting module
        
        if present_sc[0]==1 :
            if present_sc[1]==1:
                occ11+=1
            else :
                occ1m+=1
        else :
            if present_sc[1]==1:
                occm1+=1
            else :
                occmm+=1
                    
        history_config[count%tau]=present_sc
        count+=1 #Update the counter
    
    print(M_count[1]/(np.sum(M_count[1])))
    
    print('P(++) = ',occ11/count)
    print('P(--) = ', occmm/count)
    print( 'P(+-) = ',occ1m/count)
    print('P(-+) = ',occm1/count)

    # Plotting things properly
    Pth = Canonical_Distribution(n,j)
    plt.plot(Pth[0],Pth[1],'bo--',label='Canonical Distribution')
    plt.plot(Pth[0],M_count[1]/(np.sum(M_count[1])),'ro',label='Empirical distribution' )
    plt.xlabel('Magnetisation values')
    plt.ylabel('Probability')
    plt.legend()
    plt.title("Simulation results for n= %i, j=%1.2f, T= %i, dt=%1.2f , $ \\epsilon $ = %1.1f , $\\tau = %1.1f $ "  %(n,j,T,dt,eps,tau))
    plt.show()



#%% Progressive Quenching on Glauber dynamics 


"""
The plan is to create a list of updatable spins, and to remove progressively 
the quenched ones after waiting a sufficient timen (so that we let the system
thermalize according to Glauber dynamics).

Since the spins are all equivalent, 
we can just create an integer "k" that starts 
at 0, then 1, 2 ... up to n-1 that will act on the "for i in range(k,n)"


"""

def Crap_PQ_Simulation(n,T,dt,j,eps) : #Only for NO-DELAY

    present_sc=Generate_random_config(n) # Present spin configuration
    
    PQ_index=0
    
    M_count=np.zeros(n+1) # setting the counter for hist
    
    while(PQ_index!=n):
        for t in np.arange(0,T,dt):
            for i in range(PQ_index,n):
                if(random.random()<Transition_proba(i, present_sc, present_sc,j,eps)*dt):
                    present_sc[i]*=-1 # Inversion
            M_count[(np.sum(present_sc)+n)//2]+=1 #Update the count for magn
        PQ_index+=1 #Important for programm completion
        


    Pth = Canonical_Distribution(n,j) 
    plt.plot(Pth[0],Pth[1],'bo--',label='Canonical Distribution')
    plt.plot(Pth[0],M_count/(np.sum(M_count)),'ro',label='Empirical distribution' )
    plt.xlabel('Magnetisation values')
    plt.ylabel('Probability')
    plt.legend()
    plt.title("PQ Simulation results for n= %i, j=%1.2f, T= %i, dt=%1.2f , $ \\epsilon $ = %1.1f "  %(n,j,T,dt,eps))
    plt.show()

#%%

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
    
def PQ_Simulation(n,j,T,dt,num_config,eps):
    """
    

    Parameters
    ----------
    n : INT
        Number of SPINS.
    j : Float
        Value of the interaction constant.
    T : Float
        Simulation duration (units of dt).
    dt : Float < 1
        Time step.
    num_config : INT
        Number of configurations generated.
        Equivalent to the precision, assuming T*dt is large enough
    eps : Float >1
        Kinetic parameter of Glauber.

    Returns a normalized count of the magnetizations obtained
    and a comparison with the canonical probability
    -------

    """
    
    M_count=np.zeros(n+1) #Stock of M for generated conf, n+1 possible values
    for p in tqdm(range(num_config)) :
        #The first configuration needs to be from canonical distribution
        present_sc=Generate_canoncial_config(n, j)
        PQ_index=0
        while(PQ_index!=n):
            for t in np.arange(0,T,dt):
                for i in range(PQ_index,n):
                    if(random.random()<Transition_proba(i, present_sc, present_sc,j,eps)*dt):
                        present_sc[i]*=-1 # Inversion
            PQ_index+=1 #Important for programm completion
        M_count[int((np.sum(present_sc)+n)//2)]+=1 #Update the count for magn after everything is done
    
    print(M_count/(np.sum(M_count)))
    
    Pth = Canonical_Distribution(n,j) 
    plt.plot(Pth[0],Pth[1],'bo--',label='Canonical Distribution')
    plt.plot(Pth[0],M_count/(np.sum(M_count)),'ro',label='Empirical distribution' )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)

    textstr = '\n'.join((
        r'%i configs' %num_config,
        ))

    # place a text box in upper left in axes coords
    plt.text(np.min(Pth[0]),np.max(Pth[1]), textstr, fontsize=8, bbox=props)

    
    plt.xlabel('Magnetisation values')
    plt.ylabel('Probability')
    plt.legend()
    plt.title("PQ Simulation results for n= %i, j=%1.2f, T= %i, dt=%1.2f , $ \\epsilon $ = %1.1f "  %(n,j,T,dt,eps))
    plt.show()