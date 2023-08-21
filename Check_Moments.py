#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:55:40 2022

@author: CMoslonka
"""
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


def Generate_static_configuration_v4(n,j,tau,duration,dt,epsilon):
    
    r = random.randint(0,l-1)
    
    actual_spin_list = possible_history[r][0]
    memory = possible_history[r][1]
    
    actual_spin_list, memory = Continuous_Glauber_with_Delay(actual_spin_list, 
        memory, 0 , j, epsilon, duration, dt)
    
    return(np.sum(actual_spin_list))




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


def Generate_static_distribution(n, j, tau, duration, simulsize, dt, epsilon, init_memory_type='canon'):
    M_count = np.zeros(n+1)
    for i in tqdm(range(simulsize)):
        M = Generate_static_configuration(
            n, j, tau, duration, dt, epsilon, init_memory_type)
        M_count[int(M+n)//2] += 1

    Canon = Canonical_Distribution(n, j)
    plt.plot(Canon[0], M_count/simulsize, '.')
    plt.plot(Canon[0], Canon[1], '--')
    plt.show()
    asym = 0
    for i in range(n+1):
        asym += np.abs(M_count[i]-M_count[-i-1])
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


cpu = cpu_count() - 2


f = open('lebonfichier2.txt', 'r')
raw = f.read()
raw = raw.replace('. ', ',')

possible_history = ast.literal_eval(raw)

l = len(possible_history)

del(raw)
f.close()

def PQ_on_Continuous_Glauber_with_Delay_from_History(T) :
    r = random.randint(0,l-1)
    
    actual_spin_list = possible_history[r][0]
    memory = possible_history[r][1]

    for PQ_index in range(1,16):
        actual_spin_list, memory=Continuous_Glauber_with_Delay(actual_spin_list, memory, PQ_index, 1.25/16, 1.5, T, dtt)
        
    return(actual_spin_list)

def Dist_PQ_from_History(T,size) :
    stt = time.perf_counter()    
    M_count=np.zeros(17)    
    for i in range(size):
        M=np.sum(PQ_on_Continuous_Glauber_with_Delay_from_History(T))
        M_count[int(M+16)//2]+=1
        if i == 19 :
            print('Estimated remaining time : ', (time.perf_counter() - stt)*size/20, 'seconds')
        
    return(M_count)


def Compute_Second_Moment(mag, proba):
    mean = np.dot(mag, proba)
    cor = (mag - mean)**2
    return np.dot(cor, proba)


def Compute_Second_Moment_v2(mag, proba):
    mean = np.dot(mag, proba)
    magsq = mag**2
    mom = np.dot(magsq, proba)
    return(mom - mean**2)



#%%
dtt = 0.1
Size = 60000
n, j, tau , dt, epsilon = (16, 1.25/16, 0.8, dtt, 1.5)
mag, can = Canonical_Distribution(n, j)
list_T = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

list_of_moments = np.zeros(len(list_T))
index = 0
for T in list_T :
    
    #Le plan c'est : faire les PQ avec tous les T différents
    #Imprimer les courbes
    #Et faire la courbe avec les moments
    #Et la convergence vers la valeur hypothétique qu'est la courbe canonique.
    s = Dist_PQ_from_History(T, Size)
    
    proba = s/np.sum(s)
    
    file = open(f'Result_PQ_T={T}_dt={dtt}_Size={Size}.txt', 'w')
    file.write(str(proba))
    file.close()
    
    list_of_moments[index] = Compute_Second_Moment(mag, proba)
    
    plt.plot(mag, proba, 'b.', label = f'PQ distribution T = {T}')
    plt.plot(mag, static, 'r*', label = 'Stationnary distrubution')
    plt.plot(mag, can, 'r-', label = 'Canonical distribution')
    plt.title(f"PQ distribution with $dt = {dtt} $ over {Size} samples")
    plt.legend()
    plt.savefig(f'PQ_from_history_T={T}_dt={dtt}_Size={Size}.pdf')
    plt.show()
    
    index += 1

can_mom = Compute_Second_Moment(mag, can)   
stat_mom = Compute_Second_Moment(mag, static) 
can_line = [can_mom]*len(list_T)    
stat_line = [stat_mom]*len(list_T)
plt.plot(list_T, list_of_moments, 'bo-')
plt.plot(list_T, can_line, 'r--', label = 'Canonical')
plt.plot(list_T, stat_line, 'g--', label = 'Stationnary')
plt.xlabel('Time $T$ between quenches')
plt.ylabel('Second moment of computed distributions')
plt.title('Second moment of the PQ distributions with T')
plt.legend()
plt.savefig(f'SecondMoments_size={Size}.pdf')
plt.show()
    
    
#%% Making a plot from data (txt files)
Starting = [0.031944, 0.071581, 0.086571, 0.081568, 0.067807, 0.05562 ,
       0.046128, 0.040751, 0.039008, 0.040342, 0.044832, 0.054958,
       0.069087, 0.082034, 0.085641, 0.07036 , 0.031768]


proba01 = [0.0328425, 0.07063  , 0.0860125, 0.0808675, 0.06823  , 0.05557  ,
       0.045765 , 0.0407375, 0.039115 , 0.0408575, 0.04674  , 0.055825 ,
       0.06878  , 0.0813375, 0.08497  , 0.069415 , 0.032305 ]

mom01 = Compute_Second_Moment(mag, proba01)

proba02 = [0.03796 , 0.073095, 0.08558 , 0.07883 , 0.066315, 0.05523 ,
       0.045855, 0.040475, 0.0388  , 0.04129 , 0.046125, 0.05458 ,
       0.06643 , 0.07732 , 0.08307 , 0.0724  , 0.036645]

mom02 = Compute_Second_Moment(mag, proba02)


proba05 = [0.04514, 0.07698, 0.08469, 0.07529, 0.06358, 0.05381, 0.04496,
       0.04046, 0.03883, 0.04017, 0.04539, 0.05055, 0.06361, 0.07504,
       0.08339, 0.07466, 0.04345]

mom05 = Compute_Second_Moment(mag, proba05)

proba1 = [0.047295, 0.0796225, 0.08401, 0.07421, 0.061375, 0.0516975, 0.04421, 0.03995,
          0.0386225, 0.0400125, 0.0442125, 0.0515425, 0.0618475, 0.0736625, 0.0831125, 0.07803, 0.0465875]

mom1 = Compute_Second_Moment(mag, proba1)

proba2 = [0.0519125, 0.083625, 0.0855, 0.0736625, 0.0591875, 0.049125, 0.0427125, 0.036725,
          0.03625, 0.038225, 0.040575, 0.048325, 0.0610375, 0.073725, 0.0849375, 0.0829625, 0.0515125]

mom2 = Compute_Second_Moment(mag, proba2)

proba3 = [0.0545125, 0.0861125, 0.0863875, 0.07345, 0.0583625, 0.04755, 0.0424625, 0.035675,
          0.0346375, 0.0368875, 0.0407625, 0.0487125, 0.0575, 0.0723, 0.0851125, 0.0846625, 0.0549125]

mom3 = Compute_Second_Moment(mag, proba3)

proba5 = [0.058075, 0.0878625, 0.08855, 0.0738, 0.057175, 0.047175, 0.0385, 0.03455,
          0.03205, 0.0344375, 0.039225, 0.0463625, 0.0579625, 0.074175, 0.0866, 0.086175, 0.057325]

mom5 = Compute_Second_Moment(mag, proba5)

proba8 = [0.0594875, 0.0922, 0.0893875, 0.0714375, 0.0569875, 0.04515, 0.037275, 0.0334,
          0.031875, 0.033625, 0.0376875, 0.0443875, 0.057425, 0.0727875, 0.0881125, 0.0905875, 0.0581875]

mom8 = Compute_Second_Moment(mag, proba8)

proba10 = [0.06025, 0.0907125, 0.0885, 0.069575, 0.056375, 0.0443, 0.0368875, 0.0336125,
           0.0313, 0.0341375, 0.038, 0.0454125, 0.05855, 0.0731, 0.0879625, 0.09105, 0.060275]

mom10 = Compute_Second_Moment(mag, proba10)

proba15 = [0.0608875, 0.0909875, 0.0893375, 0.0739625, 0.05595, 0.045, 0.0363125, 0.0311875,
           0.0310375, 0.03305, 0.03615, 0.0443375, 0.0565125, 0.072475, 0.0895375, 0.0923, 0.060975]

mom15 = Compute_Second_Moment(mag, proba15)
## From here its bollowwww
proba20 = [0.06084615, 0.09043269, 0.08890385, 0.07301923, 0.05677885, 0.04475, 0.03771154, 0.03266346,
           0.03193269, 0.03267308, 0.03757692, 0.04501923, 0.05471154, 0.07220192, 0.08853846, 0.09226923, 0.05997115]

mom20 = Compute_Second_Moment(mag, proba20)

proba50 = [0.06298077, 0.09286538, 0.08970192, 0.07208654, 0.05729808, 0.04463462, 0.03595192, 0.03149038,
           0.03066346, 0.03170192, 0.03547115, 0.044125, 0.05508654, 0.07206731, 0.08835577, 0.09361538, 0.06190385]

mom50 = Compute_Second_Moment(mag, proba50)

proba100 = [0.06321154, 0.09446154, 0.09123077, 0.073125, 0.05444231, 0.04343269, 0.03561538, 0.03165385,
            0.03039423, 0.03139423, 0.03675962, 0.04319231, 0.055375, 0.07259615, 0.08779808, 0.092125, 0.06319231]

mom100 = Compute_Second_Moment(mag, proba100)

list_of_moments = [mom02, mom05, mom1, mom2, mom3, mom5, mom8, mom10, mom15, mom20, mom50, mom100]

list_of_moments2 = [mom02, mom05, mom1, mom2, mom3, mom5, mom8, mom10, mom15]

list_T = [0.2, 0.5, 1, 2, 3, 5, 8, 10, 15]

L = len(list_of_moments2)

can_mom = Compute_Second_Moment(mag, can)   
stat_mom = Compute_Second_Moment(mag, static) 
can_line = [can_mom]*L   
stat_line = [stat_mom]*L
plt.plot(list_T, list_of_moments2, 'ko')
plt.plot(list_T, can_line, 'b--', label = 'Canonical')
plt.plot(list_T, stat_line, 'r--', label = 'Stationnary')
plt.xlabel('Time $\Delta T$ between quenches')
plt.ylabel('Second moment of computed distributions')
plt.title('Second moment of the PQ distributions from history, 80k points each')
plt.legend()
plt.savefig('SecondMoments_size=80k.pdf')
plt.show()

 #%% 

plt.plot(mag, proba1,'.' ,label = '$\Delta T = 1$')
plt.plot(mag, proba2,'o' ,label = '$\Delta T = 2$')
plt.plot(mag, proba3,'*' ,label = '$\Delta T = 3$')
plt.plot(mag, proba15,'v' ,label = '$\Delta T=15$')


plt.plot(mag, can,'b-', label = 'Canonical')
plt.plot(mag, static,'r--', label = "Stationnary")
plt.legend()
plt.xlabel('Magnetisation values')
plt.ylabel('Probability value')
plt.title('Evolution of the PQ distribution with $\Delta T$')
plt.grid()
plt.savefig('Distribution_evol_DT.pdf')
plt.show()

#%%


plt.plot(mag, can,'bo-.', label = 'Canonical')
plt.plot(mag, static,'ro-.', label = "Stationnary")
plt.legend()
plt.xlabel('Magnetisation values')
plt.ylabel('Probability value')
plt.grid()
plt.title('Canonical distribution and NESS for $N = 16, \\tau = 0.8, dt = 0.1, \epsilon = 1.5$')
plt.savefig('Comparison_canon_statio.pdf')
plt.show()
#%%

