#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:05:19 2021

Analysis of meq, checking if bimodality arises from stuff

@author: CMoslonka

Note : to get interactive figures, use %matplotlib Qt5, otherwise the default parameter is inline plotting

"""
#%% Loading relevant modules
import numpy as np
import random
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

plt.rcParams['animation.ffmpeg_path'] = '/Users/CMoslonka/ffmpeg'



#%% Parameters 

L=8
N0=2**L #System size

T0=2**7

N_iter=10**4

#%% Loading the data || NEEDS TO BE FOR THE RIGHT N0 WITH THE RIGHT PATH ||

file=open("/Users/CMoslonka/Desktop/Numerical_computations/mList2P8good.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        
        
#del(data)
#del(data_string)



#%% Making tries for a meq function

x=np.arange(-T0,T0+1)

plt.plot(x,meq[N0-2][0::2])
plt.plot(x,meq[N0-50][0::2])
plt.plot(x,meq[N0-100][0::2])
plt.plot(x,meq[N0-150][0::2])
plt.plot(x,meq[N0-200][0::2]) #note it's probably not a tanh when N is small
plt.plot(x,meq[N0-240][0::2])
plt.plot(x,meq[N0-250][0::2])

#plt.plot(x, np.tanh(2*x/T0))

#%%

for i in [50,100,150,200,250] :
    plt.plot(x, np.tanh((N0/i)*x/T0))

plt.plot(x,meq[N0-50][0::2])
plt.plot(x,meq[N0-100][0::2])
plt.plot(x,meq[N0-150][0::2])
plt.plot(x,meq[N0-200][0::2]) #note it's probably not a tanh when N is small
plt.plot(x,meq[N0-250][0::2])

#%% Making tests for fits with meq 

i=50

plt.plot(x, np.power((N0-i)/N0,1/5)*np.tanh((N0/(N0-i)*x/T0))) #We need a variating multip coef 
plt.plot(x,meq[N0-i][0::2])

plt.xlim((i-N0,N0-i))
#%%


#%%
def make_meqtest(T,M_T,j):
    if (T==0):
        return (0)
    if (abs(M_T)>T):
        return (0)
    else :
        return (j*np.power((T)/N0,1/5)*np.tanh((N0/(T)*M_T/T0)))
    
#%%

j=0.5

meqtest=np.zeros((N0,2*N0+1))

for T in range(N0):
    for M_T in range(2*N0+1):
        if (M_T%2==T%2): #is T is even
            meqtest[T][M_T]=make_meqtest(T,M_T-N0, j)
            
                
        
#%%

x=np.arange(-T0,T0+1)

plt.plot(x,meqtest[N0-2][0::2])
plt.plot(x,meqtest[N0-50][0::2])
plt.plot(x,meqtest[N0-100][0::2])
plt.plot(x,meqtest[N0-150][0::2])
plt.plot(x,meqtest[N0-200][0::2]) #note it's probably not a tanh when N is small
plt.plot(x,meqtest[N0-240][0::2])
plt.plot(x,meqtest[N0-250][0::2])


#%%

def y_test(t,chi, a):
    return chi*t-a*(t**3)/3
#%%

t=np.linspace(-1, 1, 1000)

chi=1

a=1

plt.plot(t,y_test(t,chi,a))
#%% Can we make a tranfer matrix with that ? YES

#%% Computing the transfer matrix 


list_Pf=[]
#list_sensi=[] 
P0=np.zeros(2*T0+1)
P0[T0]=1 #Origin of the probabilities

P=P0   
list_Pf.append(P)
for M in range(T0):  # We stop 
    K=np.zeros((2*T0+1,2*T0+1))
        
    for i in range(M+1): #len(data(M)) = M+1
        #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
        K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + meqtest[M][N0+data[M][i][1]])/2  #ith line and i+1 column 
        K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - meqtest[M][N0+data[M][i][1]])/2
    
    P=np.dot(K,P) #the order is good.
    list_Pf.append(P)
    
    
#%%Making the matrix for the random pick (easier to write)

S=np.zeros((2*T0+1,2*T0+1))
for i in range (-T0,T0+1):
    if i!=T0 : #Otherwise, bad index in the edges of the matrix
        S[T0+i][T0+i+1]=(i+1+T0)/(2*T0)
    if i!=-T0 :
        S[T0+i][T0+i-1]=1-((i-1+T0)/(2*T0)) #to be checked

#%% Computing some stuff from the proba distribution P
W=np.dot(K,S)
for T in range (N_iter) :
    P=np.dot(W,P)
    #Pprime=np.dot(S,P) #Removing one spin
    #P=np.dot(K,Pprime) #The last matrix K in memory is the one for T0-1 quenched spins

#P is the final probability


#%% Values_M

Values_M=[]
for k in range(-T0,T0+1):
    Values_M.append(k)


#%% Plots

plt.plot(Values_M[::2],P[::2])
plt.xlabel('Magnetisation Value')
plt.ylabel('Probability')
plt.show()

#%% Making an animation

# First we make a list of all the density probabilities that we want to 
# plot, then we make the animation
preci=100
list_j=np.linspace(0.7,2,preci)

list_Pf=np.zeros((int(preci),2*T0+1))

for j in range(preci):
    
    P0=np.zeros(2*T0+1)
    P0[T0]=1 #Origin of the probabilities
    
    P=P0   
    meqtest=np.zeros((N0,2*N0+1))

    for T in range(N0):
        for M_T in range(2*N0+1):
            if (M_T%2==T%2): #is T is even
                meqtest[T][M_T]=make_meqtest(T,M_T-N0, list_j[j])
                 
    for M in range(T0):  # We stop 
        K=np.zeros((2*T0+1,2*T0+1))
            
        for i in range(M+1): #len(data(M)) = M+1
            #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
            K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + meqtest[M][N0+data[M][i][1]])/2  #ith line and i+1 column 
            K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - meqtest[M][N0+data[M][i][1]])/2
        
        P=np.dot(K,P) #the order is good.
    list_Pf[j]=P


Values_M=[]
for k in range(-T0,T0+1):
    Values_M.append(k)


fig, ax = plt.subplots() # initialise la figure
line, = ax.plot(Values_M[0::2],list_Pf[0][0::2]) 
#plt.xlim(-1,1)
def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * N0)
    
def animate(i):
    line.set_data(Values_M[0::2],list_Pf[i][0::2]) 
    line.set_label('$j$ = %1.3f '%list_j[i])
    
    #ax.set_xlabel('Total magnetisation $M_{N_0}$',size=12)
    #ax.set_ylabel('Proba sensibility',size=12)
   
    # ax.spines['left'].set_position('center')
    # ax.spines['right'].set_color('none')
    # ax.spines['bottom'].set_position('center')
    # ax.spines['top'].set_color('none')
    #ax.ylim=(0,np.max(list_Pf[i]))
    #ax.xaxis.set_ticks_position('bottom')
    ax.legend(loc = 'upper right')
    return line,

steps=np.arange(1,preci)

ani=animation.FuncAnimation(fig,animate,steps,interval=100)
#ani.save("test_anim_j_influence_norm.mp4")
plt.show()

# Note : This is for progressive quenching only, there is no flow 
# whatsoever, this is just a qualitative work to see how polarised states
# arises from coupling. Maybe this only works for N0=2^8, who knows.

#%%

#Hey, i'm making changes to my file !