#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:25:59 2020

@author: CMoslonka
"""


#Making a continuous process with finite memory
#%% Loading relevant modules
import numpy as np
import random
import ast
import matplotlib.pyplot as plt
#import matplotlib.animation as animation

import time

plt.rcParams['animation.ffmpeg_path'] = '/Users/CMoslonka/ffmpeg'



#%% Parameters 

L=8
N0=2**L #System size

T0=2**7

N_iter=10**4

#%% Loading the data || NEEDS TO BE FOR THE RIGHT N0 WITH THE RIGHT PATH ||

file=open("mList2P8good.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        
        
#del(data)
#del(data_string)



#%% Normal PQ until T0
#random.seed(100) #setting the same seed for the two processes will allow to make the same first trajectory
start=time.time()

M_list = np.zeros(N_iter+T0) #list of all the magnetisation
memory_spin_list = np.zeros(N_iter+T0)
forgotten_spins = np.zeros(N_iter)
spin_list=np.zeros(T0)
spin_list[0]=random.choice([-1,1])#picking the first spin
M=spin_list[0]
M_list[0]=M
M=int(M)
for T in range(1,T0): #goes form 1 to T0-1
    if random.random()<(1+meq[T][N0+M])/2 : #Update rule
        spin_list[T]=1
        M=M+1
    else :
        spin_list[T]=-1
        M=M-1
    M_list[T]=M
    
#Just save the first spins
for i in range(T0) : memory_spin_list[i]=spin_list[i]



#From now, M will be either a odd or even integer depending on T0
#Forgetting process

for k in range(N_iter):
    
    forgotten_spins[k]=spin_list[0]
    
    M=M-spin_list[0] 
    M=int(M)
    for j in range(1,T0) : spin_list[j-1]=spin_list[j]

    if random.random()<(1+meq[T0-1][N0+M])/2 : #Check that you do not get 0
        spin_list[-1]=1
        M+=1
    else :
        spin_list[-1]=-1
        M+=-1
    memory_spin_list[T0+k]=spin_list[-1]
    M_list[T0+k]=M
    
end=time.time()
print(end-start)

X_T=np.zeros(N_iter+T0)

X_T[0]=memory_spin_list[0]
for T in range(1,N_iter+T0) :
    X_T[T]=(X_T[T-1]+memory_spin_list[T])
    
Y_T=np.zeros(N_iter)
Y_T[0]=forgotten_spins[0]
for T in range(1,N_iter) :
    Y_T[T]=(Y_T[T-1]+forgotten_spins[T])
    


#plt.plot(X_T[128:],Y_T,linewidth=0.5)    

#%% Testing for randomly picked spins

M_list_rand=[] #list of all the magnetisation
memory_spin_list_rand=[]
forgotten_spins_rand=[]
spin_list_rand=[random.choice([-1,1])] #picking the first spin
for T in range(1,T0): #goes form 1 to T0-1
    M_rand=np.sum(spin_list_rand)
    M_list_rand.append(M_rand)
    if random.random()<(1+meq[T][N0+M_rand])/2 : #Update rule
        spin_list_rand.append(1)
    else :
        spin_list_rand.append(-1)
M_list_rand.append(np.sum(spin_list_rand)) #Keep in memory the value of the magnetisation
for s in spin_list_rand : memory_spin_list_rand.append(s)

#From now, M_rand will be either a odd or even integer depending on T0
#Forgetting process

for k in range(N_iter):
    
    forgotten_spins_rand.append(spin_list_rand.pop(random.randint(0, T0-1))) #forgetting a random spin and storring it
    M_rand=np.sum(spin_list_rand)

    if random.random()<(1+meq[T0-1][N0+M_rand])/2 : # PQ as usual ?
        spin_list_rand.append(1)
    else :
        spin_list_rand.append(-1)
    M_list_rand.append(np.sum(spin_list_rand))
    memory_spin_list_rand.append(spin_list_rand[-1])
    
X_T_rand=[memory_spin_list_rand[0]]
for T in range(1,len(memory_spin_list_rand)) :
    X_T_rand.append(X_T_rand[-1]+memory_spin_list_rand[T])
    
X_T_rand=np.array(X_T_rand)

Y_T_rand=[forgotten_spins_rand[0]]
for T in range(1,len(forgotten_spins_rand)) :
    Y_T_rand.append(Y_T_rand[-1]+forgotten_spins_rand[T])
    
Y_T_rand=np.array(Y_T_rand)

#%% Writting module

XTfile=open("X_T_sequential.txt","w")
YTfile=open("Y_T_sequential.txt","w")
XTrandfile=open("X_T_random.txt","w")
YTrandfile=open("Y_T_random.txt","w")

XTfile.write("{")
XTrandfile.write("{")

for T in range(len(X_T)):
    XTfile.write(str(X_T[T]))
    XTfile.write(",")
    XTrandfile.write(str(X_T_rand[T]))
    XTrandfile.write(",")
  
XTfile.write("}")
XTrandfile.write("}")

YTfile.write("{")
YTrandfile.write("{")

for T in range(len(Y_T)):
    YTfile.write(str(Y_T[T]))
    YTfile.write(",")
    YTrandfile.write(str(Y_T_rand[T]))
    YTrandfile.write(",")
  
YTfile.write("}")
YTrandfile.write("}")

XTfile.close()
YTfile.close()
XTrandfile.close()
YTrandfile.close()


#%% Making the streamplot 

list_T=[]
for T in range(T0) : list_T.append(T)
x=np.array(list_T)

list_M=[]
for T in range(-T0,T0+1) : list_M.append(T)
y=np.array(list_M)



#%% Transfer matrix for PQ
mean=np.zeros((T0,2*T0+1))
for M in range(T0):  # We stop 
    K=np.zeros((2*T0+1,2*T0+1))
        
    for i in range(M+1): #len(data(M)) = M+1
        #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
        K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
        K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2
    
    mean[M]=np.dot(K,list_M) # I have no idea
    
#%%

u=np.ones((T0,2*T0+1))
v=mean


        
#%% Transfer matrix

list_Pf=[]
#list_sensi=[] 
P0=np.zeros(2*T0+1)
P0[T0]=1 #Origin of the probabilities

P=P0   
list_Pf.append(P)
for N in range(T0):  # 
    K=np.zeros((2*T0+1,2*T0+1))
    for M in range(-N,N+1,2) :   
        K[T0+M+1][T0+M]=(1 + meq[N][N0+M])/2  #ith line and i+1 column 
        K[T0+M-1][T0+M]=(1 - meq[N][N0+M])/2
    # for i in range(N+1): #len(data(M)) = M+1
    #     #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
    #     K[T0+data[N][i][1]+1][T0+data[N][i][1]]=(1 + data[N][i][2])/2  #ith line and i+1 column 
    #     K[T0+data[N][i][1]-1][T0+data[N][i][1]]=(1 - data[N][i][2])/2
    
    P=np.dot(K,P) #the order is good.
    list_Pf.append(P)





#%%

for i in np.linspace(2,8,20):
    for N in range(N0): 
        for T in range(-N,N+1,2):
            meqtest[N][N0+T]=test((T/2),i)
    runcell('Transfer matrix', '/Users/CMoslonka/Desktop/Numerical computations/Memory.py')
    plt.plot(P[0::2],label='j= %.2f' %i)
    plt.legend(loc = 'upper right')
    plt.show()
    
    
#%%