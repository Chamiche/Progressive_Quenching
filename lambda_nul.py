#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:06:44 2020

@author: CMoslonka
"""
#%% Theory of lambda = 0 

#%% Loading relevant modules
import numpy as np
import random
import ast
import matplotlib.pyplot as plt
#import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = '/Users/CMoslonka/ffmpeg'

#%%


L=4
N0=2**L #System size

T0=4

N_iter=10**5

#%%


file=open("meq4good.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]

#%%

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
        K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
        K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2
    
    P=np.dot(K,P) #the order is good.
    #list_Pf.append(P)
    
#%%


S=np.zeros((2*T0+1,2*T0+1))
for i in range (-T0,T0+1):
    if i!=T0 : #Otherwise, bad index in the edges of the matrix
        S[T0+i][T0+i+1]=(i+1+T0)/(2*T0)
    if i!=-T0 :
        S[T0+i][T0+i-1]=1-((i-1+T0)/(2*T0)) #to be checked
        
W=np.dot(K,S)

#%% Removing half of the lines and column (namely odd indicies)

Wprime=np.zeros((T0+1,T0+1))

for i in range(T0+1):
    for j in range(T0+1):
        Wprime[i][j]=W[2*i][2*j]
        
#%% WprimeT

WprimeT=Wprime.T
        
#%% Writting Wprime in a file
 
f=open("Reduced_transfer_matrix_4.txt",'w') #To be fixed sometime },} pbm
f.write('{')
for i in range(T0+1):
    f.write('{')
    for j in range(T0+1):
        plip=str(Wprime[i][j])
        f.write(plip)
        f.write(',')
    f.write('}')
f.write('}')
f.close()
#%%
f=open("Reduced_transposed_matrix_4.txt",'w') #To be fixed sometime },} pbm
f.write('{')
for i in range(T0+1):
    f.write('{')
    for j in range(T0+1):
        plip=str(WprimeT[i][j])
        f.write(plip)
        f.write(',')
    f.write('}')
f.write('}')
f.close()

#%% Loading the Eigenvectors/valuex


#%% Loading Eigenvalues from the text

file=open("EValues4.txt","r") #fixing the structure to make it right.
eigenvalues_string=file.read()
#We can probably use np.loadtext instead 
EValues = ast.literal_eval(eigenvalues_string)


#%%Same for the Evectors


file=open("EVectors4.txt","r") #fixing the structure to make it right.
eigenvectors_string=file.read()
#We can probably use np.loadtext instead 

EVectors4 = ast.literal_eval(eigenvectors_string)



#%%

file=open("EVectors4T.txt","r") #fixing the structure to make it right.
eigenvectors_string=file.read()
#We can probably use np.loadtext instead 
EVectors4T = ast.literal_eval(eigenvectors_string)


#%%


plt.plot(EVectors4[-1])
plt.plot(EVectors4T[-1])
