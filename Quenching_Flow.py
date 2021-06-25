#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:37:08 2020

@author: CMoslonka
"""
#Making a continuous process with finite memory
#%% Loading relevant modules
import numpy as np
import random
import math
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = '/Users/CMoslonka/ffmpeg'


import time

from scipy.sparse.linalg import eigs



#%% Loading the data || NEEDS TO BE FOR THE RIGHT N0 WITH THE RIGHT PATH ||
L=8
N0=2**L #System size 


T0=N0

N_iter=10**5


file=open("mList2P8good.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0
    

meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        
        
#del(data)
#del(data_string)

#%% ATTENTION Data for j=1.2

file=open("meq8j12_good.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0

meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        
#%% For N0=2p10=1024

N0=2**10
T0=N0
file=open("mList2P10.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0

meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]

#%% Another series of data 2p4

N0=16
T0=N0

file=open("meq2P4.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0

meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        
        
#%% Same for 2p5


N0=2**5
T0=N0

file=open("meq2P5.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0

meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        

#%% Same for 2p6


N0=2**6
T0=N0

file=open("meq2P6.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0

meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        
#%% Same for 2p7


N0=2**7
T0=N0

file=open("meq2P7.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0

meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        
        
        
#%% Same for 2p9


N0=2**9
T0=N0

file=open("meq2P9.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0

meq=np.zeros((N0,2*N0+1))
for T in range(N0):
    for i in range(len(data[T])):
        meq[T][N0 + data[T][i][1]]=data[T][i][2]
        
        
        
                

#%% Normal PQ until T0
#random.seed(100) #setting the same seed for the two processes will allow to make the same first trajectory
M_list=[] #list of all the magnetisation
spin_list=[random.choice([-1,1])] #picking the first spin
for T in range(1,T0): #goes form 1 to T0-1
    M=np.sum(spin_list)
    M_list.append(M)
    if random.random()<(1+meq[T][N0+M])/2 : #Update rule
        spin_list.append(1)
    else :
        spin_list.append(-1)
    
M_list.append(np.sum(spin_list)) #Keep in memory the value of the magnetisation

#From now, M will be either a odd or even integer depending on T0
#Forgetting process

for k in range(N_iter):
    
    spin_list.pop(0) #forgetting the first spin
    #spin_list.pop(random.randint(0, T0-1)) forgetting a random spin
    M=np.sum(spin_list)

    if random.random()<(1+meq[T0-1][N0+M])/2 : #Check that you do not get 0
        spin_list.append(1)
    else :
        spin_list.append(-1)
    M_list.append(np.sum(spin_list))
    
#%%Writting a file for Ken    
    
f=open("M_list_oldest_spin_removal.txt",'w')
for i in range(len(M_list)):
    f.write(str(M_list[i]))
    f.write(',')
f.close()

#%% Plotting

# plt.plot(M_list)
# plt.xlabel('Time')
# plt.ylabel('Magnetisation of the quenched part')
# plt.title('trajectory for the elder picking')
# plt.show()

M_list_norm=np.array(M_list)/T0

#Making some stats

plt.hist(M_list,range=(-T0,T0+1),bins=(2*T0),density=True)
plt.xlabel('Normalized Magnetisation $m=M/T_0$')
plt.ylabel('Density count')
plt.title('$N_{iter}$= %i and $T_0= %i$' %(N_iter,T0))
plt.suptitle('Histogram of positions taken by $M$ for oldest spin removal process')
plt.savefig('hist_T0_16_N0_10p8.pdf')
plt.show()



#%% Testing for randomly picked spins
#random.seed(100)

M_list_rand=[] #list of all the magnetisation
spin_list_rand=[random.choice([-1,1])] #picking the first spin
for T in range(1,T0): #goes form 1 to T0-1
    M_rand=np.sum(spin_list_rand)
    M_list_rand.append(M_rand)
    if random.random()<(1+meq[T][N0+M_rand])/2 : #Update rule
        spin_list_rand.append(1)
    else :
        spin_list_rand.append(-1)
M_list_rand.append(np.sum(spin_list_rand)) #Keep in memory the value of the magnetisation

#From now, M_rand will be either a odd or even integer depending on T0
#Forgetting process
spin_list_rand=np.array(spin_list_rand)
start = time.time()
for k in range(N_iter):
    #spin_list_rand.pop(random.randint(0, T0-1)) #forgetting a random spin
    rank=random.randint(0, T0-1)
    spin_list_rand[rank]=0
    M_rand=np.sum(spin_list_rand)

    if random.random()<(1+meq[T0-1][N0+M_rand])/2 : # PQ as usual ?
        spin_list_rand[rank]=1
        #spin_list_rand.append(1)
    else :
        #spin_list_rand.append(-1)
        spin_list_rand[rank]=-1
    M_list_rand.append(np.sum(spin_list_rand))
    
end = time.time()
print(end - start)
    
# Plotting

# plt.plot(M_list_rand)
# plt.xlabel('Time')
# plt.ylabel('Magnetisation of the quenched part')
# plt.title('Trajectory for random picking')
# plt.show()

#M_list_rand_norm=np.array(M_list_rand)/T0

#%% Making some stats

plt.hist(M_list_rand,range=(-T0,T0+1),bins=(2*T0),density=True)
plt.xlabel('Normalized Magnetisation $m=M/T_0$')
#plt.text('$T_0 = %i $' %T0,bbox=dict(facecolor='wheat', alpha=0.9, loc= 'upper right' ))
plt.grid()
plt.title('Histogram of the positions taken by $M$ for the random spin removal process')
#plt.savefig("historand_T0_128_N0_10p8.pdf")
plt.show()

#%% Making another file

f=open("M_list_random_spin_removal.txt",'w')
for i in range(len(M_list_rand)):
    f.write(str(M_list_rand[i]))
    f.write(',')
f.close()

#%%Making the correlation function 
import time
start=time.time()

Corr_old=[]

#Corr_old=np.zeros(3000)
for tau in range(3000):
    mean_tau=0
    for i in range(len(M_list)-tau):
        mean_tau+=(M_list[i]*M_list[i+tau])
    Corr_old.append(mean_tau/(len(M_list)-tau))
    #Corr_old[tau]=(mean_tau/(len(M_list)-tau))
    
end=time.time()
print(end-start)
#%% Same for random spin removal

Corr_rand=[]
for tau in range(3000):
    mean_tau=0
    for i in range(len(M_list_rand)-tau):
        mean_tau+=(M_list_rand[i]*M_list_rand[i+tau])
    Corr_rand.append(mean_tau/(len(M_list_rand)-tau))
    
#%% Plotting the 2 correlation fucntions

plt.plot(Corr_old,label='Corr_old')
plt.plot(Corr_rand, label='Corr_rand')
plt.grid()
plt.legend()

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '\n'.join((
    r'$N_{iter}=10^4$ points',
    r'$T_0= %i$' %T0))

# place a text box in upper left in axes coords
plt.text(1500,1000, textstr, fontsize=14, bbox=props)

plt.xlabel('$\\tau $')
plt.ylabel('$<M(t)M(t+\\tau )>_t$')
plt.title('Correlation functions of the two forgetting processes')
#plt.savefig('Good_correlations.pdf')
plt.show()


#%% Generating different curves for values of T0

for k in [4,5,6,7]:
    T0=2**k
    runcell('Normal PQ until T0')
    runcell('Testing for randomly picked spins')
    runcell('Making the correlation function')
    runcell('Same for random spin removal')
    runcell('Plotting the 2 correlation fucntions')
    


#%% Oscillations for tau>T0 ?

Corr_old_test=[]
for tau in range(T0,3000):
    mean_tau=0
    for i in range(len(M_list)-tau):
        mean_tau+=(M_list[i]*M_list[i+tau])
    Corr_old_test.append(mean_tau/(len(M_list)-tau))
    
# Conclusion : Curious curve, but I cannot explain it
    
#%% Same for the random stuff

Corr_rand_test=[]
for tau in range(T0,3000):
    mean_tau=0
    for i in range(len(M_list_rand)-tau):
        mean_tau+=(M_list_rand[i]*M_list_rand[i+tau])
    Corr_rand_test.append(mean_tau/(len(M_list_rand)-tau))


#%% Testing if the PQ really works (sampling the proba with a lot of trajectories)
N_test=100000
M_test=[] #list of all the magnetisation
for plop in range(N_test):

    spin_test=[random.choice([-1,1])] #picking the first spin
    for T in range(1,T0): #goes form 1 to T0-1
        M=np.sum(spin_test)
        if random.random()<(1+meq[T][N0+M])/2 : #Update rule
            spin_test.append(1)
        else :
            spin_test.append(-1)
    M_test.append(np.sum(spin_test))
    
    
    
#%%

prout=open('Corr_func.txt',"w")
prout.write(str(Corr_old))
prout.write(str(Corr_rand))
prout.close
#%% Plotting those things (checking the PQ)

plt.hist(M_test,bins=2*T0, density=True)
plt.xlabel('Magnetisation')
plt.ylabel('Density count')
plt.title('Sampling trajectories')
#plt.savefig('Test_sampling.pdf')
plt.show()

# Verdict : It's good and does what we want

#%% Counting without the histogram

Values_M=np.arange(-T0,T0+1,2)
    
M_count=[]
for k in Values_M :
    M_count.append(M_list_rand.count(k))
  

npM=np.array(M_count) #si on veut faire des opérations du type passer au log
npVal=np.array(Values_M)

npM=npM/np.sum(M_count) #normalizing


#%%

plt.plot(Values_M, npM, 'b.')
plt.plot(Values_M,np.max(npM)*P[0::2]/np.max(P),'r-')
plt.xlabel('$M$ values')
plt.ylabel('Count value')
plt.title('Value count')
plt.show()

#%% Gaussian fit ?

from scipy.optimize import curve_fit

#%% gaussian function

def fit(x,a,sigma):
    return((a/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x**2)/(2*sigma**2)))

#%% actual fit 

OK=curve_fit(fit, Values_M, npM)
#We find that sigma is sqrt(T0) and a=2 (approx) which is kinda what we expected

#%% Plotting data and fit


t=np.linspace(-T0,T0,1000)
plt.plot(t,fit(t,2,np.sqrt(T0)),label='Theoretical curve')
plt.plot(Values_M,npM,'r.',label='Numerical count')
plt.legend()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '\n'.join((
    r'$N_{iter}=10^6$',
    r'$T_0= %i$' %T0))

# place a text box in upper left in axes coords
plt.text(-T0,fit(np.sqrt(T0),2,np.sqrt(T0)), textstr, fontsize=14, bbox=props)
plt.xlabel('$M$')
plt.ylabel('Normalized density')
plt.title('Normalized count of $M$ values')
plt.savefig('fit_T0_%i.pdf' %T0)
plt.show()

#%% Generating the cruves

for k in [4,5,6,7]:
    T0=2**k
    runcell('Normal PQ until T0')
    Values_M=[]
    for k in range(-T0,T0+1,2):
        Values_M.append(k)
    M_count=[]
    for k in Values_M :
        M_count.append(M_list.count(k))
      
    
    npM=np.array(M_count) #si on veut faire des opérations du type passer au log
    npVal=np.array(Values_M)
    
    npM=npM/np.sum(M_count) #normalizing
    runcell('Plotting data and fit')



#%% Making the same thing for the random pick of spins


for k in [4,5,6,7]:
    T0=2**k
    runcell('Testing for randomly picked spins')
    Values_M=[]
    for k in range(-T0,T0+1,2):
        Values_M.append(k)
    M_count=[]
    for k in Values_M :
        M_count.append(M_list_rand.count(k))
      
    
    npM=np.array(M_count) #si on veut faire des opérations du type passer au log
    npVal=np.array(Values_M)
    
    npM=npM/np.sum(M_count) #normalizing
    runcell('Plotting data and fit')

#%% Computing the transfer matrices and get the PQ


list_Pf=[]
#list_sensi=[] 
P0=np.zeros(2*T0+1)
P0[T0]=1 #Origin of the probabilities
#P0[T0+2]=1 #Origin of the probabilities

P=P0   
list_Pf.append(P)
for M in range(T0):  # We stop 
    K=np.zeros((2*T0+1,2*T0+1))
        
    for i in range(M+1): #len(data(M)) = M+1
        #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
        K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
        K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2
    
    P=np.dot(K,P) #the order is good.
    # list_Pf.append(P)    # UNCOMMENT TO ACCESS ALL THE DISTRIBUTIONS
    

#%% Compute directly the total transfer matrix

for M in range(T0):  # We stop 
    K=np.zeros((2*T0+1,2*T0+1))    
    for i in range(M+1): #len(data(M)) = M+1
        #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
        K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
        K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2
    if(M==0):
        A=K #We stock the first value of the TM
    elif(M==1):
        S=np.dot(K,A) #We make the first product
    else  :
        S=np.dot(K,S)

    
#%%Making the matrix for the random pick (easier to write)

S=np.zeros((2*T0+1,2*T0+1))
for i in range (-T0,T0+1):
    if i!=T0 : #Otherwise, bad index in the edges of the matrix
        S[T0+i][T0+i+1]=(i+1+T0)/(2*T0)
    if i!=-T0 :
        S[T0+i][T0+i-1]=1-((i-1+T0)/(2*T0)) #to be checked

#%% Computing some stuff from the proba distribution P
W=np.dot(K,S)
P0=np.zeros(2*T0+1)
P0[T0]=1
for T in range (N_iter) :
    P0=np.dot(W,P0)
    #Pprime=np.dot(S,P) #Removing one spin
    #P=np.dot(K,Pprime) #The last matrix K in memory is the one for T0-1 quenched spins

#P is the final probability

#%% Regular plotting from a given distribution

P0=np.zeros(2*T0+1)
P0[T0]=1
#P0[-(T0-50+1)]=0.5
N_iter=3000
W=np.dot(K,S)
for T in range (N_iter) :
    P0=np.dot(W,P0)
    if(T%300==0) :
        plt.plot(P0[0::2], label='$T= %i$' %T)
        plt.title('$S= %1.3f $' %np.sum(P0))
        plt.legend()
        plt.show()


#%% Values_M

Values_M=np.arange(-T0,T0+1,2)


#%% Plots

plt.plot(Values_M[::2],P[::2],'-b',label='Stationnary')
plt.plot(Values_M[0::2],list_Pf[-1][0::2],'-r', label='ProgressiveQuenching')
plt.xlabel('Magnetisation Value')
plt.ylabel('Probability')
plt.grid()
plt.legend()
plt.title('Comparison between the 2 methods')
#plt.savefig('Proba_density.pdf')
plt.show()

#%% Assymetry (Please make meq = 0)

P=np.array(P)
list_Pf=np.array(list_Pf)

plt.plot(Values_M[0::2],list_Pf[-1][0::2] - P[::2],'.b')
plt.xlabel('Magnetisation Value')
plt.ylabel('Probability difference')
plt.title('Difference between the 2 PDF')
plt.grid()
plt.show()

#%% Alt test

for T in range (N_iter) :
    Q=np.zeros(2*T0+1)
    for i in range(1,2*T0-1) :
        Q[i]=(1+i)/(2*T0)*P[i+1]+(1-((i-1)/(2*T0)))*P[i-1]
    P=np.dot(K,Q)
#%% Other plots

plt.pcolormesh(np.dot(K,S))
plt.title('Transfer matrix components')
#plt.savefig('TM.pdf')
plt.show()



#%%

def g(x,m,s):
    return np.exp(-((x-m)**2)/(2*s**2))


#%% Writting a file for mathematica computation 

f=open("Transfer_matrix.txt",'w')
for i in range(2*T0+1):
    f.write('{')
    for j in range(2*T0+1):
        plip=str(W[i][j])
        f.write('{')
        f.write(plip)
        f.write('}')
    f.write('}')
f.close()

#%% Removing half of the lines and column (namely odd indicies)

W=np.dot(K,S)

Wprime=np.zeros((T0+1,T0+1))

for i in range(T0+1):
    for j in range(T0+1):
        Wprime[i][j]=W[2*i][2*j]
        
#%% Writting Wprime in a file
 
f=open("Reduced_transfer_matrix_2p10_682.txt",'w')
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

WprimeT=Wprime.T
#%% Writting Wprime.T in a file
 
f=open("Reduced_transfer_matrix_transposed.txt",'w')
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

#%% Loading Eigenvalues from the text

file=open("EigenValues_Reduced_Matrix_ready.txt","r") #fixing the structure to make it right.
eigenvalues_string=file.read()
#We can probably use np.loadtext instead 
EValues = ast.literal_eval(eigenvalues_string)


#%%Same for the Evectors


file=open("Eigenvectors_Mathematica_ready.txt","r") #fixing the structure to make it right.
eigenvectors_string=file.read()
#We can probably use np.loadtext instead 
EVectors = ast.literal_eval(eigenvectors_string)
EVectors = -np.array(EVectors)


#%% Ploting the Spectrum

plt.plot(EValues, '-b')
plt.grid()
plt.xlabel('Index of the Eigenvalue')
plt.ylabel('Value of the Eigenvalue')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '$T_0= %i$' %T0

# place a text box in upper left in axes coords
plt.text(textstr, fontsize=14, bbox=props)

plt.title('Spectrum of the Transfer Matrix')
#plt.savefig('Spectrum_TM.pdf')
plt.show()




#%% Plotting the first eigenvector

plt.plot(Values_M[::2], EVectors[0], '-b')

plt.grid()
plt.xlabel('Values of $M$')
plt.ylabel('Components of the eigenvector')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '\n'.join(( '$T_0= %i$' %T0 , r'$ \lambda = 1 $'))

# place a text box in upper left in axes coords
#plt.text(textstr, fontsize=14, bbox=props, loc="upper left")

plt.title('Eigenvector associated with the $ \lambda = 1 $ eigenvalue')
#lt.savefig('First_eigenvector.pdf')
plt.show()

#%% Comparing the others 

plt.plot(Values_M[::2], EVectors[0], label='1st EV')
plt.plot(Values_M[::2], EVectors[1], label='2nd EV')
plt.plot(Values_M[::2], EVectors[2], label='3rd EV')
plt.plot(Values_M[::2], EVectors[3], label='4th EV')

plt.grid()
plt.xlabel('Values of $M$')
plt.ylabel('Components of the eigenvector')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '$T_0= %i$' %T0

# place a text box in upper left in axes coords
plt.text(80,-0.12, textstr, fontsize=14, bbox=props)

plt.legend()

plt.title('First 4 Eigenvectors')
#plt.savefig('First_4_eigenvector.pdf')
plt.show()


#%% Generating 200 ? curves and making stats out of this

N_curves = 2000

Traj_random=[]
for i in range(N_curves):
    
    runcell('Testing for randomly picked spins')
    Traj_random.append(M_list_rand)
    
#%% Writting a file to prevent recalculations

# f=open("Trajectories_10p4_points.txt",'w')
# f.write(str(Traj_random))
#%% Reading the said file and making a list

f=open("Trajectories_10p4_points_T0=128.txt",'r')
Traj_random_str=f.read()
Traj_random=ast.literal_eval(Traj_random_str)
del(Traj_random_str)

    
#%% Computing some stats with those curves

#Distribution of last points

last_points=[]
for i in range(N_curves):
    last_points.append(Traj_random[i][-1])

#%% counting occurences of points

Values_M=[]
for k in range(-T0,T0+1,2):
    Values_M.append(k)
    
last_points_count=[]
for k in Values_M :
    last_points_count.append(last_points.count(k))
  

last_points_count=np.array(last_points_count) #si on veut faire des opérations du type passer au log
Values_M=np.array(Values_M)

#%% Plotting those

plt.plot(Values_M,last_points_count,'.')
plt.grid()
plt.show()

#Conclusion, we get a gaussian : WHY ? 


#%% Same with Histogram

plt.hist(last_points,range=(-T0,T0+1), bins=(2*T0)) #To have a good histogram, we need to specify the range
plt.xlabel('$M$ value')
plt.ylabel('Historgram count')
plt.title("Histogram of the last positions of 2000 trajectories")
plt.text(70,80,'\n'.join(( '$N_0= 10^4$', r'$T_0=128$')),bbox=dict(facecolor='blue',alpha=0.3))
plt.savefig('histo_lastpos_2000curves.pdf')
plt.show()

#%% Correlation function

list_corr_rand=[]
for k in range(20) : #number of curves we want the mean on
    Corr_rand=[]
    for tau in range(1000):
        mean_tau=0
        for i in range(len(Traj_random[k])-tau):
            mean_tau+=(Traj_random[k][i]*Traj_random[k][i+tau])
        Corr_rand.append(mean_tau/(len(Traj_random[k])-tau))
    list_corr_rand.append(Corr_rand)
    
    
#%% Mean of the correlation functions

mean_corr_func=[]
for i in range(len(list_corr_rand[1])) :
    mean_point=0
    for j in range(len(list_corr_rand)):
        mean_point+=list_corr_rand[j][i]
    mean_corr_func.append(mean_point/len(list_corr_rand))
#%%

plt.plot(mean_corr_func)
plt.title("Mean correlation function other 20 trajectories")
plt.xlabel('$\\tau $')


textstr = '\n'.join(('$T_0= %i$' %T0 , 
                    r'$N_{iter}= 10^4 $'))

props = dict(boxstyle='round', facecolor='blue', alpha=0.3)

plt.text(600,80, textstr, fontsize=14, bbox=props)
plt.grid()
plt.show()
        
#%%
 

file=open("Eigenvectors_Transposed.txt","r") #fixing the structure to make it right.
eigenvectors_string=file.read()
#We can probably use np.loadtext instead 
EVectors_transposed = ast.literal_eval(eigenvectors_string)


#%% ON VÉRIFIE L'ORTHOGONALITÉ DES VECTEURS PROPRES

print(np.dot(EVectors[1],EVectors_transposed[5]))



#%% Eigenvalues matrix with j12 : loading from data

file=open("Eigenvalues_j12.txt","r") #fixing the structure to make it right.
eigenvalues_string=file.read()
#We can probably use np.loadtext instead 
EValues12 = ast.literal_eval(eigenvalues_string)



#%% Plotting


plt.plot(EValues12, '-b')
plt.grid()
plt.xlabel('Index of the Eigenvalue')
plt.ylabel('Value of the Eigenvalue')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '$T_0= %i$' %T0

# place a text box in upper left in axes coords
plt.text(80,0.8, textstr, fontsize=14, bbox=props)

plt.title('Spectrum of the Transfer Matrix')
#plt.savefig('Spectrum_TM.pdf')
plt.show()



#%%

file=open("Eigenvectors_j12.txt","r") #fixing the structure to make it right.
eigenvectors_string=file.read()
#We can probably use np.loadtext instead 
EVectors12 = ast.literal_eval(eigenvectors_string)


#%%

plt.plot(Values_M, EVectors12[0], '-b')

plt.grid()
plt.xlabel('Values of $M$')
plt.ylabel('Components of the eigenvector')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '\n'.join(( '$T_0= %i$' %T0 , r'$ \lambda = 1 $'))

# place a text box in upper left in axes coords
plt.text(80,0.10, textstr, fontsize=14, bbox=props)

plt.title('Eigenvector associated with the $ \lambda = 1 $ eigenvalue')
#plt.savefig('First_eigenvector.pdf')
plt.show()


#%% Trying an animation 


fig, ax = plt.subplots() # initialise la figure
val=np.arange(-N0,N0+1,2,dtype=float)/N0
line, = ax.plot(val,list_Pf[1][0::2]*(1/np.max(list_Pf[i]))) 
#plt.xlim(-1,1)
plt.ylim((0,1.1))
def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * N0)
    
def animate(i):
    line.set_data(np.arange(-i,i+1,2,dtype=float)/i,list_Pf[i][N0-i:N0+i+1:2]*(1/np.max(list_Pf[i]))) 
    line.set_label('$T_0$ = {}'.format(i))
    
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

steps=np.arange(1,N0)

ani=animation.FuncAnimation(fig,animate,steps,interval=100)
ani.save("anim_proba2P8.mp4")
plt.show()


#%% Let this blank

x=np.linspace(-1, 1,200)
#%%
fig, ax = plt.subplot()

ax.plot(x,np.tanh(x))

#%%import matplotlib.pyplot as plt
plt.ioff()
plt.plot([1.6, 2.7])

#%% Asymetry study !!


for i in range(0,N0,2):
    data[i][i//2][2]= 0#random.gauss(0,0.001) # This sets all randomness to 0 in 
    # Numerical estimation of meq with M ==0
    
    # Result is no more asymetry
#%% recompute 

runcell('Computing the transfer matrix', '/Users/CMoslonka/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')

#%%
    
a=np.zeros(T0//2)

for i in range(len(a)):
    a[i]=P[0::2][i]-P[0::2][-(i+1)] #The "asymmetry function"

plt.plot(a,'.b') 

#%%As a comparison we can plot the same asymmetry function for the Eigenvalue

b=np.zeros(T0//2)

for i in range(len(b)):
    b[i]=EVectors[0][i]-EVectors[0][-(i+1)]
    
plt.plot(b,'.r')

#%%Now we can compare the 2 distributions : 

plt.plot(EVectors[0]-P[0::2]) #Now we see that it's probably random noise (and it's symmetric) that comes from computation (I guess ?)

#%%

#%% Trying our formula for the Stationnary distribution 


Pst_test=np.zeros(T0+1)
for i in range((T0+1)): #the i's are basically the values of M
    prod1=1
    prod2=1
    for k in range(i-1):#k  must take into account up to 
        prod1 *= (1+data[T0-1][k+1][2])
        prod2 *= (1-data[T0-1][k+1][2])
    Pst_test[i] = math.comb(T0,i)* prod1/prod2

Pst_test*=1/np.sum(Pst_test)  




#%% 

plt.plot(Pst_test)

#%% Testing what random values give

Pst_test2=np.zeros(T0+1)
for i in range((T0+1)):
    prod1=1
    prod2=1
    for k in range(i):
        a=random.random()
        prod1 *= (1+a)
        prod2 *= (1-a)
    Pst_test2[i] = math.comb(T0+1,i)* prod1/prod2

Pst_test2*=1/np.sum(Pst_test2)  


#%% Showing the predicted values

T1=N0-1
P_pred=np.zeros(T1+1)
M_values_pred=np.zeros(T1+1)

for i in range(T1+1):
    M_values_pred[i]=(2*i-T1)+(N0-T1)*data[T1][i][2]
    
P_pred[0]=2*list_Pf[T1][N0-T1:N0+T1+1:2][0]/(M_values_pred[1]-M_values_pred[0])

for j in range(1,T1):
    P_pred[j]=2*list_Pf[T1][N0-T1:N0+T1+1:2][j]/(M_values_pred[j+1]-M_values_pred[j-1])

P_pred[T1]=2*list_Pf[T1][N0-T1:N0+T1+1:2][T1]/(M_values_pred[-1]-M_values_pred[-2])

P_pred*=2

#%% Making a nice plot for the presentation

list_P_pred=[]
list_M_values=[]
for T1 in [8,16,32,64,128] :
    P_pred=np.zeros(T1+1)
    M_values_pred=np.zeros(T1+1)
    
    for i in range(T1+1):
        M_values_pred[i]=(2*i-T1)+(N0-T1)*data[T1][i][2]
        
    P_pred[0]=2*list_Pf[T1][N0-T1:N0+T1+1:2][0]/(M_values_pred[1]-M_values_pred[0])
    
    for j in range(1,T1):
        P_pred[j]=2*list_Pf[T1][N0-T1:N0+T1+1:2][j]/(M_values_pred[j+1]-M_values_pred[j-1])
    
    P_pred[T1]=2*list_Pf[T1][N0-T1:N0+T1+1:2][T1]/(M_values_pred[-1]-M_values_pred[-2])
    
    P_pred*=2
    list_P_pred.append(P_pred)
    list_M_values.append(M_values_pred)

#%% Generating the plot


plt.plot(list_M_values[0],list_P_pred[0],'.--',label='$T_i=8$',)
plt.plot(list_M_values[1],list_P_pred[1],'.--',label='$T_i=16$')
plt.plot(list_M_values[2],list_P_pred[2],'.',label='$T_i=32$')
plt.plot(list_M_values[3],list_P_pred[3],'.',label='$T_i=64$')
plt.plot(list_M_values[4],list_P_pred[4],'.',label='$T_i=128$')

plt.plot(Values_M,list_Pf[-1][0::2],'b-',label = 'True Value')

plt.xlabel('Magnetisation value')
plt.ylabel('Probability density')

plt.title('Inference of the probability distribution for different points $T_i$')

plt.legend()

plt.savefig('Inference_2P8.pdf')

plt.show()

#%% Just showing the shape of the T0=16, just to show how our inference is cool
val16=np.arange(-16,17,2)
plt.plot(val16,list_Pf[16][N0-16:N0+16+1:2],'r.-')
plt.xlabel('Magnetisation value')
plt.ylabel('Probability density')
plt.title('Magnetisation distribution at $T_0=16$')
plt.savefig('distribT016.pdf')
plt.show()

#%% Change of shape figure

for T0 in [16,32,64,128,256] :
    val=np.arange(-T0,T0+1,2,dtype=float)
    val/=T0
    plt.plot(val,list_Pf[T0][N0-T0:N0+T0+1:2]/np.max(list_Pf[T0][N0-T0:N0+T0+1:2]),'.--',label='$N_0=$ %i' %T0)
plt.legend()    
plt.xlabel('Normalized magnetisation $M/N_0$')
plt.ylabel('Scaled probability distribution')
plt.title('Change of shape with system size $N_0$')
plt.savefig('Shape_change.pdf')
plt.show()


#%% Writing a sequence of transfert matrices in order to have the different spectrums and lamnda 1 values

for T0 in [10,24,48,75,112,156,188,220]: 
    #The order is : compute K, compute W, write W in different files, job done
    runcell('Computing the transfer matrix', 'C:/Users/Chamo/Desktop/Numerical computations/Quenching_Flow.py')
    runcell('Making the matrix for the random pick (easier to write)', 'C:/Users/Chamo/Desktop/Numerical computations/Quenching_Flow.py')
    W=np.dot(K,S)
    Wprime=np.zeros((T0+1,T0+1))

    for i in range(T0+1):
        for j in range(T0+1):
            Wprime[i][j]=W[2*i][2*j]
            
     
    f=open("Reduced_transfer_matrix_2p8_%i.txt" %T0,'w')
    f.write('{{')
    for i in range(T0+1):
        f.write('{')
        for j in range(T0+1):
            plip=str(Wprime[i][j])
            f.write(plip)
            f.write(',')
        f.write('},')
    f.write('}}')
    f.close()
#%%plotting some spectrums

s8=[1.,0.92859,0.832375,0.725838,0.608372,0.480157,0.339516,0.182622,-2.77556e-17]
s10=[1.,0.949193,0.877153,0.798075,0.711026,0.616993,0.515621,0.406139,0.286838,0.154212,1.11022e-16]
s16=[1.,0.976267,0.9377,0.896186,0.850042,0.800624,0.747952,0.692215,0.633383,0.57137,0.505977,0.436886,0.363598,0.285324,0.200741,0.107391,-5.55112e-17]
s24=[1.,0.988246,0.966351,0.94311,0.916868,0.88876,0.858797,0.8272,0.794027,0.75934,0.723157,0.685479,0.646285,0.605536,0.56317,0.519103,0.473222,0.425376,0.375364,0.32291,0.267632,0.208963,0.146022,0.0773099,0.]
s32=[1.,0.99302,0.978678,0.963514,0.946196,0.927577,0.907685,0.886691,0.864659,0.84165,0.8177,0.792836,0.767071,0.740412,0.712859,0.684404,0.655033,0.624725,0.593453,0.561182,0.52787,0.493461,0.457891,0.421079,0.382925,0.343304,0.302054,0.258966,0.213758,0.166032,0.115207,0.0603728,2.84495e-16]
s48=[1.,0.996734,0.98911,0.980991,0.971601,0.961405,0.950459,0.938862,0.926666,0.913916,0.900643,0.886872,0.872621,0.857906,0.842737,0.827122,0.811069,0.79458,0.777659,0.760305,0.742518,0.724296,0.705634,0.686528,0.666972,0.646957,0.626475,0.605515,0.584065,0.562111,0.539637,0.516624,0.493053,0.4689,0.444138,0.418737,0.392663,0.365874,0.338325,0.30996,0.280715,0.250511,0.219254,0.186827,0.153082,0.11783,0.0808187,0.0417038,6.245e-17]
s64=[1., 0.99812, 0.993374, 0.988253, 0.982291, 0.975757, 0.968713,0.96122, 0.953319, 0.945042, 0.936413, 0.927451, 0.918172, 0.908588,0.89871, 0.888546, 0.878102, 0.867385, 0.856399, 0.845148, 0.833634,0.821859, 0.809826, 0.797534, 0.784985, 0.772178, 0.759112, 0.745786,0.732199, 0.718349, 0.704233, 0.689848, 0.675191, 0.660259, 0.645047,0.629549, 0.613763, 0.59768, 0.581296, 0.564603, 0.547594, 0.53026,0.512593, 0.494583, 0.476218, 0.457487, 0.438378, 0.418876, 0.398964,0.378627, 0.357844, 0.336594, 0.314853, 0.292594, 0.269786, 0.246395,0.22238, 0.197696, 0.172288, 0.146093, 0.119036, 0.0910241,0.0619466, 0.031664, -4.85723e-17]
s75=[1.,0.998618,0.994996,0.991049,0.986441,0.981364,0.975875,0.970022,0.96384,0.957354,0.950585,0.943549,0.936258,0.928725,0.920957,0.912963,0.904749,0.89632,0.88768,0.878835,0.869786,0.860536,0.851087,0.841442,0.831601,0.821565,0.811336,0.800912,0.790295,0.779484,0.768479,0.757278,0.745881,0.734287,0.722494,0.7105,0.698304,0.685904,0.673296,0.660479,0.647449,0.634204,0.620739,0.607052,0.593138,0.578994,0.564613,0.549992,0.535125,0.520006,0.50463,0.488988,0.473076,0.456883,0.440403,0.423626,0.406543,0.389142,0.371413,0.353343,0.334918,0.316123,0.296941,0.277355,0.257345,0.236887,0.215956,0.194525,0.172562,0.15003,0.126886,0.103085,0.0785677,0.0532697,0.0271119,3.1225e-17]
s128=[1.,0.999516,0.998115,0.996531,0.994664,0.992572,0.990287,0.987829,0.985215,0.982456,0.979563,0.976543,0.973403,0.97015,0.966788,0.96332,0.959752,0.956085,0.952324,0.94847,0.944525,0.940492,0.936373,0.932169,0.927881,0.923511,0.91906,0.914528,0.909918,0.905229,0.900463,0.89562,0.890701,0.885706,0.880635,0.87549,0.87027,0.864976,0.859608,0.854167,0.848651,0.843062,0.837399,0.831663,0.825854,0.819971,0.814014,0.807984,0.80188,0.795702,0.789449,0.783122,0.776721,0.770245,0.763693,0.757065,0.750362,0.743582,0.736726,0.729792,0.72278,0.71569,0.708521,0.701273,0.693944,0.686536,0.679045,0.671473,0.663819,0.656081,0.648259,0.640352,0.632359,0.624279,0.616112,0.607857,0.599512,0.591077,0.582551,0.573932,0.56522,0.556412,0.547509,0.538509,0.529411,0.520212,0.510913,0.501511,0.492006,0.482395,0.472677,0.46285,0.452913,0.442864,0.432701,0.422423,0.412027,0.401511,0.390874,0.380112,0.369225,0.358209,0.347062,0.335781,0.324364,0.312809,0.301111,0.289269,0.277279,0.265138,0.252843,0.240389,0.227773,0.214992,0.202041,0.188916,0.175612,0.162124,0.148448,0.134579,0.12051,0.106236,0.0917501,0.0770467,0.0621183,0.0469576,0.0315567,0.0159072,-4.51028e-17]
s256=[1.,0.999878,0.999495,0.999044,0.998506,0.997892,0.997213,0.996476,0.995683,0.99484,0.99395,0.993016,0.992039,0.991021,0.989965,0.988871,0.987741,0.986577,0.985379,0.984148,0.982885,0.981591,0.980267,0.978913,0.97753,0.976119,0.974679,0.973212,0.971718,0.970198,0.968651,0.967078,0.96548,0.963856,0.962208,0.960535,0.958838,0.957117,0.955372,0.953603,0.951811,0.949996,0.948158,0.946297,0.944413,0.942507,0.940578,0.938627,0.936654,0.934659,0.932643,0.930604,0.928544,0.926462,0.924359,0.922235,0.920089,0.917922,0.915734,0.913524,0.911294,0.909043,0.906771,0.904478,0.902164,0.899829,0.897474,0.895098,0.892701,0.890283,0.887845,0.885386,0.882907,0.880407,0.877886,0.875345,0.872783,0.870201,0.867597,0.864974,0.86233,0.859665,0.856979,0.854273,0.851547,0.848799,0.846031,0.843242,0.840433,0.837603,0.834752,0.83188,0.828987,0.826073,0.823139,0.820183,0.817207,0.814209,0.811191,0.808151,0.80509,0.802008,0.798904,0.79578,0.792633,0.789466,0.786277,0.783066,0.779834,0.77658,0.773304,0.770007,0.766688,0.763347,0.759983,0.756598,0.753191,0.749761,0.746309,0.742835,0.739338,0.735819,0.732277,0.728713,0.725125,0.721515,0.717882,0.714226,0.710547,0.706845,0.703119,0.69937,0.695597,0.691801,0.687981,0.684138,0.680271,0.676379,0.672464,0.668524,0.66456,0.660572,0.656559,0.652521,0.648459,0.644372,0.64026,0.636123,0.631961,0.627774,0.623561,0.619323,0.615058,0.610769,0.606453,0.602111,0.597743,0.593349,0.588928,0.584481,0.580007,0.575507,0.570979,0.566424,0.561842,0.557233,0.552596,0.547932,0.543239,0.538519,0.533771,0.528994,0.524189,0.519355,0.514493,0.509602,0.504682,0.499733,0.494754,0.489746,0.484708,0.47964,0.474543,0.469415,0.464257,0.459068,0.453849,0.448599,0.443318,0.438005,0.432661,0.427286,0.421879,0.41644,0.410969,0.405465,0.399929,0.394361,0.388759,0.383125,0.377457,0.371755,0.36602,0.360252,0.354449,0.348611,0.34274,0.336833,0.330892,0.324915,0.318903,0.312856,0.306773,0.300654,0.294498,0.288306,0.282078,0.275812,0.26951,0.26317,0.256792,0.250377,0.243924,0.237432,0.230902,0.224332,0.217724,0.211077,0.20439,0.197663,0.190896,0.184089,0.177241,0.170352,0.163422,0.156451,0.149438,0.142384,0.135287,0.128147,0.120965,0.11374,0.106471,0.0991593,0.0918034,0.0844032,0.0769586,0.0694692,0.0619347,0.0543549,0.0467294,0.0390579,0.0313401,0.0235758,0.0157645,0.00790603,6.41848e-17]



#%%

plt.plot(np.arange(0,9,1,dtype=float)/8,s8,'.',label='$T_0=8$')
plt.plot(np.arange(0,17,1,dtype=float)/16,s16,'.',label='$T_0=16$')
plt.plot(np.arange(0,33,1,dtype=float)/32,s32,'.',label='$T_0=32$')
plt.plot(np.arange(0,65,1,dtype=float)/64,s64,'.',label='$T_0=64$')
plt.plot(np.arange(0,129,1,dtype=float)/128,s128,'.',label='$T_0=128$')
plt.plot(np.arange(0,257,1,dtype=float)/256,s256,'-',label='$T_0=256$')
plt.grid()
plt.legend()
plt.xlabel('Eigenvalue index normalized')
plt.ylabel('Eigenvalue module')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '$N_0= 256$'

# place a text box in upper left in axes coords
plt.text(0,0.6, textstr, fontsize=14, bbox=props)


plt.title('Spectrums of the tranfer matrices for different $T_0$')
#plt.savefig('Spectrums.pdf')
plt.show()



#%% 
list_tau=np.array([-1/np.log(s8[1]),-1/np.log(s10[1]),-1/np.log(s16[1]),-1/np.log(s24[1]),-1/np.log(s32[1]),-1/np.log(s48[1]),-1/np.log(s64[1]),-1/np.log(s75[1]),-1/np.log(s128[1]),-1/np.log(s256[1])])
plt.loglog([8,10,16,24,32,48,64,75,128,256],list_tau,'b.')
plt.xlabel('Number of quenched spins $T_0$')
plt.ylabel('$ \\tau_1=- \\frac{1}{ \\log( \\lambda_1)}  $') 
plt.grid()
plt.title('Log-log plot of the time constant value with system size')

#plt.savefig('loglogplot_notlinked.pdf')
plt.show()



#%%
from sklearn.linear_model import LinearRegression
X=np.log(np.array([8,10,16,24,32,48,64,75,128,256])).reshape(-1,1)
y=np.log(list_tau)
reg = LinearRegression().fit(X, y)

#%% Generate the list way more easily 



list_lambda=np.zeros(N0-1) #On part de T0=1 et non T0=0
for T0 in range(1,N0):
    K=np.zeros((2*T0+1,2*T0+1))
    for M in range(T0):  # We stop 
        for i in range(M+1): #len(data(M)) = M+1
            #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
            K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
            K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2
    
    
    
    S=np.zeros((2*T0+1,2*T0+1))
    for i in range (-T0,T0+1):
        if i!=T0 : #Otherwise, bad index in the edges of the matrix
            S[T0+i][T0+i+1]=(i+1+T0)/(2*T0)
        if i!=-T0 :
            S[T0+i][T0+i-1]=1-((i-1+T0)/(2*T0)) #to be checked

    W=np.dot(K,S)
    Wprime=np.zeros((T0+1,T0+1))

    for i in range(T0+1):
        for j in range(T0+1):
            Wprime[i][j]=W[2*i][2*j]
    list_lambda[T0-1]=eigs(Wprime,k=2,which='LM')[0][1]
    
# there are some problems bc we sometime get complex values, you have to check it manually
# Just redoing with the wrong T0 solves the pbm

 #%% plotting 
x=np.arange(2,N0+1)
y=(-1/(np.log(list_lambda)))

plt.loglog(x,y,'b.')
plt.loglog(x,x**2,'r-')
plt.grid()
plt.title('Relaxation time vs system size')
plt.xlabel('Value of $T_0$') 
plt.ylabel('Value of $\\tau $')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '$N_0= %i$'%N0

# place a text box in upper left in axes coords
plt.text(3,1e3, textstr, fontsize=14, bbox=props)
#plt.savefig('Relax_vs_size_2p10.pdf')
plt.show()
#%%Asymptote

logx=np.log(x)
logy=np.log(y)

plt.plot(logx,logy,'b.',label='data')
plt.plot(logx, 2*logx - (2*logx[-1]-logy[-1]),'r',label='$y=2x$')
plt.grid()

plt.xlabel('Value of log($T_0$)') 
plt.ylabel('Value of log($\\tau $)')
plt.legend()
plt.title('Asymptotic study : $\\tau \\propto T_0^2$')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '$N_0= %i$'%N0

# place a text box in upper left in axes coords
plt.text(2,0, textstr, fontsize=14, bbox=props)

plt.savefig('Asymptotic2p5.pdf')
plt.show()



#%% Writting in a file, again

x=np.arange(2,N0+1)

f=open("list_data_2p10.txt",'w')
for i in range(N0-1):
    plip=str(np.log(x[i]))
    plop=str(np.log(-1/np.log(list_lambda[i])))
    f.write('{')
    f.write(plip)
    f.write(',')
    f.write(plop)
    f.write('}')
    if i!=N0-2 :
        f.write(',')
f.close()

#%%

test=np.real(eigs(Wprime,k=1,which='LM')[1])


#%%  Adjoint problem


T0=128

K=np.zeros((2*T0+1,2*T0+1))
M=T0-1
for i in range(M+1): #len(data(M)) = M+1
    #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
    K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
    K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2



S=np.zeros((2*T0+1,2*T0+1))
for i in range (-T0,T0+1):
    if i!=T0 : #Otherwise, bad index in the edges of the matrix
        S[T0+i][T0+i+1]=(i+1+T0)/(2*T0)
    if i!=-T0 :
        S[T0+i][T0+i-1]=1-((i-1+T0)/(2*T0)) #to be checked

W=np.dot(K,S)
Wprime=np.zeros((T0+1,T0+1))

for i in range(T0+1):
    for j in range(T0+1):
        Wprime[i][j]=W[2*i][2*j]

PstT0=np.real(eigs(Wprime,k=1,which='LM')[1])
PstT0*=1/np.sum(PstT0)




K=np.zeros((2*T0+1,2*T0+1))
M=T0-1
for i in range(M+1): #len(data(M)) = M+1
    #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
    K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
    K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2



S=np.zeros((2*T0+1,2*T0+1))
for i in range (-T0,T0+1):
    if i!=T0 : #Otherwise, bad index in the edges of the matrix
        S[T0+i][T0+i+1]=(i+1+T0)/(2*T0)
    if i!=-T0 :
        S[T0+i][T0+i-1]=1-((i-1+T0)/(2*T0)) #to be checked

W=np.dot(K,S)
Wprime=np.zeros((T0+1,T0+1))

for i in range(T0+1):
    for j in range(T0+1):
        Wprime[i][j]=W[2*i][2*j]


PstT0_1=np.real(eigs(Wprime,k=1,which='LM')[1])
PstT0_1*=1/np.sum(PstT0_1)

T0-=1
PstT0l=np.zeros(2*T0+1)
for i in range(T0+1):
    PstT0l[2*i]=PstT0[i]

Pst_prime=np.dot(K,PstT0l)

plt.plot(Pst_prime[1::2]-PstT0_1)


#%% Bimodality study 

#You need to manually input the data before

runcell('Computing the transfer matrix') #This gives the list_Pf thing (for the PQ)


indices=np.zeros(len(list_Pf))
                 
for i in range(len(list_Pf)) :
    indices[i]=np.abs(np.argmax(list_Pf[i][(i+1)%2::2])+i%2) #probleme d'indice sur le data avec les valeurs de j=1.2
    
indices-=T0//2
indices=np.abs(indices)
plt.plot(indices)



#%% doing the same for the sequencial process

#In order : data, transfer matrix (for all T<T0=N0), compute the first EiV,compute the max
indices=np.zeros(N0)
for T0 in range(N0):
      # We stop 
    K=np.zeros((2*T0+1,2*T0+1))
    M=T0-1    
    for i in range(M+1): #len(data(M)) = M+1
        #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
        K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
        K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2
        
        

    S=np.zeros((2*T0+1,2*T0+1))
    for i in range (-T0,T0+1):
        if i!=T0 : #Otherwise, bad index in the edges of the matrix
            S[T0+i][T0+i+1]=(i+1+T0)/(2*T0)
        if i!=-T0 :
            S[T0+i][T0+i-1]=1-((i-1+T0)/(2*T0)) #to be checked

    W=np.dot(K,S)
    Wprime=np.zeros((T0+1,T0+1))

    for i in range(T0+1):
        for j in range(T0+1):
            Wprime[i][j]=W[2*i][2*j]
            
    indices[T0]=np.abs(np.argmax(np.abs(np.real(eigs(Wprime,k=1,which='LM')[1])))-T0//2)
    

x=np.arange(0,N0,1)
mp8= 0.16405225626285802 #value of mes at the edge 
def Mstar(m,T0):
    return m*(T0+1-0.095*N0//2)-1
#plt.plot(x,Mstar(mp8,x))
plt.plot(indices)

#%% Plotting after renaming
#This is for different N0


plt.plot(np.linspace(0,1,1024),indices2p10,'b-',label='$N_0=1024$')
plt.plot(np.linspace(0,1,256),indices2p8,'r-',label='$N_0=256$')
plt.plot(np.linspace(0,1,32),indices2p5,'g-',label='$N_0=32$')
plt.grid()
plt.xlabel('Ratio $T_0/N_0$') 
plt.ylabel('Most probable magnetisation $M*$')
plt.legend()
plt.title('Bimodality emergence')

#plt.savefig('Bim_emergence_3.pdf')
plt.show()


#%% Plotting for same N0 and different T0 : is that useless ? 

plt.plot(np.linspace(0,1,1024),indices2p10,'b-',label='$N_0=1024$')
plt.plot(np.linspace(0,1,512),indices2p10[0:512],'r-',label='$N_0=1024$')
plt.plot(np.linspace(0,1,256),indices2p10[0:256],'g-',label='$N_0=1024$')



#%% Study of the energy part Vs entropy, trying to have a nice plot

Pst_test=np.ones(T0+1) #for a given T0
En_test=np.ones(T0+1)
Entropy_test=np.zeros(T0+1)
for M in range(1,T0): #the i's are basically the values of M
    prod1=1
    prod2=1
    for T in range(M):#k  must take into account up to 
        prod1 *= (1+data[T0-1][T][2])
        prod2 *= (1-data[T0-1][T][2])
    Pst_test[M] = math.comb(T0,M)*prod1/prod2
    Entropy_test[M]=math.comb(T0,M)
    En_test[M] = Pst_test[M]/Entropy_test[M]
Pst_test*=1/np.sum(Pst_test)  
En_test*=1/np.sum(En_test)  
Entropy_test*=1/np.sum(Entropy_test)  

vals=np.linspace(-T0,T0, T0+1)
plt.plot(vals,Pst_test)
plt.plot(vals,En_test)
plt.plot(vals,Entropy_test)
plt.grid()
plt.show()

#%% Analytical adjoint distribution 

Qst_test=np.zeros(T0+1)
for M in range((T0+1)): #the i's are basically the values of M
    if T0==N0 :
        print("You've reached the limit")
        break
    prod1=1
    prod2=1
    for T in range(M):#k  must take into account up to 
        prod1 *= (1+data[T0][T][2])
        prod2 *= (1-data[T0][T+1][2])
    Qst_test[M] = math.comb(T0,M)*prod1/prod2
    
Qst_test*=1/np.sum(Qst_test)  
plt.plot(Qst_test)

 #%% Trying to guess where the max is goinf to be
 
a=np.zeros(T0//2 -1)
b=np.zeros(T0//2 -1)
 


for i in range(T0//2 -1):
    a[i]=data[T0][T0//2+i+1][2]-data[T0][T0//2+i][2]
    b[i]=2/(T0+1)

print(2*np.argmin(np.abs(a-b)))

plt.plot(a,'b.')
plt.plot(b)

#%% systematic for all T0 ? 
peaks=np.zeros(N0)
for T0 in range(4,N0):
    a=np.zeros(T0//2 -1)
    b=np.zeros(T0//2 -1)
     
    for i in range(T0//2 -1):
        a[i]=data[T0][T0//2+i+1][2]-data[T0][T0//2+i][2]
        b[i]=2/(T0+1)
    
    peaks[T0]=2*np.argmin(np.abs(a-b))

plt.plot(peaks)
#Conclusion is that there is a problem for large T0 : the slope is not good
#Also the threashold is not valid.

#%% Peak analysis for the theoretical solutions P and Q

#Compute the distribution, then locate the maximum
peaks_st=np.zeros(N0)
for T0 in range(N0):
    Pst=np.zeros(T0+1)
    for i in range((T0+1)): #the i's are basically the values of M
        prod1=1
        prod2=1
        for k in range(i-1):#k  must take into account up to 
            prod1 *= (1+data[T0-1][k+1][2])
            prod2 *= (1-data[T0-1][k+1][2])
        Pst[i] = math.comb(T0,i)* prod1/prod2
    
    peaks_st[T0]=np.abs(np.argmax(Pst)-T0//2)

x=np.linspace(0,1,N0)

plt.plot(x,peaks_st)

#%% Let's make all thje plots
runcell('Loading the data || NEEDS TO BE FOR THE RIGHT N0 WITH THE RIGHT PATH ||', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Peak analysis for the theoretical solutions P and Q', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
peaks2p8=peaks_st

runcell('Same for 2p5', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Peak analysis for the theoretical solutions P and Q', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
peaks2p5=peaks_st

runcell('Same for 2p6', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Peak analysis for the theoretical solutions P and Q', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
peaks2p6=peaks_st
runcell('Same for 2p7', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Peak analysis for the theoretical solutions P and Q', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
peaks2p7=peaks_st
runcell('Same for 2p9', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Peak analysis for the theoretical solutions P and Q', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
peaks2p9=peaks_st
runcell('For N0=2p10=1024', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Peak analysis for the theoretical solutions P and Q', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
peaks2p10=peaks_st


plt.plot(np.linspace(0,1,2**5),peaks2p5/2**5,label='2p5')
plt.plot(np.linspace(0,1,2**6),peaks2p6/2**6,label='2p6')
plt.plot(np.linspace(0,1,2**7),peaks2p7/2**7,label='2p7')
plt.plot(np.linspace(0,1,2**8),peaks2p8/2**8,label='2p8')
plt.plot(np.linspace(0,1,2**9),peaks2p9/2**9,label='2p9')
plt.plot(np.linspace(0,1,2**10),peaks2p10/2**10,label='2p10')
plt.legend()
plt.grid()
plt.show()

#%% Testign the extremal paths for the PQ

Ppq=np.zeros(T0+1)
entrop=np.zeros(T0+1)
weights=np.zeros(T0+1)
for M in range(T0+1):
    weight=1
    entrop[M]=math.comb(T0,M)
    for T in range(T0-M):
        weight*=1-meq[T][N0-T]
    for T in range(T0-M,T0):
        weight*=1-meq[T][N0-((2*M-T0)-(T0-T))] #superduper janky 
    weights[M]=weight
    Ppq[M]=weight*entrop[M]

#plt.plot(np.linspace(-T0,T0,T0+1),Ppq)

plt.plot(np.linspace(-T0,T0,T0+1),-np.log(entrop)-np.min(-np.log(entrop)),'b-',label='Path-count entropy $S$')
plt.plot(np.linspace(-T0,T0,T0+1),np.log(weights)-np.min(np.log(weights)),'r-',label='Path-weight potential $-\\beta E$')
diff=(-np.log(entrop)-np.min(-np.log(entrop)))-(np.log(weights)-np.min(np.log(weights)))
plt.plot(np.linspace(-T0,T0,T0+1),-diff)
plt.xlabel('Magnetisation M')
#plt.yscale('log')
plt.ylabel('Relative contributions in probability')
plt.grid()
plt.legend()
plt.show()

#%% Let's try to make a nice pic


M=np.linspace(-T0,T0,T0+1)
S=-np.log(entrop)-np.min(-np.log(entrop))
E=np.log(weights)-np.min(np.log(weights))
diff=S-E


fig, ax1 = plt.subplots()
ax1.grid()
ax1.set_ylim(-19.25,175)

ax1.set_xlabel('Magnetisation $M$')
ax1.set_ylabel('Relative contributions in probability')

pS, =ax1.plot(M, S, color='b',label='Path-count entropy $S$')
pE, =ax1.plot(M,E, color='k',label='Path-weight potential $\\beta E$')

ax2 = ax1.twinx()
ax2.set_ylim(-0.22,2)
ax2.set_ylabel('Difference of the two contributions',color='r')
ax2.tick_params(axis='y', labelcolor='r')
pdiff, =ax2.plot(M[49:N0-48],diff[49:N0-48],color='r',label='Difference $S-\\beta E$')

P2=P[0::2]

ptrue, = ax2.plot(M[49:N0-48],-np.log(P2[49:N0-48])+np.log(P2[N0//2]),'r.')

axins = ax1.inset_axes([0.35,0.6, 0.3,0.3])
X=110
Y=115
axins.plot(M[X:Y],E[X:Y], '-k')
axins.plot(M[X:Y],S[X:Y],'b-')

ax1.indicate_inset_zoom(axins, edgecolor="black")

labels=['$S$','$\\beta E$','$S-\\beta E$','ME solution']


ax1.legend([pS,pE,pdiff,ptrue],labels,loc='lower left')



#fig.set_size_inches(10,7)

#plt.savefig('contributions_2.pdf')
plt.show()


#%% Making the plot for the signature

t=np.arange(0,6)
x=np.zeros(6)
traj=np.array([0,1,0,1,2,1])
traj2=np.array([0,1,2,1,2,1])
traj3=np.array([0,1,2,3,2,1])
fig, ax=plt.subplots()


ax.vlines(5,-5, 5 , color='r', linestyles='dotted')
#ax.set_ylim(5, -5)
ax.set_xlabel('$T$')
ax.set_ylabel('Magnetisation $M$')

#ax.set_yticks(np.arange(-5,6))
#ax.set_yticklabels(['-5','','-3','','-1','','1','','3','','5'])
#ax.set_yticklabels(['','-4','','-2','','','','2','','4',''])



ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.plot(t,t,'--k')
ax.plot(t,-t,'--k')

p3, = ax.plot(t,-traj3,color='blue',label='Modified twice')
p2, = ax.plot(t,-traj2,color='orange',label='Modified once')
p1, = ax.plot(t,-traj,color='limegreen',label='Initial trajectory')

labels=['Initial trajectory','Modified once','Modified twice']

ax.set_yticks(np.arange(-5,6))
#ax.set_yticklabels(['-5','','-3','','-1','','1','','3','','5'])
ax.set_xticks(np.arange(0,6))
ax.set_xticklabels(['','1','2','3','4','5'])
ax.set_yticklabels(['-5','-4','-3','-2','-1','','1','2','3','4','5'])

fig.text(0.09,0.46,'0')

fig.legend([p1,p2,p3],labels,loc='upper left',  bbox_to_anchor=(0.15, 0.90))
#plt.title('Trajectory with a (+,+,+,-,-) signature')
plt.tight_layout()


#plt.savefig('Correction.pdf')
plt.show()

#%% Computing the Kr Sr distributions







#%%  Making cool plots for the diffrence between PQ and stable distributions

# =============================================================================
# The plan : 
# Loading data for increasing N0
# Compute PQ and Pst 
# Compute the difference, and take the maximum
# Then plot the results in the coolest way
# =============================================================================
listN0=[2**4,2**5,2**6,2**7,2**8,2**9,2**10]

diff=[]


runcell('Another series of data 2p4', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')

runcell('Computing the transfer matrices and get the PQ', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Making the matrix for the random pick (easier to write)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Removing half of the lines and column (namely odd indicies)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
x=eigs(Wprime,k=1,which='LM')[1] #x is nom the Eigenvalue distribution
x=np.abs(x)
x/=np.sum(x)
y=np.zeros(N0+1)
for i in range(N0+1) : y[i]=x[i][0]
diff.append(np.max(np.abs(P[0::2]-y)))

runcell('Same for 2p5', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Computing the transfer matrices and get the PQ', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Making the matrix for the random pick (easier to write)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Removing half of the lines and column (namely odd indicies)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
x=eigs(Wprime,k=1,which='LM')[1] #x is nom the Eigenvalue distribution
x=np.abs(x)
x/=np.sum(x)
y=np.zeros(N0+1)
for i in range(N0+1) : y[i]=x[i][0]
diff.append(np.max(np.abs(P[0::2]-y)))

runcell('Same for 2p6', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Computing the transfer matrices and get the PQ', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Making the matrix for the random pick (easier to write)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Removing half of the lines and column (namely odd indicies)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
x=eigs(Wprime,k=1,which='LM')[1] #x is nom the Eigenvalue distribution
x=np.abs(x)
x/=np.sum(x)
y=np.zeros(N0+1)
for i in range(N0+1) : y[i]=x[i][0]
diff.append(np.max(np.abs(P[0::2]-y)))


runcell('Same for 2p7', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Computing the transfer matrices and get the PQ', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Making the matrix for the random pick (easier to write)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Removing half of the lines and column (namely odd indicies)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
x=eigs(Wprime,k=1,which='LM')[1] #x is nom the Eigenvalue distribution
x=np.abs(x)
x/=np.sum(x)
y=np.zeros(N0+1)
for i in range(N0+1) : y[i]=x[i][0]
diff.append(np.max(np.abs(P[0::2]-y)))

runcell('Loading the data || NEEDS TO BE FOR THE RIGHT N0 WITH THE RIGHT PATH ||', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Computing the transfer matrices and get the PQ', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Making the matrix for the random pick (easier to write)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Removing half of the lines and column (namely odd indicies)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
x=eigs(Wprime,k=1,which='LM')[1] #x is nom the Eigenvalue distribution
x=np.abs(x)
x/=np.sum(x)
y=np.zeros(N0+1)
for i in range(N0+1) : y[i]=x[i][0]
diff.append(np.max(np.abs(P[0::2]-y)))

runcell('Same for 2p9', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Computing the transfer matrices and get the PQ', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Making the matrix for the random pick (easier to write)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Removing half of the lines and column (namely odd indicies)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
x=eigs(Wprime,k=1,which='LM')[1] #x is nom the Eigenvalue distribution
x=np.abs(x)
x/=np.sum(x)
y=np.zeros(N0+1)
for i in range(N0+1) : y[i]=x[i][0]
diff.append(np.max(np.abs(P[0::2]-y)))

runcell('For N0=2p10=1024', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Computing the transfer matrices and get the PQ', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Making the matrix for the random pick (easier to write)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
runcell('Removing half of the lines and column (namely odd indicies)', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
x=eigs(Wprime,k=1,which='LM')[1] #x is nom the Eigenvalue distribution
x=np.abs(x)
x/=np.sum(x)
y=np.zeros(N0+1)
for i in range(N0+1) : y[i]=x[i][0]
diff.append(np.max(np.abs(P[0::2]-y))/np.max(y))


diff=np.asarray(diff)

#%%

fig, ax1 = plt.subplots()

ax1.set_xlabel('System size $N_0$')
ax1.set_ylabel('Relative difference between $\\vec{P}^{st}$ and $\\vec{P}^{PQ}$ ')
ax1.grid(b=True,which='both')
ax1.loglog(listN0,diff,'ro')
plt.title('Log-log plot of the error between PQ and the stationnary distribution')
plt.tight_layout()

#plt.savefig('loglogdiff.pdf')
plt.show()


#%%
# =============================================================================
# The plan is :
    # Compute the product of R matrices S then K
    # Compute S^R K^R on a certain P 
# =============================================================================

R=100 #Needs to be between 1 and T0
N_iter=10000

def makeS(T0):
    S=np.zeros((2*N0+1,2*N0+1))
    for i in range (-T0,T0+1):
        if i!=T0 : #Otherwise, bad index in the edges of the matrix
            S[N0+i][N0+i+1]=(i+1+T0)/(2*T0)
        if i!=-T0 :
            S[N0+i][N0+i-1]=1-((i-1+T0)/(2*T0)) #to be checked
    return(S)

    
def makeK(T0):
    K=np.zeros((2*N0+1,2*N0+1))    
    M=T0
    for i in range(T0): #len(data(M)) = M+1
        #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
        K[N0+data[M][i][1]+1][N0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
        K[N0+data[M][i][1]-1][N0+data[M][i][1]]=(1 - data[M][i][2])/2
    return(K)
#%%
R= 100 #Needs to be between 1 and T0
N_iter= 1

KR=makeK(N0-R-1)
SR=makeS(N0)
for i in range (1,R+1):
    SR=np.dot(makeS(N0-i),SR)
for j in range(R,0,-1):
    KR=np.dot(makeK(N0-j),KR)
    
# for M in range (T0-R,T0):  # We stop 
#     K=np.zeros((2*T0+1,2*T0+1))       
#     for i in range(M+1): #len(data(M)) = M+1
#         #To be clear = data[T][M] = a list that contains : [T,M,meq(T,M)]
#         K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
#         K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2
#     if (M==T0-R) :
#         K1=K
#     elif (M==T0-R+1):
#         KR=np.dot(K,K1)
#     else :
#         KR=np.dot(K,KR)
    
    
        

# P0=np.zeros((N0 +1))
# P0[N0//2]=1

P0=np.ones((N0 +1))
P0/=N0+1

M=np.dot(KR,SR)
Mprime=np.zeros((N0+1,N0+1))

for i in range(N0+1):
    for j in range(N0+1):
        Mprime[i][j]=M[2*i][2*j]
for i in range(0,N_iter):
    P0=np.dot(Mprime,P0) 
    if (i%100==0):
        plt.plot(P0)
    
#plt.plot(P0)
#%%

Mprime=np.zeros((N0+1,N0+1))

for i in range(N0+1):
    for j in range(N0+1):
        Mprime[i][j]=M[2*i][2*j]
        
#%%

Pcontrol=eigs(Mprime,k=1,which='LM')
plt.plot(-Pcontrol[1])

#%%

list_eig=np.zeros(254)

for R in range(3,255):
    KR=makeK(N0-R-1)
    SR=makeS(N0)
    for i in range (1,R+1):
        SR=np.dot(makeS(N0-i),SR)
    for j in range(R,0,-1):
        KR=np.dot(makeK(N0-j),KR)
        
    M=np.dot(KR,SR)
    Mprime=np.zeros((N0+1,N0+1))
    
    for i in range(N0+1):
        for j in range(N0+1):
            Mprime[i][j]=M[2*i][2*j]
    list_eig[R-3]=np.abs(eigs(Mprime, k=1, which='LM')[0])
    
plt.plot(list_eig)


#%% Making plots for the different distributions 

plt.plot(Values_M,P[0::2])
plt.xlabel('Magnetisation $M$')
plt.ylabel('PQ distribution $P^{(PQ)}$')

plt.show()

#%%
Pst0=eigs(Wprime,k=1,which='LM')[1]

Pst=np.zeros(len(Pst0))

for i in range(len(Pst)):
    Pst[i]=np.abs(Pst0[i][0])
    
Pst/=np.sum(Pst)

plt.plot(Values_M,Pst)

plt.xlabel('Magnetisation $M$')
plt.ylabel('RQ distribution $P^{(RQ)}$')

plt.grid()

plt.show()

#%%

plt.plot(np.abs((Pst-P[0::2])))
plt.xlabel('Magnetisation $M$')
plt.ylabel('$| P^{(RQ)} - P^{(PQ)} |$')

plt.show()

#%%

plt.figure()
plt.subplot(3,1,1)

plt.plot(Values_M,P[0::2])
plt.ylabel(' $P^{(PQ)}$')

plt.grid()

plt.subplot(3,1,2)

plt.plot(Values_M,Pst)

plt.ylabel('$P^{(RQ)}$')

plt.grid()

plt.subplot(3,1,3)

plt.plot(Values_M,np.abs((Pst-P[0::2])))
plt.xlabel('Magnetisation $M$')
plt.ylabel('$| P^{(RQ)} - P^{(PQ)} |$')

plt.grid()


plt.tight_layout()


plt.show()


#%%
plt.figure()
plt.subplot(1,2,1)

plt.plot(Values_M,P[0::2])
plt.ylabel(' $P^{(PQ)}$')

plt.grid()

plt.subplot(1,2,2)

plt.plot(Values_M,Pst)

plt.ylabel('$P^{(RQ)}$')

plt.grid()


plt.tight_layout()

#plt.savefig('SameSame.pdf')

plt.show()

#%% Making the plot for the differences between martingale estimation and 
# The Master Equation solution

fig, ax1 =plt.subplots()

ax1.plot(Values_M,P[0::2],'b--',label='$P^{(PQ)}$')
ax1.set_xlabel('Magnetisation $M$')
ax1.set_ylabel('PQ distribution $P^{(PQ)}$')

ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(Values_M,np.abs((Pst-P[0::2])),'r', label='Diff')
ax2.set_ylabel('$| P^{(RQ)} - P^{(PQ)} |$')
ax2.legend(loc='upper right')

ax1.grid()
#plt.savefig('PandDiff.pdf')
plt.show()

#%% Trying another plot with insets

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, ax =plt.subplots()

p1, = ax.plot(Values_M,np.abs((Pst-P[0::2])),'r', label='Difference')
ax.set_xlabel('Magnetisation $M$')
ax.set_ylabel('$| P^{(RQ)} - P^{(PQ)} |$')

ax.set_ylim(0,4e-7)

ax1 = ax.twinx()
p2, = ax1.plot(Values_M,P[0::2],'b--',label='$P^{(PQ)}$')
ax1.set_xlabel('Magnetisation $M$')
ax1.set_ylabel('PQ distribution $P^{(PQ)}$')

axins1 = inset_axes(ax, width=1, height=0.8,loc='center right',borderpad=0.5)

axins1.tick_params(labelsize="small",labelleft=False,grid_alpha=0.5)

axins1.plot(Values_M,P[0::2],label='$P^{(PQ)}$')
axins1.legend(loc="lower center",fontsize="small")
axins1.grid()

axins2 = inset_axes(ax, width=1, height=0.8,loc='upper right',borderpad=0.5)

axins2.plot(Values_M,Pst,label='$P^{(RQ)}$')
axins2.tick_params(labelsize="small",labelleft=False ,labelbottom=False,grid_alpha=0.5)
axins2.grid()
axins2.legend(loc="lower center",fontsize="small")

ax1.set_ylim(0,0.0085)

ax.grid()


plt.tight_layout()

labels=['Difference','$P^{(PQ)}$']

fig.legend([p1,p2],labels,loc='upper left',  bbox_to_anchor=(0.12, 0.90))

#plt.savefig('Contributions_inset.pdf')

plt.show()
#%%

for T0 in [4,5,6,7,8,9,10]: 
    #The order is : compute K, compute W, write W in different files, job done
    runcell('Computing the transfer matrices and get the PQ', 'C:/Users/Chamo/Documents/GitHub/Progressive_Quenching/Quenching_Flow.py')
    runcell('Making the matrix for the random pick (easier to write)')
    W=np.dot(K,S)
    Wprime=np.zeros((T0+1,T0+1))

    for i in range(T0+1):
        for j in range(T0+1):
            Wprime[i][j]=W[2*i][2*j]
            
    Pst0=eigs(Wprime,k=1,which='LM')[1]

    Pst=np.zeros(len(Pst0))
    
    for i in range(len(Pst)):
        Pst[i]=np.abs(Pst0[i][0])
        
    Pst/=np.sum(Pst)
    
    plt.plot(P[0::2],'ro')
    plt.plot(Pst,'b*')
    plt.show()
    
#%% é sé bartiiii

list_T=np.arange(0,N0)
def Peaks8(T,N0):
    c=5.06
    v=0.933
    #alpha=np.sqrt(3*N0*(1+c*(N0**(1-v))))
    alpha=np.sqrt(3*N0*((1+(N0*0.03005))))
    # return (np.floor((((T+2)/N0)*alpha)+0.5))
    return((((T+1)/N0)*alpha))

def Peaks10(T,N0):
    c=5.06
    v=0.933
    #alpha=np.sqrt(3*N0*(1+c*(N0**(1-v))))
    alpha=np.sqrt(3*N0*((1+(N0*0.0082))))
    # return (np.floor((((T+2)/N0)*alpha)+0.5))
    return((((T+1)/N0)*alpha))

def Peaks_test(T,N0):
    if(N0=256): 
        j=1.03005
    if(N0=1024):
        j=1.0082
    alpha=(1/j)*np.sqrt(N0-1-(N0/j))

plt.plot(list_T,Peaks(list_T,N0))

#%%shifted
shift_T=np.zeros(N0)
# beg=19 #for N0=256
beg=np.int(np.floor(0.05 * N0))
for i in range(beg,N0):
    shift_T[i]=i-beg

plt.plot(list_T/N0, peaks_st,'b-',label='$M°(T)$ for $N_0=256$')
plt.plot(np.arange(0,2**10)/2**10, peaks_2p10,label='$M°(T)$ for $N_0=1024$')
plt.plot(list_T/N0, Peaks8(list_T,N0)/2,'r--' ,label='$\\alpha (N_0) \\frac{(T+1)}{N_0} $ (offseted) for $N_0=256$')
plt.plot(np.linspace(0, 2**10,2**10)/2**10,Peaks10(np.linspace(0, 2**10,2**10),2**10)/2,'--',
         label='$\\alpha (N_0) \\frac{(T+1)}{N_0} $ (offseted) for $N_0=1024$')
plt.xlabel('$T/N_0$')
plt.ylabel('Maximum of probability $M°(T)$')
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('Peaks.pdf')
plt.show()

#%%

