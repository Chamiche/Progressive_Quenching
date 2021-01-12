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
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = '/Users/CMoslonka/ffmpeg'


import time



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

#%% ATTENTION Data for j=1.2

file=open("meq8j12_good.txt","r") #fixing the structure to make it right.
data_string=file.read()
#We can probably use np.loadtext instead 
data = ast.literal_eval(data_string)


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

# Making some stats

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
    r'$N_{iter}=10^6$ points',
    r'$T_0= %i$' %T0))

# place a text box in upper left in axes coords
plt.text(1500,1000, textstr, fontsize=14, bbox=props)

plt.xlabel('$\\tau $')
plt.ylabel('$<M(t)M(t+\\tau )>_t$')
plt.title('Correlation functions of the two forgetting processes')
plt.savefig('Good_correlations.pdf')
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

Values_M=[]
for k in range(-T0,T0+1,2):
    Values_M.append(k)
    
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
        K[T0+data[M][i][1]+1][T0+data[M][i][1]]=(1 + data[M][i][2])/2  #ith line and i+1 column 
        K[T0+data[M][i][1]-1][T0+data[M][i][1]]=(1 - data[M][i][2])/2
    
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

Wprime=np.zeros((T0+1,T0+1))

for i in range(T0+1):
    for j in range(T0+1):
        Wprime[i][j]=W[2*i][2*j]
        
#%% Writting Wprime in a file
 
f=open("Reduced_transfer_matrix_j12.txt",'w')
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

#%% Ploting the Spectrum

plt.plot(EValues, '-b')
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




#%% Plotting the first eigenvector

plt.plot(Values_M[::2], EVectors[0], '-b')

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

np.dot(EVectors[3],EVectors_transposed[5])



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
line, = ax.plot(Values_M,list_Pf[0]) 
#plt.xlim(-1,1)
def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * N0)
    
def animate(i):
    line.set_data(Values_M,list_Pf[i]*(1/np.max(list_Pf[i]))) 
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

steps=np.arange(1,T0)

ani=animation.FuncAnimation(fig,animate,steps,interval=100)
ani.save("test_anim_proba2P8.mp4")
plt.show()


#%% Let this blank

x=np.linspace(-1, 1,200)
#%%
fig, ax = plt.subplot()

ax.plot(x,np.tanh(x))

#%%import matplotlib.pyplot as plt
plt.ioff()
plt.plot([1.6, 2.7])

