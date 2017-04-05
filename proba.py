
import numpy as np
import math
import scipy.special
import itertools
import matplotlib.pyplot as plt

#Computes the proability to draw at least one success among N total elements (and n success elements). We draw c elements withtout replacement.
#This is defined as the hypergeometric distribution : p(k>=1)
def p1(N,n,c):
    return 1-scipy.special.binom(N-n,c)/scipy.special.binom(N,c)

#Computes the p1 probabilities vector for case N:N-n+1 and n:1.
def opt_draw(N,n,c):
    n_vec=np.arange(n,0,-1)
    N_vec=np.arange(N,N-n,-1)
    p_opt=p1(N_vec,n_vec,c)
    return np.array(p_opt)

#Computes the probability to have diff non-optimal draws
def diff_draw(P_vect,diff):
    bindex=itertools.combinations(np.arange(0,P_vect.size),diff)
    P_vect_diff=np.zeros(scipy.special.binom(P_vect.size,diff))
    index=0
    for i in bindex:
        P_vect_temp=np.copy(P_vect)
        P_vect_temp[np.asarray(i)]=1-P_vect_temp[np.asarray(i)]
        P_vect_diff[index]=np.prod(P_vect_temp)
        index=index+1
    return sum(P_vect_diff)

#Computes the density for the diffs
def density_diff(N,n,c):
    Diff_v=np.arange(1,n)
    dens=np.zeros(n)
    P_opt=opt_draw(N,n,c)
    for i in Diff_v:
        dens[i]=diff_draw(P_opt,i)
    dens[0]=np.prod(P_opt)
    return dens

#Repartition for the diffs
def repartition_fun(dens):
    return np.cumsum(dens)

def plot_dens(dens):
    plt.plot(dens)
    plt.ylabel("Probability to have a difference diff")
    plt.xlabel("Difference diff from optimal draw")
    plt.title("Probability density for difference from optimal draw")
    plt.show()

def plot_rep(rep):
        plt.plot(rep)
        plt.ylabel("Probability to have a difference of at least diff")
        plt.xlabel("Difference diff from optimal draw")
        plt.title("Probability repartition for difference from optimal draw")
        plt.show()

#Computes Probability to have less than percent % errors in function of c (from 1 to max_c)
def prob_c(N,n,percent,max_c):
    c_v=np.arange(1,max_c+1)
    diff_max=math.floor(percent*n)
    p_cum_diff=np.zeros(max_c)
    for i in c_v:
        P_d=opt_draw(N,n,i)
        S_p=np.prod(P_d)
        for j in np.arange(1,diff_max+1):
            S_p=S_p+diff_draw(P_d,int(j))
        p_cum_diff[i-1]=S_p
    return p_cum_diff

def plot_prob_c(p_c,percent):
        plt.plot(np.arange(1,p_c.size+1),p_c)
        plt.ylabel("Probability to have less than "+str(100*percent)+" percent diff")
        plt.xlabel("Value of c")
        plt.title("Probability to have less than "+str(100*percent)+" percent diff for different c values")
        plt.show()

#Main :
N=100
n=20
c=20

dd=density_diff(N,n,c)
plot_dens(dd)
pc=prob_c(N,n,0.1,max_c=50)
plot_prob_c(pc,0.1)
