# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:22:18 2017

@author: student
"""

import numpy as np
import random
import time
import os
import numpy as np
import scipy as sp
import pandas as pd
from tabulate import tabulate
import pprint
import math

def forward(seq,em,tm,init_prob):
    mat=np.zeros((len(tm),len(seq)))
    for s in init_prob.keys():
        mat[:,0]=[np.round(init_prob[s]*em[s][seq[0]],decimals=2) for s in init_prob.keys()]
    for k in range(1,len(seq)):
        mat[:,k]=[np.round(np.inner(mat[:,k-1],tm_em(em,tm,seq[k],m)),5) for m in tm.keys()]
    #mat=np.array(map(tuple,mat.T),[(s,'float') for s in tm.keys()])
    mat=np.array(list(map(tuple,mat.T)),[(s,'float') for s in tm.keys()])
    return(mat)


def backward(seq,em,tm,init_prob):
    mat=np.zeros((len(tm),len(seq)))
    mat[:,-1]=1
    for k in reversed(range(0,len(seq)-1)):
        #mat[:,k]=[round(float(np.sum(tm[m].values()*em_obs(em,seq[k+1])*mat[:,k+1])),3) for m in tm.keys()]
        mat[:,k]=[np.round(np.sum(list(tm[m].values())*em_obs(em,seq[k+1])*mat[:,k+1]),5) for m in tm.keys()]
    #mat=np.array(map(tuple,mat.T),[(s,'float') for s in tm.keys()])   
    mat=np.array(list(map(tuple,mat.T)),[(s,'float') for s in tm.keys()])        
    return(mat)
            
def em_obs(em,obs):
    q=np.asarray([em[s][obs] for s in em.keys()])
    return(q)       
                            
def tm_em(em,tm,obs,state):
    q=[tm[n][state]*em[state][obs] for n in tm.keys()]
    return(np.asarray(q))                       
        

#seq=''.join(('r','w','b','b'))
#f=forward(seq,em,tm,init_prob)
#b=backward(seq,em,tm,init_prob)

def state_at_pos(state,position,f,b,seq):
    return(round(float((f[state][position]*b[state][position])/sum(list(f[-1]))),5))

def state_transn_at_pos(fs,ts,position,f,b,seq,tm,em):
    a=(f[fs][position]*b[ts][position+1]*tm[fs][ts]*em[ts][seq[position+1]])/sum(list(f[-1]))
    return(round(float(a),5))
    
    


def gen_seq_hmm(em,tm,seq_len,init_prob):
    x=[np.random.choice(list(init_prob.keys()),p=list(init_prob.values()))]
    y=[np.random.choice(list(em[x[0]].keys()),p=list(em[x[0]].values()))]
    for i in range(1,seq_len):
        x.append(np.random.choice(list(tm[x[i-1]].keys()),p=list(tm[x[i-1]].values())))
        y.append(np.random.choice(list(em[x[i]].keys()),p=list(em[x[i]].values())))
    return(x,y)        
    
    
    
def log_likelihood(list_of_seqns,init_prob,tmat,emat):
    ll=0
    uu=np.unique(list_of_seqns,return_counts=True)
    for i in range(len(uu)):
        ll += uu[1][i]*math.log(sum(forward(uu[0][i],emat,tmat,init_prob)[-1]))
    return(ll)        
           
  
def baum_welch(input_sequences,theta_not,theta,theta_prime,threshold):
    dll=log_likelihood(list_of_seqns=input_sequences,init_prob=theta_not,tmat=theta,emat=theta_prime)
    counter=0
    while(abs(dll) > threshold):
        old_ll=log_likelihood(list_of_seqns=input_sequences,init_prob=theta_not,tmat=theta,emat=theta_prime)
        a=np.unique(input_sequences,return_counts=True)  
        freq_seq=dict(zip(a[0],a[1]))
        exp_ip={fs:0 for fs in theta_not.keys()}
        exp_tm={fs:{ts:0 for ts in theta.keys()} for fs in theta.keys()}
        exp_em={fs:{obs:0 for obs in theta_prime[list(theta_prime.keys())[0]].keys()} for fs in theta.keys()}
        counter = counter+1       
        for seq in freq_seq.keys():
            f=forward(seq,theta_prime,theta,theta_not)
            b=backward(seq,theta_prime,theta,theta_not)
            for fs in theta_not.keys():
                exp_ip[fs] += (freq_seq[seq]*theta_not[fs]*theta_prime[fs][seq[0]]*b[fs][0])/sum(f[-1])
                for j in range(len(seq)):
                    exp_em[fs][seq[j]] += (f[fs][j]*b[fs][j]*freq_seq[seq])/sum(f[-1])
                for ts in theta_not.keys():
                    for position in range(len(seq)-1):
                        exp_tm[fs][ts] += (f[fs][position]*b[ts][position+1]*theta[fs][ts]*theta_prime[ts][seq[position+1]]*freq_seq[seq])/sum(f[-1])  
                            
        theta_not={state:exp_ip[state]/sum(list(exp_ip.values())) for state in theta.keys()}
        theta={fst:{tst:exp_tm[fst][tst]/sum(list(exp_tm[fst].values())) for tst in theta.keys()} for fst in theta.keys()}
        theta_prime={st:{sym:exp_em[st][sym]/sum(list(exp_em[st].values())) for sym in theta_prime[list(theta_prime.keys())[0]].keys()} for st in theta.keys()}
               
    #new_theta_not={state:exp_ip[state]/sum(list(exp_ip.values())) for state in theta.keys()}
    #new_theta={fst:{tst:exp_tm[fst][tst]/sum(list(exp_tm[fst].values())) for tst in theta.keys()} for fst in theta.keys()}
    #new_theta_prime={st:{sym:exp_em[st][sym]/sum(list(exp_em[st].values())) for sym in theta_prime[list(theta_prime.keys())[0]].keys()} for st in theta.keys()}
        new_ll=log_likelihood(list_of_seqns=input_sequences,init_prob=theta_not,tmat=theta,emat=theta_prime)
        dll=new_ll-old_ll
        print("diff of log-likelihood for iteration %d is %f" %(counter,dll))
    return(theta_not,theta,theta_prime,new_ll)

    

########### rolling dice
init_prob={'f':0.5,'l':0.5}
tm={'f':{'f':0.95,'l':0.05},'l':{'f':0.1,'l':0.9}}
em={'f':{'1':1./6,'2':1./6,'3':1./6,'4':1./6,'5':1./6,'6':1./6},\
'l':{'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.5}}



'''
init_prob={'s':0.85,'t':0.15}
tm={'s':{'s':0.3,'t':0.7},'t':{'s':0.1,'t':0.9}}
em={'s':{'A':0.4,'B':0.6},'t':{'A':0.5,'B':0.5}}
sequences=sum([['ABBA']*10,['BAB']*20],[])
seq=sequences[0]
f=forward(seq,em,tm,init_prob)
b=backward(seq,em,tm,init_prob)
gamma=[state_at_pos(state,position,f,b,seq) for state in tm.keys() for position in range(len(seq))]
gamma=np.array(list(map(tuple,np.asarray(gamma).reshape(2,4).T)),[(s,'float') for s in tm.keys()])

delta=[state_transn_at_pos(fs,ts,position,f,b,seq,tm,em) for fs in tm.keys() for ts in tm.keys() for position in range(len(seq)-1)]
delta={fs:{ts:{} for ts in tm.keys()} for fs in tm.keys()}
for fs in tm.keys():
    for ts in tm.keys():
        for position in range(len(seq)-1):
            delta[fs][ts][position]=state_transn_at_pos(fs,ts,position,f,b,seq,tm,em)
#delta=np.array(list(map(tuple,np.asarray(delta).reshape(3,4).T)),[(s,'float') for s in tm.keys()])
'''

theta_not={'s':0.85,'t':0.15}
theta={'s':{'s':0.3,'t':0.7},'t':{'s':0.1,'t':0.9}}
theta_prime={'s':{'A':0.4,'B':0.6},'t':{'A':0.5,'B':0.5}}
sequences=sum([['ABBA']*10,['BAB']*20],[])
bw=baum_welch(sequences,theta_not,theta,theta_prime,1e-04)
est_theta_not=bw[0]
est_theta=bw[1]
est_theta_prime=bw[2]




theta_not={'f':0.5,'l':0.5}
theta={'f':{'f':0.95,'l':0.05},'l':{'f':0.1,'l':0.9}}
theta_prime={'f':{'1':1./6,'2':1./6,'3':1./6,'4':1./6,'5':1./6,'6':1./6},\
'l':{'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.5}}
input_sequences=[''.join(gen_seq_hmm(theta_prime,theta,300,theta_not)[1]) for q in range(0,10)]

bw=baum_welch(input_sequences,theta_not,theta,theta_prime,1e-04)
est_theta_not=bw[0]
est_theta=bw[1]
est_theta_prime=bw[2]


        
    
        
    
    
    
