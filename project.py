# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:16:19 2017

@author: MANISREE
"""

import numpy as np
'''
import random
import time
import os
import numpy as np
import scipy as sp
import pandas as pd
from tabulate import tabulate
import pprint
import math
'''

np.random.seed(42)

def gen_seq_hmm(em,tm,seq_len,init_prob):
    x=[np.random.choice(list(init_prob.keys()),p=list(init_prob.values()))]
    y=[np.random.choice(list(em[x[0]].keys()),p=list(em[x[0]].values()))]
    for i in range(1,seq_len):
        x.append(np.random.choice(list(tm[x[i-1]].keys()),p=list(tm[x[i-1]].values())))
        y.append(np.random.choice(list(em[x[i]].keys()),p=list(em[x[i]].values())))
    return(x,y)   
    
def log_sum(x,y):
    if x > y:
        return(x+np.log(1+np.exp(y-x)))
    else:
        return(y+np.log(1+np.exp(x-y)))

def log_fwd(seq,em,tm,init_prob):
    mat=np.zeros((len(tm),len(seq)))
    mat=np.array(list(map(tuple,mat.T)),[(s,'float') for s in tm.keys()])
    for i in init_prob.keys():
        mat[i][0]=init_prob[i] + em[i][seq[0]]
    for k in range(1,len(seq)):
        for m in tm.keys():
            mat[m][k] = mat[tm.keys()[0]][k-1] + tm[tm.keys()[0]][m]
            for n in tm.keys()[1:]:
                mat[m][k] = log_sum(mat[m][k],mat[n][k-1] + tm[n][m])
            mat[m][k] = mat[m][k] + em[m][seq[k]] 
    return(mat)

def log_bwd(seq,em,tm,init_prob):
    mat=np.zeros((len(tm),len(seq)))
    mat=np.array(list(map(tuple,mat.T)),[(s,'float') for s in tm.keys()])
    for k in reversed(range(0,len(seq)-1)):
        for m in tm.keys():
            mat[m][k] = mat[tm.keys()[0]][k+1] + tm[m][tm.keys()[0]] + em[tm.keys()[0]][seq[k+1]]
            for n in tm.keys()[1:]:
                mat[m][k] = log_sum(mat[m][k],mat[n][k+1] + tm[m][n] + em[n][seq[k+1]])
    return(mat)

def count_init_prob(count_init,init_prob,em,bwd,seq):
    for i in init_prob.keys():
        count_init[i] = init_prob[i] + em[i][seq[0]] + bwd[i][0]
    return(count_init)
    
def count_transitions(count_tm,init_prob,tm,em,fwd,bwd,seq):
    for i in tm.keys():
        for j in tm.keys():
            count_tm[i][j] = fwd[i][0] + tm[i][j] + em[j][seq[1]] + bwd[j][1]
            for k in range(1,len(seq)-1):
                v=fwd[i][k] + tm[i][j] + em[j][seq[k+1]] + bwd[j][k+1]
                count_tm[i][j] = log_sum(count_tm[i][j],v)
    return(count_tm)
    
'''    
def count_emissions(count_em,em,fwd,bwd,seq):
    for i in em.keys():
        count_em[i][seq[0]] = fwd[i][0] + bwd[i][0]
    for j in em.keys():
        for k in range(1,len(seq)):
            count_em[j][seq[k]] = log_sum(count_em[j][seq[k]],fwd[j][k] + bwd[j][k])
    return(count_em)            
'''
def count_emissions(count_em,em,fwd,bwd,seq):
    for i in count_em.keys():
        eliminate=[]
        for j in set(seq):
            count_em[i][j] = fwd[i][seq.index(j)] + bwd[i][seq.index(j)]
            eliminate.append(seq.index(j))            
    for k in count_em.keys():
        for n in np.delete(range(len(seq)),eliminate):
            count_em[k][seq[n]] = log_sum(count_em[k][seq[n]],fwd[k][n]+bwd[k][n])
    return(count_em)


def log_likelihood1(input_seqncs,ipl,tpl,epl):
    ll = 0
    for i in input_seqncs:
        fl=log_fwd(i,epl,tpl,ipl)
        seqprob = fl[ipl.keys()[0]][-1]
        for j in ipl.keys()[1:]:
            seqprob = log_sum(seqprob,fl[j][-1])
        ll += seqprob
    return(ll)


def baum_welch(input_sequences,theta_not,theta,theta_prime,threshold):
    dll=log_likelihood1(input_sequences,theta_not,theta,theta_prime)
    counter=0
    while(abs(dll) > threshold):
        old_ll=log_likelihood1(input_sequences,theta_not,theta,theta_prime)
        exp_ip={fs:0 for fs in theta_not.keys()}
        exp_tp={fs:{ts:0 for ts in theta.keys()} for fs in theta.keys()}
        exp_ep={fs:{obs:0 for obs in theta_prime[list(theta_prime.keys())[0]].keys()} for fs in theta.keys()}
        new_ip={fs:0 for fs in theta_not.keys()}
        new_tp={fs:{ts:0 for ts in theta.keys()} for fs in theta.keys()}
        new_ep={fs:{obs:0 for obs in theta_prime[list(theta_prime.keys())[0]].keys()} for fs in theta.keys()}
        counter = counter+1       
        for q in range(len(input_sequences)):
            seqnc=input_sequences[q]
            f=log_fwd(seqnc,theta_prime,theta,theta_not)
            b=log_bwd(seqnc,theta_prime,theta,theta_not)
            prb_seq = f[theta_not.keys()[0]][-1]
            for m in theta_not.keys()[1:]:
                prb_seq = log_sum(prb_seq,f[m][-1])
            exp_ip=count_init_prob(exp_ip,theta_not,theta_prime,b,seqnc)
            exp_tp=count_transitions(exp_tp,theta_not,theta,theta_prime,f,b,seqnc)
            exp_ep=count_emissions(exp_ep,theta_prime,f,b,seqnc)
            
            if q==0:
                new_ip={a:exp_ip[a]-prb_seq for a in exp_ip.keys()}
            else:
                new_ip={a:log_sum(new_ip[a],exp_ip[a]-prb_seq) for a in exp_ip.keys()}
            if q==0:
                new_tp={a:{b:exp_tp[a][b]-prb_seq for b in exp_tp[a].keys()} for a in exp_tp.keys()}
            else:
                new_tp={a:{b:log_sum(new_tp[a][b],exp_tp[a][b]-prb_seq) for b in exp_tp[a].keys()} for a in exp_tp.keys()}            
            if q==0:
                new_ep={a:{b:exp_ep[a][b]-prb_seq for b in exp_ep[a].keys()} for a in exp_ep.keys()}
            else:
                new_ep={a:{b:log_sum(new_ep[a][b],exp_ep[a][b]-prb_seq) for b in exp_ep[a].keys()} for a in exp_ep.keys()}            
            
            '''
            if q==0:
                new_ip={a:np.exp(exp_ip[a]-prb_seq) for a in exp_ip.keys()}
            else:
                new_ip={a:np.exp(log_sum(new_ip[a],exp_ip[a]-prb_seq)) for a in exp_ip.keys()}
            if q==0:
                new_tp={a:{b:np.exp(exp_tp[a][b]-prb_seq) for b in exp_tp[a].keys()} for a in exp_tp.keys()}
            else:
                new_tp={a:{b:np.exp(log_sum(new_tp[a][b],exp_tp[a][b]-prb_seq)) for b in exp_tp[a].keys()} for a in exp_tp.keys()}            
            if q==0:
                new_ep={a:{b:np.exp(exp_ep[a][b]-prb_seq) for b in exp_ep[a].keys()} for a in exp_ep.keys()}
            else:
                new_ep={a:{b:np.exp(log_sum(new_ep[a][b],exp_ep[a][b]-prb_seq)) for b in exp_ep[a].keys()} for a in exp_ep.keys()}            
            '''    
        init={a:np.exp(new_ip[a])/sum(np.exp(new_ip.values())) for a in new_ip.keys()}
        tp={a:{b:np.exp(new_tp[a][b])/sum(np.exp(new_tp[a].values())) for b in new_tp[a].keys()} for a in new_tp.keys()}
        ep=theta_not={a:{b:np.exp(new_ep[a][b])/sum(np.exp(new_ep[a].values())) for b in new_ep[a].keys()} for a in new_ep.keys()}
        theta_not={a:np.log(init[a]) for a in init.keys()}
        theta={a:{b:np.log(tp[a][b]) for b in tp[a].keys()} for a in tp.keys()}
        theta_prime={a:{b:np.log(ep[a][b]) for b in ep[a].keys()} for a in ep.keys()}
        new_ll=log_likelihood1(input_sequences,theta_not,theta,theta_prime)
        dll=new_ll-old_ll
        print("diff of log-likelihood, new-lilkelihood at iteration %d is %f and %f" %(counter,dll,new_ll))
    #return(theta_not,theta,theta_prime,new_ll)
    return(init,tp,ep,new_ll)







########### Testing
'''
'
theta_not={'s':0.85,'t':0.15}
theta={'s':{'s':0.3,'t':0.7},'t':{'s':0.1,'t':0.9}}
theta_prime={'s':{'A':0.4,'B':0.6},'t':{'A':0.5,'B':0.5}}
#sequences=sum([['ABBA']*10,['BAB']*20],[])
#sequences=[['ABBA'*10]*10,['BAB']*20]

sequences=[['ABBA'*10]]
l = []
map(l.extend, sequences)
sequences=l

sequences=[['ABBA']*10,['BAB']*20]
l = []
map(l.extend, sequences)
sequences=l

theta_not={i:np.log(j) for i,j in theta_not.items()}
theta={m:{n:np.log(p) for n,p in theta[m].items()} for m in theta.keys()}
theta_prime={m:{n:np.log(p) for n,p in theta_prime[m].items()} for m in theta_prime.keys()}

input_sequences=sequences
bw=baum_welch(input_sequences,theta_not,theta,theta_prime,threshold=1e-04)
'''
######## rolling dice
theta_not={'f':0.5,'l':0.5}
theta={'f':{'f':0.95,'l':0.05},'l':{'f':0.1,'l':0.9}}
theta_prime={'f':{'1':1./6,'2':1./6,'3':1./6,'4':1./6,'5':1./6,'6':1./6},\
'l':{'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.5}}

seq1=''.join(gen_seq_hmm(theta_prime,theta,30000,theta_not)[1])

theta_not={i:np.log(j) for i,j in theta_not.items()}
theta={m:{n:np.log(p) for n,p in theta[m].items()} for m in theta.keys()}
theta_prime={m:{n:np.log(p) for n,p in theta_prime[m].items()} for m in theta_prime.keys()}

input_sequences=[seq1]
bw=baum_welch(input_sequences,theta_not,theta,theta_prime,threshold=1e-04)


######## rolling dice with random parameters
theta_not={'f':0.5,'l':0.5}
theta={'f':{'f':0.95,'l':0.05},'l':{'f':0.1,'l':0.9}}
theta_prime={'f':{'1':1./6,'2':1./6,'3':1./6,'4':1./6,'5':1./6,'6':1./6},\
'l':{'1':0.1,'2':0.1,'3':0.1,'4':0.1,'5':0.1,'6':0.5}}

seq1=''.join(gen_seq_hmm(theta_prime,theta,30000,theta_not)[1]) ### generate the seq with the above HMM parameters

#### now initialize random parameters and try to re-estimate the above model parameters
  
theta_not=dict(zip(('f','l'),np.random.dirichlet(np.ones(2))))
theta={i:dict(zip(theta_not.keys(),np.random.dirichlet(np.ones(2)))) for i in theta_not.keys()}
theta_prime={i:dict(zip(('1','2','3','4','5','6'),np.random.dirichlet(np.ones(6)))) for i in theta_not.keys()}

theta_not={i:np.log(j) for i,j in theta_not.items()}
theta={m:{n:np.log(p) for n,p in theta[m].items()} for m in theta.keys()}
theta_prime={m:{n:np.log(p) for n,p in theta_prime[m].items()} for m in theta_prime.keys()}

input_sequences=[seq1]
bw=baum_welch(input_sequences,theta_not,theta,theta_prime,threshold=1e-03)

'''
There is no assurance for a global minimum in BW algorithm. So different random initialparameters behave
differently. Some end up with a global minimum, while some are trapped in local minima. Therefore, initial 
parameters paly a very important role. This check can be included in the initialization itself. If initial 
parameters encounter a local minima, immediately switch to a different set of initializations.
'''

        
        
        
