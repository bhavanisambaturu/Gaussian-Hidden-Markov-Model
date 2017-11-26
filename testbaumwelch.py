#Coded by Bhavani Sambaturu
import random as rand
import numpy as np

possibilites = ['A','B']
sequences = [['A','B','B','A'],['B','A','B']]
c = [10,20]
start = [0.85,0.15]
s = [[0.3,0.7],[0.1,0.9]]
em = [[0.4,0.6],[0.5,0.5]]

s1 = np.array(s)
start1 = np.array(start)
em1 = np.array(em)
c1 = np.array(c)

def generate_alpha(sequences,c,s,em):
    alphax = []
    for x in sequences:
        alpha = np.zeros((s1.shape[0],len(x)))
        iter1 = 0
        iter1 = iter1 + 1
        for i in range(0,s1.shape[0]):
            ind = 0
            
            for y in possibilites:
                if(y == x[0]):
                    break
                ind = ind + 1
            
            alpha[i,0] = start1[i] * em1[i,ind]

        for i in range(1,len(x)):
      
            ind = 0
            
            for y in possibilites:
                if(y == x[i]):
                    break
                ind = ind + 1

            for j in range(0,s1.shape[0]):
                #print(alpha[j,i-1])
                for k in range(0,s1.shape[0]):
                    #print(alpha[k,i-1])
                    #print(s1[k,j])
                    #print(em1[j,ind])
                    alpha[j,i] = alpha[j,i] + alpha[k,i-1] * s1[k,j] * em1[j,ind]
        
                
        #print(alpha)
        alphax.append(list(alpha))
        #if(iter1 > 0):
         #   break
        
    return alphax

def generate_beta(sequences,c1,s1,em1):
    betax = []
    niter = 0
    for x in sequences:
        niter = niter + 1
        beta = np.zeros((s1.shape[0],len(x)))
        
        for i in range(0,s1.shape[0]):
            beta[i,len(x)-1] = 1
      
        for i in range(len(x)-2,-1,-1):
            
            ind = 0
            
            for y in possibilites:
                if(y == x[i]):
                    break
                ind = ind + 1
            

            for j in range(0,s1.shape[0]):
                #print(beta[j,i+1])
                for k in range(0,s1.shape[0]):
                    #print(s1[j,k])
                    #print(em1[k,ind-1])
                    #print(beta[k,i+1])
                    beta[j,i] = beta[j,i] + s1[j,k] * em1[k,ind-1] * beta[k,i+1]

        
        betax.append(list(beta))
    return betax

def generate_gamma(sequences,alphax,betax,s1,em1):
    gammax = []
    niter = 0

    for x in sequences:
        
        alpha = np.array(alphax[niter])
        beta = np.array(betax[niter])
        gamma = np.zeros((s1.shape[0] * s1.shape[0],len(x) - 1))

        Prh = 0
        for i in range(0,alpha.shape[0]):
            Prh = Prh + alpha[i,len(x) - 1]

        for i in range(0,gamma.shape[1]):

            ind = 0
            
            for y in possibilites:
                if(y == x[i+1]):
                    break
                ind = ind + 1
                       
            for j in range(0,gamma.shape[0]):
                x1 = np.floor(j/s1.shape[0])
                y1 = j%s1.shape[0]

                a = alpha[x1,i]
                z1 = s1[x1,y1]
                o1 = em1[y1,ind]
                b = beta[y1,i+1]
                gamma[j,i] = (a * z1 * o1 * b)/Prh

        gammax.append(list(gamma))  
        niter = niter + 1
        
    return gammax

def generate_delta(sequences,s1,alphax):
    niter = 0
    deltax = []
    for x in sequences:
        alpha = np.array(alphax[niter])
        delta = np.zeros((alpha.shape[0] * alpha.shape[1],1))

        ind = 0
        alpha = alpha.T

        for i in range(alpha.shape[0]):
            scols = 0
            for j in range(alpha.shape[1]):
                scols = scols + alpha[i,j]

            for j in range(0,alpha.shape[1]):
                delta[ind] = alpha[i,j]/scols
                ind = ind + 1
        
        niter = niter + 1
        deltax.append(list(delta))
        
    return deltax

def recompute_initial_probs(sequences,deltax,start1,c1,s1):

    niter = 0
    init_probs = np.zeros((len(start1),1))
    
    for x in sequences:
        delta = np.array(deltax[niter])

        for i in range(0,s1.shape[0]):
            init_probs[i] = init_probs[i] + delta[i] * c1[niter]
        niter = niter + 1

    sumn = 0 
    for i in range(0,len(init_probs)):
        sumn = sumn + init_probs[i]

    for i in range(0,len(init_probs)):
        init_probs[i] = init_probs[i]/sumn
        
    return init_probs

def recompute_state_transition(sequences,gammax,s1,c1):
    niter = 0

    new_state_temp = np.zeros((s1.shape[0] * s1.shape[1],1))
    new_state_mat = np.zeros((s1.shape[0],s1.shape[1]))

    for x in sequences:
        gamma = np.array(gammax[niter])

        for i in range(0,gamma.shape[0]):
            for j in range(0,gamma.shape[1]):
                new_state_temp[i] = new_state_temp[i] + gamma[i,j] * c1[niter]

        niter = niter + 1

    ind = 0
    for i in range(0,new_state_mat.shape[0]):
        for j in range(0,new_state_mat.shape[1]):
            new_state_mat[i,j] = new_state_temp[ind]
            ind = ind + 1

    for i in range(0,new_state_mat.shape[0]):
        sumn = 0
        for j in range(0,new_state_mat.shape[1]):
            sumn = sumn + new_state_mat[i,j]

        for j in range(0,new_state_mat.shape[1]):
            new_state_mat[i,j] = new_state_mat[i,j]/sumn
    
    return new_state_mat

def recompute_emission_mat(sequences,deltax,em1,possibilites,c1):
    niter = 0

    new_emission_mat = np.zeros((em1.shape[0],em1.shape[1]))
    
    for x in sequences:

        delta = np.array(deltax[niter])
        delta1 = np.reshape(delta,(len(x),em1.shape[0]))
        
        for i in range(0,len(x)):
            ind = 0
            
            for y in possibilites:
                if(y == x[i]):
                    break
                ind = ind + 1

            for j in range(0,em1.shape[0]):
                new_emission_mat[j,ind] = new_emission_mat[j,ind] + delta1[i,j] * c1[niter]

        #print(new_emission_mat)
        niter = niter + 1
        
    for i in range(0,new_emission_mat.shape[0]):
        summn = 0

        for j in range(0,new_emission_mat.shape[1]):
            summn = summn + new_emission_mat[i,j]

        for j in range(0,new_emission_mat.shape[1]):
            new_emission_mat[i,j] = new_emission_mat[i,j]/summn
    
    return new_emission_mat

def likelihood(sequences,alphax,c1):

    niter = 0
    lle = 0
    
    for x in sequences:
        alpha = np.array(alphax[niter])
        summn = 0
        
        for i in range(0,alpha.shape[0]):
            summn = summn + alpha[i,alpha.shape[1] - 1]

        lle = lle + c1[niter] * np.log(summn)

        niter = niter + 1
    return lle

if __name__ == "__main__":
    alphax = generate_alpha(sequences,c1,s1,em1)
    
    lle = likelihood(sequences,alphax,c1)
    eps = 10 ^ -10
    prev_lle = 0
    diff_lle = 10 ^ 10
    
    betax = generate_beta(sequences,c1,s1,em1)
    gammax = generate_gamma(sequences,alphax,betax,s1,em1)
    deltax = generate_delta(sequences,s1,alphax)

    new_init_prob = recompute_initial_probs(sequences,deltax,start1,c1,s1)
    new_state_mat = recompute_state_transition(sequences,gammax,s1,c1) 
    new_emission_mat = recompute_emission_mat(sequences,deltax,em1,possibilites,c1)

    niter = 0
    eps = 3
    if(diff_lle < eps):
        print(niter)
        prev_lle = lle
        alphax = generate_alpha(sequences,new_init_prob,new_state_mat,new_emission_mat)
        lle = likelihood(sequences,alphax,new_init_prob)
        betax = generate_beta(sequences,new_init_prob,new_state_mat,new_emission_mat)
        gammax = generate_gamma(sequences,alphax,betax,new_state_mat,new_emission_mat)
        deltax = generate_delta(sequences,new_state_mat,alphax) 
        
        init_prob = recompute_initial_probs(sequences,deltax,start1,new_init_prob,new_state_mat)
        new_state_mat = recompute_state_transition(sequences,gammax,new_state_mat,new_init_prob) 
        new_emission_mat = recompute_emission_mat(sequences,deltax,new_emission_mat,possibilites,new_init_prob)
        diff_lle = np.abs(prev_lle - lle)
        niter = niter + 1

    
