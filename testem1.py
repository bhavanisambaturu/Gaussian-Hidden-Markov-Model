import numpy as np

tiny = 10**(-6)

def start_em(data,k):
    mean_data = np.mean(data,0)
    #cov_data = np.cov(data.T)
    cov_data = np.zeros((data.shape[1],data.shape[1]))
    prob_classes = np.empty(k)
    prob_classes.fill(1/k)
    cov_classes = np.zeros((data.shape[1],data.shape[1],k))
    diff_data = np.zeros((data.shape[0],data.shape[1]))

    for i in range(0,data.shape[0]):
        diff_data[i,:] = data[i,:] - mean_data
    
    cov_data = np.matmul(diff_data.T,diff_data)
    cov_data = cov_data/data.shape[0]
    
    wic = np.zeros((data.shape[0],k))

    for i in range(0,data.shape[0]):
        for j in range(0,k):
            wic[i,j] = 1/data.shape[0]

    lambda1 = np.empty(k)
    sumwic = np.sum(np.sum(wic))

    for j in range(0,k):
        temp = 0
        
        for i in range(0,data.shape[0]):
            temp = temp + wic[i,j]

        lambda1[j] = temp/sumwic
            

    mean_classes = np.zeros((k,mean_data.shape[0]))
    
    #cov_classes[0,0,0]  = cov_data[0,0]
    #cov_classes[0,0,1]  = cov_data[0,1]
    #cov_classes[0,1,0] = cov_data[1,0]
    #cov_classes[0,1,1] = cov_data[1,1]

    #cov_classes[1,0,0]  = cov_data[0,0]
    #cov_classes[1,0,1]  = cov_data[0,1]
    #cov_classes[1,1,0] = cov_data[1,0]
    #cov_classes[1,1,1] = cov_data[1,1]


    for i in range(0,k):
        mean_classes[i,:] = mean_data
        cov_classes[i,:,:] = cov_data
        #cov_classes[0,0,0]  = cov_data[0,0]
        #cov_classes[0,1,0]  = cov_data[0,1]
        #cov_classes[1,0,0] = cov_data[1,0]
        #cov_classes[1,1,0] = cov_data[1,1]
        #for j in range(0,data.shape[1]):
         #   for l in range(0,data.shape[1]):
         #       print(cov_data[j,l])
                #cov_classes[j,l,i] = cov_data[j,l]
        
        
    return prob_classes,mean_classes,cov_classes, wic,lambda1

def update_mean(data,class_prob,prob,k,wic):
    mean = np.zeros((k,data.shape[1]))

    for j in range(0,k):
        sumwic = 0 
        for i in range(0,data.shape[0]):
            #mean[j,:] = mean[j,:] + data[i,:] * class_prob[i,j]
            mean[j,:] = mean[j,:] + wic[i,:] * data[i,:]
            sumwic = sumwic + wic[i,j]

        #mean[j,:] = mean[j,:]/(data.shape[0] * prob[j] + tiny)
        mean[j,:] = mean[j,:]/sumwic

    return mean

def update_cov(data,k,prob,mean,class_prob,wic):
    cov_classes = np.zeros((data.shape[1],data.shape[1],k))
    diff_data = np.zeros((data.shape[0],data.shape[1]))

    for j in range(0,k):
        sumwic = 0
        
        for i in range(0,data.shape[0]):
            #diff_data[i,:] = data[i,:] - mean[j,:]
            diff_data[i,:] = wic[i,:] * (data[i,:] - mean[j,:])
            sumwic = sumwic + wic[i,:]

        temp = np.matmul(diff_data.T,diff_data)
               
        cov_classes[j,:,:] = temp/sumwic
        #cov_classes[j,:,:] = temp/data.shape[0]

    return cov_classes

def update_class_prob(data,k,prob,mean,cov):
    npoints = data.shape[0]
    ndim = data.shape[1]

    cov_inv_classes = np.zeros((cov.shape[0],cov.shape[1],cov.shape[2]))

    for i in range(0,k):
        cov_inv_classes[i,:,:] = np.linalg.inv(cov[i,:,:])

    class_prob1 = np.empty((npoints,k))

    for i in range(0,k):
        diff_data = np.zeros((data.shape[0],data.shape[1]))

        denom =  np.sqrt(((2 * np.pi) ** k) * np.linalg.det(cov[:,:,i]))
        sum_prob = 0

        for j in range(0,len(data)):
            diff_data[j,:] = data[j,:] - mean[i,:]
            temp = np.matmul(diff_data[j,:],cov[:,:,i])
            temp = np.matmul(temp,diff_data[j,:].T)
            temp = np.exp(-0.5 * temp)
            class_prob1[j,i] = temp/(denom + tiny) 
            sum_prob = sum_prob + class_prob1[j,i] * prob[i]
        
        for j in range(0,len(data)):
            class_prob1[j,i] = (class_prob1[j,i] * prob[i])/sum_prob
              
    return class_prob1

def update_prob(n,k,class_prob):
    prob_classes = np.empty(k)

    for j in range(0,k):
        prob_classes[j] = 0.0;

        for i in range(0,n):
            prob_classes[j] = prob_classes[j] + class_prob[i,j]

        prob_classes[j] = prob_classes[j]/n

    return prob_classes

def compute_loglikelihood(cov_classes,data,mean_classes,k):
    loglikelihood = np.empty(data.shape[0])
    
    for j in range(0,k):
        diff_data = np.zeros((data.shape[0],data.shape[1]))
        covdata = cov_classes[j,:,:]
        cov_inv = np.linalg.inv(covdata)
        
        for i in range(data.shape[0]):
            diff_data[i,:] = data[i,:] - mean_classes[j,:]

        for x in range(data.shape[0]):
            t = np.matmul(diff_data[x,:],cov_inv)
            t = np.matmul(t,diff_data[x,:])
            t1 = np.log(np.abs(np.linalg.det(covdata))) + t
            t = t + k * np.log(2 * np.pi)
            
            loglikelihood[j] = loglikelihood[j] + t

    lle = np.sum(np.abs(loglikelihood))

    return lle
        

if __name__ == "__main__":

    geyserdata = np.genfromtxt('C:\Bhavani\SMAI\Projects\oldgeyser.txt',delimiter=',')
    numpoints = geyserdata.shape[0]
    numdim = geyserdata.shape[1]
    numclasses = 2
    prev_likelihood = 0
    eps = 10 ^ -6
    diff_lle = 10 ^ 4
    prev_lle = 0
    numiter = 0
    prob_classes,mean_classes,cov_classes,wic,lambda1 = start_em(geyserdata,numclasses)

    while(diff_lle > eps):
        numiter = numiter + 1
        class_prob = update_class_prob(geyserdata,numclasses,prob_classes,mean_classes,cov_classes)
        prob_classes = update_prob(geyserdata.shape[0],numclasses,class_prob)
        mean_classes = update_mean(geyserdata,class_prob,prob_classes,numclasses,wic)
        cov_classes = update_cov(geyserdata,numclasses,prob_classes,mean_classes,class_prob,wic)
        lle = compute_loglikelihood(cov_classes,geyserdata,mean_classes,numclasses)
        diff_lle = np.abs(lle - prev_lle)
        prev_lle = lle
        

            
    
    
    
    
