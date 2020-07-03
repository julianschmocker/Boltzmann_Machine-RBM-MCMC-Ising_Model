# RMB
import numpy as np
import matplotlib.pyplot as plt

def cond_s(sigma,W,b,beta, prior = "g"):

    m = W.shape[1]

    s = np.zeros(m)

    for i in range(m):
        ec = beta * (np.array([sigma]).dot(np.transpose(W[:, i]))+b[i])
        s[i] = np.random.normal(ec,1,1)

        if prior == "b":
            x = np.random.random()

            prob_change = (1/2)*(1-np.tanh(-ec))

            if x < prob_change:
                s[i] = 1.
            else:
                s[i] = -1.
    
    return(s)



def MCMC_sigma(sigma,W,a,b,beta,iterations):

    n = len(sigma)

    for i in range(iterations):

        s = cond_s(sigma, W,b, beta)

        for i in range(n):

            # draw a random uniform number between 0 and 1
            x = np.random.random()

            ec =  -beta*(np.array([W[i,:]]).dot(np.transpose([s]))+a[i])

            prob_change = (1/2)*(1-np.tanh(ec))

            if x < prob_change:
                sigma[i] = 1.
            else:
                sigma[i] = -1.

    return(sigma,s)




def Contrastive_Divergence(Data, W_0, a_0, b_0, k, iterations, beta, p = False):
    """
    Contrastive Divergence (CD-k) alorithm for binary FVBM with N = 3 nodes

    Parameters:
    Data: matrix of M datapoints. Shape: NxM
    k: The number of steps that the MCMC chain will take
    iterations: number of times the update equation is applied
    beta, J, h : additional parameters

    Returns:
    total_mag: empirical magnetization at each step i, vector of length iterations
    sigma: configuration at final step
    """

    n = Data.shape[0]
    m = W_0.shape[1]
    M = Data.shape[1]
    W = np.zeros((n,m,iterations))
    a = np.zeros((n,iterations))
    b = np.zeros((m,iterations))
    W[:,:,0] = W_0
    a[:,0] = a_0
    b[:,0] = b_0
    MM = int(M/10)
    Data_sample = np.zeros((n, MM))
    S_sample = np.zeros((m, MM))

    D1 = np.zeros((n, MM))
    S1 = np.zeros((m, MM))


    if p == True:
        D2 = Data.copy()

    for i in range(1,iterations):

        res_2 = np.zeros((n,m))
        res_1 = np.zeros((n,m))


        if p == False:
            D2 = Data.copy()

        kkk = 0
        for j in np.random.choice(M,MM, replace = False):
            D2[:, j],S_sample[:, kkk] = MCMC_sigma(sigma = D2[:,j],
                                                   W = W[:,:,i-1], a = a[:,i-1],
                                                   b = b[:,i-1],
                                                   beta = beta, iterations = k)
            Data_sample[:, kkk] = D2[:, j]

            D1[:,kkk] = Data[:,j]
            S1[:, kkk] = cond_s(Data[:, j],W = W[:,:,i-1],
                                                   b = b[:,i-1], beta = beta)
            kkk = kkk + 1

        for ii in range(n):
            for jj in range(m):
                res_2[ii,jj] = Data_sample[ii, :].dot(np.transpose(S_sample[jj, :]))
                res_1[ii,jj] = D1[ii, :].dot(np.transpose(S1[jj, :]))

        res_sigma_1 = np.mean(D1, axis = 1)
        res_sigma_2 = np.mean(Data_sample, axis = 1)

        res_s_1 = np.mean(S1, axis = 1)
        res_s_2 = np.mean(S_sample, axis = 1)


        W[:,:, i] = W[:,:,i-1] + 0.1*((1/MM)*beta*res_1 - (1/MM)*beta*res_2)
        a[:,i] = a[:,i-1] + 0.1*(beta*res_sigma_1 - beta*res_sigma_2)
        b[:, i] = b[:, i - 1] + 0.1 * (beta * res_s_1 - beta * res_s_2)
    return(W,a,b)


