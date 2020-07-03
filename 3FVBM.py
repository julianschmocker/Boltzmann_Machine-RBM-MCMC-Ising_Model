def Contrastive_Divergence(Data, theta_0, k, iterations, beta):
    """
    Contrastive Divergence (CD-k) algorithm for binary FVBM with
    N = 3 nodes

    Parameters:
    Data: matrix of M datapoints. Shape: NxM
    theta_0: initial parameter estimate
    k: The number of steps that the MCMC chain will take
    iterations: number of times the update equation is applied
    beta: additional parameters

    Returns:
    theta: sequence of the parameters obtained by CD-k algorithm
    """
    
    M = Data.shape[1]
    theta = np.zeros((3,iterations))
    theta[:,0] = theta_0

    res_1 = np.zeros(3)

    res_1[0] = Data[0,:].dot(np.transpose(Data[1, :]))
    res_1[1] = Data[0, :].dot(np.transpose(Data[2, :]))
    res_1[2] = Data[1, :].dot(np.transpose(Data[2, :]))

    for i in range(1,iterations):

        res_2 = np.zeros(3)
        D1 = Data.copy()
        D2 = np.zeros((3,M))
        for j in range(M):
            D2[:,j] = MCMC_sample(D1[:,j], createW(theta[:,i-1]),
                                  beta, k)

        res_2[0] = D2[0, :].dot(np.transpose(D2[1, :]))
        res_2[1] = D2[0, :].dot(np.transpose(D2[2, :]))
        res_2[2] = D2[1, :].dot(np.transpose(D2[2, :]))
        theta[:, i] = theta[:,i-1] + 0.1*((1/M)*beta*res_1 -
                                          (1/M)*beta*res_2)
    return(theta)

