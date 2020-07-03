def Contrastive_Divergence(Data, theta_0, k, iterations, beta):
    """
    Contrastive Divergence (CD-k) alorithm for binary RBM with
    n = 2 visible units and m = 1 hidden unit

    Parameters:
    Data: matrix of M datapoints. Shape: nxM
    theta_0: initial parameter estimate
    k: The number of steps that the MCMC chain will take
    iterations: number of times the update equation is applied
    beta, J, h : additional parameters

    Returns:
    total_mag: empirical magnetization at each step i, vector of
    length iterations
    sigma: configuration at final step
    """

    M = Data.shape[1]
    theta = np.zeros((2,iterations))
    theta[:,0] = theta_0

    for i in range(1,iterations):
        res_1 = np.zeros(2)
        res_2 = np.zeros(2)

        S1 = np.zeros((1,M))
        D2 = np.zeros((2,M))
        S2 = np.zeros((1,M))
        D1 = Data.copy()
        for j in range(M):
            D2[:,j], S2[:,j] = MCMC_sigma(D1[:,j], theta[:,i-1],
                                          beta, k)
            S1[:,j] = cond_s(Data[:,j], theta[:,i-1], beta)

        res_1[0] = Data[0, :].dot(np.transpose(S1))
        res_1[1] = Data[1, :].dot(np.transpose(S1))

        res_2[0] = D2[0, :].dot(np.transpose(S2))
        res_2[1] = D2[1, :].dot(np.transpose(S2))

        theta[:, i] = theta[:,i-1] + 0.1*((1/M)*beta*res_1 -
                                          (1/M)*beta*res_2)
    return(theta)

