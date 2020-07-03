def gibbs(N,beta,J,h,iterations):
    """
    Random scan Gibbs sampling algorithm for Curie-Weiss model

    Parameters:
    N: number of spins
    iterations: number of steps that the Markov chain takes
    beta, J, h : additional parameters

    Returns:
    total_mag: empirical magnetization at each step i,
    vector of length iterations
    sigma: configuration at final step
    """

    # random initial spin assignment
    sigma = np.random.choice([-1,1],N)

    # preallocate variable
    emp_mag = np.zeros(iterations)
    
    for i in range(iterations):
        # choose a random spin
        spin_index = np.random.randint(N)

        # draw a random uniform number between 0 and 1
        x = np.random.random()
        ec = energy_change(sigma,spin_index,J,h)

        prob_change = (1/2)*(1-np.tanh(beta*ec/2))

        if x < prob_change:
            sigma[spin_index] = sigma[spin_index]*-1

        emp_mag[i] = sum(sigma)/len(sigma)
    
    return(emp_mag, sigma)

