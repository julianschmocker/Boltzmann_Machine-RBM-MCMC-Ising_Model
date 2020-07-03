### Implementation of MCMC methods for the Curie-Weiss Model
import numpy as np
import random
import matplotlib.pyplot as plt


def energy_change(sigma,spin_index,J,h):
    """
    Computes the energy change if spin at
    position spin_index is flipped.

    Parameters:
    sigma: configuration
    spin_index: spin to be flipped
    J, h : additional parameters

    Returns:
    ec: energy change
    """
    N_spins = len(sigma)
    ec = 2*J/N_spins*sum(sigma[spin_index]*np.delete(sigma, spin_index))
    +2*h*sigma[spin_index]
    return(ec)


def metropolis(N,beta,J,h,iterations):
    """
    Metroplis-Hastings algorithm for Curie-Weiss model

    Parameters:
    N: number of spins
    iterations: number of steps that the Markov chain takes
    beta, J, h : additional parameters

    Returns:
    total_mag: empirical magnetization at each step i, vector of length iterations
    sigma: configuration at final step
    """

    # random initial spin assignement
    sigma = np.random.choice([-1,1],N)

    # preallocate variable
    emp_mag = np.zeros(iterations)
    
    for i in range(iterations):
        # choose a random spin
        spin_index = np.random.randint(N)

        # draw a random uniform number between 0 and 1
        x = np.random.random()
        ec = energy_change(sigma,spin_index,J,h)

        if x < np.exp(-beta*ec):
            sigma[spin_index] = sigma[spin_index]*-1

        emp_mag[i] = sum(sigma)/len(sigma)
    
    return(emp_mag, sigma)


def gibbs(N,beta,J,h,iterations):
    """
    Random scan Gibbs sampling algorithm for Curie-Weiss model

    Parameters:
    N: number of spins
    iterations: number of steps that the Markov chain takes
    beta, J, h : additional parameters

    Returns:
    total_mag: empirical magnetization at each step i, vector of length iterations
    sigma: configuration at final step
    """

    # random initial spin assignement
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


# set random seed
np.random.seed(2)
plt.style.use(['thesis'])

iteration = 200000
N_spins = 300
h = 0
J = 1


# create a chain with beta = 1.1
beta = 1.1
total_mag_met,sigma_met = metropolis(N_spins,beta,J,h,iteration)
total_mag_gibbs, sigma_gibbs = gibbs(N_spins,beta,J,h,iteration)

# plot the empirical magnetization
fig,ax = plt.subplots(figsize=(4,4))
ax.plot(total_mag_met, label = "Metropolis-Hastings", color='#0073C2FF', linewidth=0.6)
ax.plot(total_mag_gibbs, label='Gibbs sampling', color='#EFC000FF', linewidth=0.6)
ax.set(xlabel=r'iteration $t$')
ax.set(ylabel=r'empirical magnetization $m(t)$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          fancybox=True, shadow=True, ncol=5)
fig.savefig('figures/fig1.pdf')

# histogram
fig,ax = plt.subplots(figsize=(4,4))
ax.hist(total_mag_gibbs, bins = 20, color = "#003C67FF")
fig.savefig('figures/fig1_hist.pdf')


# create a chain with beta = 0.8
beta = 0.8
total_mag_met,sigma_met = metropolis(N_spins,beta,J,h,iteration)
total_mag_gibbs, sigma_gibbs = gibbs(N_spins,beta,J,h,iteration)

# plot of empirical magnetization
fig,ax = plt.subplots(figsize=(4,4))
ax.plot(total_mag_met, label = "Metropolis-Hastings", color='#0073C2FF', linewidth=0.7)
ax.plot(total_mag_gibbs, label='Gibbs sampling', color='#EFC000FF', linewidth=0.7)
ax.set(xlabel=r'iteration $t$')
ax.set(ylabel=r'empirical magnetization $m(t)$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          fancybox=True, shadow=True, ncol=5)
fig.savefig('figures/fig2.pdf')

# histogram
fig,ax = plt.subplots(figsize=(4,4))
ax.hist(total_mag_gibbs, bins = 20, color = "#003C67FF")
fig.savefig('figures/fig2_hist.pdf')


# create histogram empirical magnetization of uniform distribution
draw_unif = np.random.choice([-1,1],(N_spins, iteration))
mag_unif = np.mean(draw_unif, axis = 0)

fig,ax = plt.subplots(figsize=(4,4))
ax.hist(mag_unif, bins = 20, color = "#003C67FF")
fig.savefig('figures/fig3_hist.pdf')
