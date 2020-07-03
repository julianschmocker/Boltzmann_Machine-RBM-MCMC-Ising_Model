# Boltzmann Machine - RBM - MCMC - Ising Model
Contains implementations of MCMC methods for the fully connected Ising Model (Curie-Weiss). Then, we also run numerical simulations for fully visible Boltzmann Machine, for Bernoulli-Bernoulli RBM and Bernoulli-Gaussian RBM. For this the Contrastive Divergence and the persistent Contrastive Divergence algorithm were implemented for BM and RBM.

## Ising Model
The function in EC.py computes the energy change if one spin is flipped. The files GIBBS.py and MH.py contains implementations of the Gibbs sampler and the Metropolis-Hastings algorithm for the Curie-Weiss model. The simulations are computed in the file Curie_Weiss.py. The results of the simulations are displayed in Ising-Model_Simulations.pdf.

## Fully Visible BM
The file NFVBM.py contains the persistent CD algorithm for the fully visible Boltzmann Machine with N nodes. The number of nodes can be specified. 

## Restricted Boltzmann Machine
The CD-k algorithms was implemented for the RBM with two visible units and one hidden units in RBM.py. The (persistent) CD-k algorithm for a RBM (with bias term) with n visible and m hidden units and the corresponding MCMC algorithms can be found in the file nxm_RMB_Bias.py.

The results of the numerical simulations for FVBM and RMB are displayed in the file BM_RBM_Simulations.pdf.
