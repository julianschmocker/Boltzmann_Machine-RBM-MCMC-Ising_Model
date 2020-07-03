# BM-RBM-MCMC-Ising
Contains implementations of MCMC methods for the fully connected Ising Model (Curie-Weiss). Then, we also run numerical simulations for fully visible Boltzmann Machine, for Bernoulli-Bernoulli RBM and Bernoulli-Gaussian RBM. For this the Contrastive Divergence and the persistent Contrastive Divergence algorithm were implemented for BM and RBM.

## Ising Model
The function in EC.py computes the energy change if one spin is flipped. The files GIBBS.py and MH.py contains implementations of the Gibbs sampler and the Metropolis-Hastings algorithm for the Curie-Weiss model. The simulations are computed in the file Curie_Weiss.py. The results of the simulations are displayed in Ising-Model_Simulations.pdf.

## Fully Visible BM
