\# MCMC Algorithm Implementations



This directory contains R pseudocode implementations for the thesis:

"Exploration of Adaptive Algorithms"



## Files

- `basic_monte_carlo.R`: Algorithm 1.1 - Basic Monte Carlo integration

- `generic_mcmc.R`: Algorithm 1.2 - Generic MCMC framework

- `metropolis_hastings.R`: Algorithms 1.3-1.4 - Metropolis-Hastings variants

- `hmc_leapfrog.R`: Algorithm 1.5 - Hamiltonian Monte Carlo



## Usage

These implementations prioritize clarity over efficiency. For production use,

consider established packages like:

- `mcmc` package for Metropolis-Hastings

- `rstan` or `cmdstanr` for HMC/NUTS



## Requirements

- R version 4.0.0 or higher

- No external dependencies (uses only base R)

