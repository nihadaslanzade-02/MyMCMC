# Algorithm 1.2: Generic MCMC
# General framework for Markov Chain Monte Carlo sampling
#
# Input:
#   target_log_density: log of target distribution (up to constant)
#   transition_kernel: function that proposes next state given current state
#   initial_value: starting value for the chain
#   n_iterations: number of MCMC iterations
#   burn_in: number of burn-in iterations to discard
#
# Output:
#   samples: matrix of MCMC samples (after burn-in)
#   acceptance_rate: proportion of accepted proposals
#   full_chain: complete chain including burn-in (optional)

generic_mcmc <- function(target_log_density, 
                         transition_kernel,
                         initial_value,
                         n_iterations,
                         burn_in = 0) {
  
  # Determine dimension
  d <- length(initial_value)
  
  # Initialize storage
  total_iterations <- n_iterations + burn_in
  chain <- matrix(nrow = total_iterations, ncol = d)
  accepted <- numeric(total_iterations)
  
  # Initialize chain
  current_state <- initial_value
  chain[1, ] <- current_state
  
  # Main MCMC loop
  for (t in 2:total_iterations) {
    # Propose next state using transition kernel
    proposal <- transition_kernel(current_state)
    
    # Extract proposed state and acceptance probability
    proposed_state <- proposal$state
    acceptance_prob <- proposal$acceptance_prob
    
    # Accept or reject
    u <- runif(1)
    if (u < acceptance_prob) {
      current_state <- proposed_state
      accepted[t] <- 1
    } else {
      accepted[t] <- 0
    }
    
    # Store current state
    chain[t, ] <- current_state
  }
  
  # Remove burn-in period
  if (burn_in > 0) {
    samples <- chain[(burn_in + 1):total_iterations, , drop = FALSE]
    acceptance_rate <- mean(accepted[(burn_in + 1):total_iterations])
  } else {
    samples <- chain
    acceptance_rate <- mean(accepted)
  }
  
  # Return results
  return(list(
    samples = samples,
    acceptance_rate = acceptance_rate,
    full_chain = chain,
    accepted = accepted
  ))
}

# Note: The transition_kernel function must return a list with:
#   - state: the proposed next state
#   - acceptance_prob: probability of accepting the proposal
# This design allows for various MCMC algorithms (Metropolis, Gibbs, etc.)
# to be implemented within this generic framework