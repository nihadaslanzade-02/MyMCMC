# Algorithm 1.2: Generic MCMC
# General framework for Markov Chain Monte Carlo sampling
#
# Input:
#   target_log_density: log of target distribution (up to constant)
#   transition_kernel: function that proposes next state given current state
#   initial_value: starting value for the chain
#   n_iterations: TOTAL number of iterations to run, burn_in included
#   burn_in: number of leading iterations to discard, must be < n_iterations
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

  # n_iterations is the whole chain, burn_in comes out of it. The guard is
  # repeated verbatim in each algorithm file rather than factored out, so that
  # any one of them can still be sourced on its own.
  if (n_iterations < 2 || burn_in < 0 || burn_in >= n_iterations) {
    stop("need n_iterations >= 2 and 0 <= burn_in < n_iterations; got ",
         "n_iterations = ", n_iterations, ", burn_in = ", burn_in)
  }

  # Initialize storage
  chain <- matrix(nrow = n_iterations, ncol = d)
  accepted <- numeric(n_iterations)

  # Initialize chain
  current_state <- initial_value
  chain[1, ] <- current_state

  # Main MCMC loop
  for (t in 2:n_iterations) {
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

  # Remove burn-in period. Iteration 1 is the initial value and had no
  # proposal, so it never counts towards the acceptance rate.
  samples <- chain[(burn_in + 1):n_iterations, , drop = FALSE]
  acceptance_rate <- mean(accepted[max(burn_in + 1, 2):n_iterations])


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