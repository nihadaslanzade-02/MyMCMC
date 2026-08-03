# Algorithm 1.3: Metropolis-Hastings Algorithm
# Samples from target distribution using proposal distribution q(y|x)
#
# Input:
#   target_log_density: log of target distribution π(x) (up to constant)
#   proposal_sample: function to sample from q(·|x)
#   proposal_log_density: log of proposal density q(y|x)
#   initial_value: starting value for the chain
#   n_iterations: TOTAL number of iterations to run, burn_in included
#   burn_in: number of leading iterations to discard, must be < n_iterations
#
# Output:
#   samples: matrix of MCMC samples (after burn-in)
#   acceptance_rate: proportion of accepted proposals

metropolis_hastings <- function(target_log_density,
                                proposal_sample,
                                proposal_log_density,
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
  current_log_density <- target_log_density(current_state)
  chain[1, ] <- current_state

  # Main MCMC loop
  for (t in 2:n_iterations) {
    # Propose new state
    proposed_state <- proposal_sample(current_state)
    
    # Calculate log densities
    proposed_log_density <- target_log_density(proposed_state)
    
    # Calculate log acceptance ratio
    log_alpha <- proposed_log_density - current_log_density +
      proposal_log_density(current_state, proposed_state) -
      proposal_log_density(proposed_state, current_state)
    
    # Accept or reject (on log scale for numerical stability)
    log_u <- log(runif(1))
    if (log_u < log_alpha) {
      current_state <- proposed_state
      current_log_density <- proposed_log_density
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

  return(list(
    samples = samples,
    acceptance_rate = acceptance_rate,
    full_chain = chain,
    accepted = accepted
  ))
}


# Algorithm 1.4: Random Walk Metropolis (Special Case)
# Uses symmetric proposal q(y|x) = q(x|y), simplifying acceptance ratio
#
# Input:
#   target_log_density: log of target distribution π(x) (up to constant)
#   proposal_sd: standard deviation for normal random walk proposal
#   initial_value: starting value for the chain
#   n_iterations: TOTAL number of iterations to run, burn_in included
#   burn_in: number of leading iterations to discard, must be < n_iterations
#
# Output:
#   samples: matrix of MCMC samples (after burn-in)
#   acceptance_rate: proportion of accepted proposals

random_walk_metropolis <- function(target_log_density,
                                   proposal_sd,
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


  # Create proposal covariance matrix
  # Scale proposal_sd by 1/sqrt(d) for high dimensions (Roberts et al., 1997)
  if (d > 10) {
    scaled_sd <- proposal_sd / sqrt(d)
  } else {
    scaled_sd <- proposal_sd
  }
  proposal_cov <- diag(scaled_sd^2, d)

  # Initialize storage
  chain <- matrix(nrow = n_iterations, ncol = d)
  accepted <- numeric(n_iterations)

  # Initialize chain
  current_state <- initial_value
  current_log_density <- target_log_density(current_state)
  chain[1, ] <- current_state

  # Main MCMC loop
  for (t in 2:n_iterations) {
    # Propose new state (random walk)
    innovation <- rnorm(d, mean = 0, sd = scaled_sd)
    proposed_state <- current_state + innovation
    
    # Calculate log densities
    proposed_log_density <- target_log_density(proposed_state)
    
    # Calculate log acceptance ratio (simplified for symmetric proposal)
    log_alpha <- proposed_log_density - current_log_density
    
    # Accept or reject
    log_u <- log(runif(1))
    if (log_u < log_alpha) {
      current_state <- proposed_state
      current_log_density <- proposed_log_density
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

  # Print tuning recommendation based on acceptance rate
  if (acceptance_rate < 0.15) {
    cat("Warning: Low acceptance rate (", round(acceptance_rate, 3), 
        "). Consider decreasing proposal_sd.\n", sep = "")
  } else if (acceptance_rate > 0.50 && d > 5) {
    cat("Warning: High acceptance rate (", round(acceptance_rate, 3), 
        "). Consider increasing proposal_sd.\n", sep = "")
  }
  
  return(list(
    samples = samples,
    acceptance_rate = acceptance_rate,
    full_chain = chain,
    accepted = accepted,
    proposal_sd = scaled_sd
  ))
}

# Example usage:
# # Define target distribution (e.g., standard normal)
# target_log_density <- function(x) {
#   -0.5 * sum(x^2)  # Log of multivariate standard normal (up to constant)
# }
# 
# # Run random walk Metropolis
# result <- random_walk_metropolis(
#   target_log_density = target_log_density,
#   proposal_sd = 2.38,  # Optimal for standard normal (Roberts et al., 1997)
#   initial_value = c(0, 0),
#   n_iterations = 10000,
#   burn_in = 1000
# )
# 
# cat("Acceptance rate:", result$acceptance_rate, "\n")
# cat("Sample mean:", colMeans(result$samples), "\n")
# cat("Sample covariance:\n")
# print(cov(result$samples))