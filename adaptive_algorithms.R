# ============================================================================
# ADAPTIVE METROPOLIS (AM)
# ============================================================================
adaptive_metropolis <- function(target_log_density, initial_value, 
                                n_iterations, burn_in = 1000,
                                initial_cov = NULL, adapt_interval = 50) {
  d <- length(initial_value)
  
  # Initialize covariance
  if (is.null(initial_cov)) {
    initial_cov <- diag(0.1^2, d)
  }
  
  chain <- matrix(nrow = n_iterations, ncol = d)
  accepted <- logical(n_iterations)
  current <- initial_value
  current_logp <- target_log_density(current)
  chain[1, ] <- current
  
  # For covariance adaptation
  sample_mean <- current
  sample_cov <- initial_cov
  n_adapted <- 0
  
  for (t in 2:n_iterations) {
    # Adapt covariance every adapt_interval iterations after burn_in
    if (t > burn_in && t %% adapt_interval == 0) {
      sample_mean <- colMeans(chain[1:(t-1), ])
      sample_cov <- cov(chain[1:(t-1), ])
      # Add small diagonal for numerical stability
      sample_cov <- sample_cov + diag(1e-6, d)
      n_adapted <- n_adapted + 1
    }
    
    # Propose using current covariance
    proposal <- MASS::mvrnorm(1, current, 2.38^2 / d * sample_cov)
    proposal_logp <- target_log_density(proposal)
    
    # Accept/reject
    log_alpha <- proposal_logp - current_logp
    if (log(runif(1)) < log_alpha) {
      current <- proposal
      current_logp <- proposal_logp
      accepted[t] <- TRUE
    }
    
    chain[t, ] <- current
  }
  
  list(
    samples = chain[(burn_in+1):n_iterations, , drop = FALSE],
    acceptance_rate = mean(accepted[(burn_in+1):n_iterations]),
    full_chain = chain,
    n_adaptations = n_adapted
  )
}

# ============================================================================
# ROBUST ADAPTIVE METROPOLIS (RAM)
# ============================================================================
robust_adaptive_metropolis <- function(target_log_density, initial_value,
                                       n_iterations, burn_in = 1000,
                                       target_acceptance = 0.234) {
  d <- length(initial_value)
  
  chain <- matrix(nrow = n_iterations, ncol = d)
  accepted <- logical(n_iterations)
  current <- initial_value
  current_logp <- target_log_density(current)
  chain[1, ] <- current
  
  # RAM-specific: scale and shape adaptation
  scale <- 1.0
  shape <- diag(d)
  
  for (t in 2:n_iterations) {
    # Adaptation schedule: gamma_t = 1/t^0.6
    if (t > burn_in) {
      gamma_t <- 1 / (t - burn_in)^0.6
      
      # Update scale based on acceptance rate
      accept_t <- as.numeric(accepted[t-1])
      scale <- scale * exp(gamma_t * (accept_t - target_acceptance))
      
      # Update shape (simplified version)
      if (t %% 50 == 0) {
        recent_samples <- chain[max(1, t-500):(t-1), ]
        shape <- cov(recent_samples) + diag(1e-6, d)
      }
    }
    
    # Propose
    proposal_cov <- scale^2 * shape
    proposal <- MASS::mvrnorm(1, current, proposal_cov)
    proposal_logp <- target_log_density(proposal)
    
    # Accept/reject
    log_alpha <- proposal_logp - current_logp
    if (log(runif(1)) < log_alpha) {
      current <- proposal
      current_logp <- proposal_logp
      accepted[t] <- TRUE
    }
    
    chain[t, ] <- current
  }
  
  list(
    samples = chain[(burn_in+1):n_iterations, , drop = FALSE],
    acceptance_rate = mean(accepted[(burn_in+1):n_iterations]),
    full_chain = chain,
    final_scale = scale
  )
}

# ============================================================================
# NON-ADAPTIVE RANDOM WALK METROPOLIS (BASELINE)
# ============================================================================
random_walk_baseline <- function(target_log_density, initial_value,
                                 n_iterations, burn_in = 1000,
                                 proposal_sd = 2.38) {
  d <- length(initial_value)
  
  # Scale by sqrt(d) AND make smaller for Gaussian targets
  scaled_sd <- (proposal_sd / sqrt(d)) * 0.1  # Add 0.1 multiplier
  
  chain <- matrix(nrow = n_iterations, ncol = d)
  accepted <- logical(n_iterations)
  current <- initial_value
  current_logp <- target_log_density(current)
  chain[1, ] <- current
  
  for (t in 2:n_iterations) {
    proposal <- current + rnorm(d, 0, scaled_sd)
    proposal_logp <- target_log_density(proposal)
    
    log_alpha <- proposal_logp - current_logp
    if (log(runif(1)) < log_alpha) {
      current <- proposal
      current_logp <- proposal_logp
      accepted[t] <- TRUE
    }
    
    chain[t, ] <- current
  }
  
  list(
    samples = chain[(burn_in+1):n_iterations, , drop = FALSE],
    acceptance_rate = mean(accepted[(burn_in+1):n_iterations]),
    full_chain = chain
  )
}