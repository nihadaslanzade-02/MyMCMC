# Algorithm 1.5: Hamiltonian Monte Carlo with Leapfrog Integration
# Samples from target distribution using Hamiltonian dynamics
#
# Input:
#   target_log_density: log of target distribution π(q) (up to constant)
#   target_gradient: gradient of log target density (∇ log π(q))
#   mass_matrix: mass matrix M (default: identity)
#   step_size: leapfrog step size ε
#   n_leapfrog_steps: number of leapfrog steps L
#   initial_value: starting value for the chain
#   n_iterations: number of HMC iterations
#   burn_in: number of burn-in iterations to discard
#
# Output:
#   samples: matrix of HMC samples (after burn-in)
#   acceptance_rate: proportion of accepted proposals
#   energy_changes: changes in Hamiltonian for diagnostics

hamiltonian_monte_carlo <- function(target_log_density,
                                    target_gradient,
                                    mass_matrix = NULL,
                                    step_size,
                                    n_leapfrog_steps,
                                    initial_value,
                                    n_iterations,
                                    burn_in = 0) {
  
  # Determine dimension
  d <- length(initial_value)
  
  # Set default mass matrix (identity)
  if (is.null(mass_matrix)) {
    mass_matrix <- diag(d)
  }
  
  # Compute inverse mass matrix for kinetic energy calculation
  mass_matrix_inv <- solve(mass_matrix)
  
  # Initialize storage
  total_iterations <- n_iterations + burn_in
  chain <- matrix(nrow = total_iterations, ncol = d)
  accepted <- numeric(total_iterations)
  energy_changes <- numeric(total_iterations)
  
  # Initialize chain
  current_q <- initial_value
  chain[1, ] <- current_q
  
  # Main HMC loop
  for (t in 2:total_iterations) {
    
    # Step 1: Sample momentum from N(0, M)
    current_p <- as.vector(mvrnorm_simple(mass_matrix))
    
    # Store initial state
    q <- current_q
    p <- current_p
    
    # Calculate initial Hamiltonian
    initial_U <- -target_log_density(q)  # Potential energy
    initial_K <- 0.5 * t(p) %*% mass_matrix_inv %*% p  # Kinetic energy
    initial_H <- initial_U + initial_K
    
    # Step 2: Leapfrog integration
    # Perform L leapfrog steps
    for (l in 1:n_leapfrog_steps) {
      # Half-step for momentum
      p <- p + (step_size / 2) * target_gradient(q)
      
      # Full-step for position
      q <- q + step_size * as.vector(mass_matrix_inv %*% p)
      
      # Half-step for momentum
      p <- p + (step_size / 2) * target_gradient(q)
    }
    
    # Negate momentum for proposal symmetry
    p <- -p
    
    # Calculate final Hamiltonian
    final_U <- -target_log_density(q)
    final_K <- 0.5 * t(p) %*% mass_matrix_inv %*% p
    final_H <- final_U + final_K
    
    # Step 3: Metropolis acceptance step
    delta_H <- final_H - initial_H
    energy_changes[t] <- as.numeric(delta_H)
    
    # Accept or reject (on log scale)
    log_u <- log(runif(1))
    if (log_u < -delta_H) {
      current_q <- q
      accepted[t] <- 1
    } else {
      # Stay at current position
      accepted[t] <- 0
    }
    
    # Store current state
    chain[t, ] <- current_q
  }
  
  # Remove burn-in period
  if (burn_in > 0) {
    samples <- chain[(burn_in + 1):total_iterations, , drop = FALSE]
    acceptance_rate <- mean(accepted[(burn_in + 1):total_iterations])
    final_energy_changes <- energy_changes[(burn_in + 1):total_iterations]
  } else {
    samples <- chain
    acceptance_rate <- mean(accepted[-1])
    final_energy_changes <- energy_changes[-1]
  }
  
  # Diagnostics
  avg_energy_change <- mean(abs(final_energy_changes[is.finite(final_energy_changes)]))
  if (acceptance_rate < 0.6) {
    cat("Warning: Low acceptance rate (", round(acceptance_rate, 3), 
        "). Consider decreasing step_size.\n", sep = "")
  } else if (acceptance_rate > 0.9) {
    cat("Warning: High acceptance rate (", round(acceptance_rate, 3), 
        "). Consider increasing step_size.\n", sep = "")
  }
  if (avg_energy_change > 1) {
    cat("Warning: Large average energy change (", round(avg_energy_change, 3), 
        "). Leapfrog integration may be unstable.\n", sep = "")
  }
  
  return(list(
    samples = samples,
    acceptance_rate = acceptance_rate,
    full_chain = chain,
    accepted = accepted,
    energy_changes = final_energy_changes,
    step_size = step_size,
    n_leapfrog_steps = n_leapfrog_steps
  ))
}

# ============================================================================
# Helper function: Simple multivariate normal sampler
mvrnorm_simple <- function(sigma) {
  d <- nrow(sigma)
  z <- rnorm(d)
  L <- chol(sigma)  # Cholesky decomposition
  return(L %*% z)
}

# ============================================================================
# Leapfrog Integrator (Standalone Function)
# Performs one trajectory of leapfrog integration
#
# Input:
#   q: initial position
#   p: initial momentum
#   gradient_U: gradient of potential energy U(q) = -log π(q)
#   mass_matrix_inv: inverse of mass matrix
#   step_size: integration step size ε
#   n_steps: number of leapfrog steps
#
# Output:
#   q: final position
#   p: final momentum

leapfrog_integrator <- function(q, p, gradient_U, mass_matrix_inv, 
                                step_size, n_steps) {
  
  # Perform n_steps leapfrog steps
  for (i in 1:n_steps) {
    # Half-step for momentum
    p <- p - (step_size / 2) * gradient_U(q)
    
    # Full-step for position
    q <- q + step_size * as.vector(mass_matrix_inv %*% p)
    
    # Half-step for momentum
    p <- p - (step_size / 2) * gradient_U(q)
  }
  
  return(list(q = q, p = p))
}

# ============================================================================
# Example usage:
# # Define target distribution (e.g., 2D correlated normal)
# target_cov <- matrix(c(1, 0.8, 0.8, 1), 2, 2)
# target_precision <- solve(target_cov)
# 
# target_log_density <- function(q) {
#   -0.5 * t(q) %*% target_precision %*% q
# }
# 
# target_gradient <- function(q) {
#   -target_precision %*% q  # Gradient of log density
# }
# 
# # Run HMC
# result <- hamiltonian_monte_carlo(
#   target_log_density = target_log_density,
#   target_gradient = target_gradient,
#   mass_matrix = diag(2),  # Use identity mass matrix
#   step_size = 0.1,
#   n_leapfrog_steps = 10,
#   initial_value = c(0, 0),
#   n_iterations = 5000,
#   burn_in = 1000
# )
# 
# cat("Acceptance rate:", result$acceptance_rate, "\n")
# cat("Sample mean:", colMeans(result$samples), "\n")
# cat("Sample covariance:\n")
# print(cov(result$samples))
# cat("True covariance:\n")
# print(target_cov)
