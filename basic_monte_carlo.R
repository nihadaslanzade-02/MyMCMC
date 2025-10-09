# Algorithm 1.1: Basic Monte Carlo Integration
# Estimates integral I = ∫_Ω f(x)dx using Monte Carlo sampling
#
# Input:
#   f: function to integrate
#   Omega: domain of integration (specified by bounds)
#   N: number of Monte Carlo samples
#   d: dimension of the integration domain
#
# Output:
#   I_hat: Monte Carlo estimate of the integral
#   se: standard error of the estimate

basic_monte_carlo <- function(f, lower_bounds, upper_bounds, N) {
  # Determine dimension
  d <- length(lower_bounds)
  
  # Calculate volume of integration domain
  V <- prod(upper_bounds - lower_bounds)
  
  # Generate N uniform random samples in Omega
  samples <- matrix(nrow = N, ncol = d)
  for (i in 1:d) {
    samples[, i] <- runif(N, min = lower_bounds[i], max = upper_bounds[i])
  }
  
  # Evaluate function at each sample point
  f_values <- numeric(N)
  for (i in 1:N) {
    f_values[i] <- f(samples[i, ])
  }
  
  # Compute Monte Carlo estimate
  I_hat <- V * mean(f_values)
  
  # Compute standard error
  se <- V * sd(f_values) / sqrt(N)
  
  # Return estimate and standard error
  return(list(
    estimate = I_hat,
    standard_error = se,
    samples = samples,
    function_values = f_values
  ))
}

# Example usage:
# f <- function(x) exp(-sum(x^2))  # Gaussian function
# result <- basic_monte_carlo(f, lower_bounds = c(-2, -2), 
#                            upper_bounds = c(2, 2), N = 10000)
# cat("Estimate:", result$estimate, "\n")
# cat("Standard Error:", result$standard_error, "\n")