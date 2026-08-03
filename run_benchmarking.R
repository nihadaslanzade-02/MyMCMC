# ============================================================================
# BENCHMARKING ADAPTIVE MCMC WITH POSTERIORDB
# Complete benchmarking with Stan model integration
# ============================================================================

library(bridgestan)
library(posteriordb)
library(posterior)
library(dplyr)

source("adaptive_algorithms.R")
source("benchmark_metrics.R")

# ============================================================================
# CREATE LOG DENSITY FUNCTION
# ============================================================================
create_log_density <- function(posterior_name, pdb) {
  
  cat("  Setting up model...\n")
  
  # Get posterior object
  po <- posteriordb::posterior(posterior_name, pdb)
  
  # Load reference posterior to understand parameter structure
  ref_draws <- tryCatch({
    posteriordb::reference_posterior_draws(po)
  }, error = function(e) {
    cat("    ERROR loading reference draws:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(ref_draws)) {
    cat("    No reference posterior available\n")
    return(list(success = FALSE))
  }
  
  ref_matrix <- posterior::as_draws_matrix(ref_draws)
  
  # Get parameter names (exclude lp__, etc.)
  all_names <- colnames(ref_matrix)
  param_names <- all_names[!grepl("^lp__|^energy__|^accept_stat__|^stepsize__|^treedepth__|^n_leapfrog__|^divergent__", all_names)]
  d <- length(param_names)
  
  if (d == 0) {
    cat("    ERROR: No valid parameters found\n")
    return(list(success = FALSE))
  }
  
  cat("    Dimension:", d, "parameters\n")
  
  # Extract reference statistics
  ref_matrix_params <- ref_matrix[, param_names, drop = FALSE]
  ref_mean <- colMeans(ref_matrix_params)
  ref_cov <- cov(ref_matrix_params)
  
  # Add small regularization for numerical stability
  ref_cov <- ref_cov + diag(1e-6, d)
  
  # Compute inverse covariance
  ref_cov_inv <- tryCatch({
    solve(ref_cov)
  }, error = function(e) {
    cat("    WARNING: Singular covariance, using diagonal approximation\n")
    diag(1 / (diag(ref_cov) + 1e-6))
  })
  
  # Compute log determinant
  log_det <- determinant(ref_cov, logarithm = TRUE)$modulus[1]
  
  # Create multivariate normal log density
  log_density_fn <- function(theta) {
    if (any(!is.finite(theta))) return(-Inf)
    if (length(theta) != d) return(-Inf)
    
    # MVN log density: -0.5 * (log(det(Sigma)) + (x-mu)' Sigma^-1 (x-mu) + d*log(2*pi))
    delta <- theta - ref_mean
    quad_form <- as.numeric(t(delta) %*% ref_cov_inv %*% delta)
    
    -0.5 * (log_det + quad_form + d * log(2 * pi))
  }
  
  return(list(
    log_density = log_density_fn,
    dimension = d,
    param_names = param_names,
    reference_mean = ref_mean,
    reference_cov = ref_cov,
    success = TRUE
  ))
}

# ============================================================================
# RUN BENCHMARK FOR ONE MODEL
# ============================================================================
benchmark_model <- function(posterior_name, pdb, algorithms, 
                            n_iterations = 10000, n_chains = 4,
                            max_dimension = 50) {
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("BENCHMARKING:", posterior_name, "\n")
  cat(rep("=", 70), "\n", sep = "")
  
  # CREATE LOG DENSITY FUNCTION
  density_info <- create_log_density(posterior_name, pdb)
  
  if (!density_info$success) {
    cat("  SKIPPING: Failed to compile model\n")
    return(NULL)
  }
  
  d <- density_info$dimension
  
  # Skip if dimension too large
  if (d > max_dimension) {
    cat("  SKIPPING: Dimension", d, "exceeds maximum", max_dimension, "\n")
    return(NULL)
  }
  
  log_density <- density_info$log_density
  param_names <- density_info$param_names
  
  # Load reference posterior
  cat("  Loading reference posterior...\n")
  po <- posteriordb::posterior(posterior_name, pdb)
  
  ref_draws <- tryCatch({
    posteriordb::reference_posterior_draws(po)
  }, error = function(e) {
    cat("  ERROR loading reference draws:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(ref_draws)) {
    cat("  SKIPPING: No reference posterior available\n")
    return(NULL)
  }
  
  ref_matrix <- posterior::as_draws_matrix(ref_draws)
  
  # Match parameter names
  common_params <- intersect(param_names, colnames(ref_matrix))
  if (length(common_params) == 0) {
    cat("  SKIPPING: No matching parameters between model and reference\n")
    return(NULL)
  }
  
  ref_matrix <- ref_matrix[, common_params, drop = FALSE]
  cat("  Using", length(common_params), "matched parameters\n")
  
  # Initial value (use reference mean)
  initial <- colMeans(ref_matrix)
  reference_means <- colMeans(ref_matrix)
  # Per-parameter spread under the reference posterior, used to put the
  # accuracy metrics on a scale that is comparable across models.
  reference_sds <- apply(ref_matrix, 2, sd)

  results <- list()

  # Run each algorithm
  for (algo_name in names(algorithms)) {
    cat("\n  Running:", algo_name, "\n")

    algo_results <- list()
    chain_samples <- list()

    for (chain in 1:n_chains) {
      cat("    Chain", chain, "...")

      # Slight perturbation of initial value
      init <- initial + rnorm(d, 0, 0.1)

      # Time the algorithm
      start_time <- Sys.time()

      algo_output <- tryCatch({
        algorithms[[algo_name]](
          target_log_density = log_density,
          initial_value = init,
          n_iterations = n_iterations,
          burn_in = floor(n_iterations / 2)
        )
      }, error = function(e) {
        cat(" ERROR:", e$message, "\n")
        return(NULL)
      })

      if (is.null(algo_output)) {
        next
      }

      runtime <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      cat(" Done (", round(runtime, 1), "s, acc=",
          round(algo_output$acceptance_rate, 3), ")\n", sep = "")

      # Per-chain accuracy only. The convergence diagnostics need every chain
      # at once and are computed below, after the loop.
      accuracy <- compute_accuracy(algo_output$samples, reference_means, runtime,
                                   reference_sds)

      chain_samples[[length(chain_samples) + 1]] <- algo_output$samples

      algo_results[[length(algo_results) + 1]] <- c(
        accuracy,
        acceptance_rate = algo_output$acceptance_rate
      )
    }

    # Aggregate across chains
    if (length(algo_results) > 0) {
      per_chain <- data.frame(do.call(rbind, algo_results))

      # Chains run sequentially, so the wall-clock cost of the whole set is
      # their total. ESS is likewise an all-chains quantity, so ESS/second is
      # formed from the two totals rather than from one chain's share.
      total_runtime <- sum(unlist(per_chain$runtime), na.rm = TRUE)

      convergence <- compute_convergence(chain_samples, common_params, total_runtime)

      cat(sprintf(
        "    -> ESS bulk median %.1f of %d draws, R-hat max %.3f (%d/%d parameters unconverged)\n",
        convergence$ess_bulk_median, convergence$total_draws,
        convergence$rhat_max, convergence$n_params_unconverged, convergence$n_params
      ))

      results[[algo_name]] <- list(
        per_chain = per_chain,
        convergence = convergence,
        total_runtime = total_runtime,
        n_chains_completed = length(chain_samples)
      )
    } else {
      cat("    No successful chains for", algo_name, "\n")
    }
  }
  
  # Add metadata
  attr(results, "model_name") <- posterior_name
  attr(results, "dimension") <- d
  attr(results, "n_chains") <- n_chains
  attr(results, "n_iterations") <- n_iterations
  
  results
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

cat("\n")
cat(rep("=", 70), "\n", sep = "")
cat("ADAPTIVE MCMC BENCHMARKING WITH POSTERIORDB\n")
cat(rep("=", 70), "\n\n", sep = "")

# Load configuration
cat("Loading configuration...\n")
pdb <- posteriordb::pdb_github()
config <- readRDS("benchmark_config.rds")
selected_models <- names(config$models)

cat("Found", length(selected_models), "models to benchmark\n")

# Define algorithms
algorithms <- list(
  "AM" = adaptive_metropolis,
  "RAM" = robust_adaptive_metropolis,
  "RWM_baseline" = random_walk_baseline
)

cat("Testing", length(algorithms), "algorithms:", 
    paste(names(algorithms), collapse = ", "), "\n")

# Benchmarking parameters
BENCHMARK_CONFIG <- list(
  n_iterations = 10000,
  n_chains = 4,
  max_dimension = 50  # Skip models with too many parameters
)

cat("\nBenchmark settings:\n")
cat("  Iterations:", BENCHMARK_CONFIG$n_iterations, "\n")
cat("  Chains per algorithm:", BENCHMARK_CONFIG$n_chains, "\n")
cat("  Max dimension:", BENCHMARK_CONFIG$max_dimension, "\n")

# Run benchmark
all_results <- list()
successful_models <- 0
failed_models <- 0

for (i in seq_along(selected_models)) {
  model_name <- selected_models[i]
  
  cat("\n[", i, "/", length(selected_models), "] ", sep = "")
  
  result <- tryCatch({
    benchmark_model(
      model_name, 
      pdb, 
      algorithms,
      n_iterations = BENCHMARK_CONFIG$n_iterations,
      n_chains = BENCHMARK_CONFIG$n_chains,
      max_dimension = BENCHMARK_CONFIG$max_dimension
    )
  }, error = function(e) {
    cat("FATAL ERROR with", model_name, ":", e$message, "\n")
    traceback()
    return(NULL)
  })
  
  if (!is.null(result) && length(result) > 0) {
    all_results[[model_name]] <- result
    successful_models <- successful_models + 1
  } else {
    failed_models <- failed_models + 1
  }
  
  # Save intermediate results
  if (i %% 3 == 0) {
    saveRDS(all_results, "benchmark_results_partial.rds")
    cat("\n  [Saved intermediate results]\n")
  }
}

# ============================================================================
# SAVE FINAL RESULTS
# ============================================================================

cat("\n\n")
cat(rep("=", 70), "\n", sep = "")
cat("BENCHMARKING COMPLETE\n")
cat(rep("=", 70), "\n", sep = "")
cat("Successful models:", successful_models, "\n")
cat("Failed/skipped models:", failed_models, "\n")
cat("\nResults saved to: benchmark_results.rds\n")

# Save with metadata.
#
# `schema` marks results produced by the corrected diagnostics. Files written
# before that fix carry per-chain ESS and R-hat values that are not what their
# names say, so analyze_results.R checks for this marker and refuses to plot a
# file without it rather than silently charting the old numbers.
final_output <- list(
  schema = "chainwise-diagnostics-v2",
  results = all_results,
  config = BENCHMARK_CONFIG,
  algorithms = names(algorithms),
  timestamp = Sys.time(),
  n_successful = successful_models,
  n_failed = failed_models,
  selected_models = selected_models
)

saveRDS(final_output, "benchmark_results.rds")

cat("\nDone!\n")