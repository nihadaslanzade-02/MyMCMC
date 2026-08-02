# ============================================================================
# BENCHMARK METRICS
# ============================================================================
# Kept separate from run_benchmarking.R so the metrics can be sourced and
# tested on their own. That file ends in a main-execution block, so sourcing
# it starts a full benchmark sweep.

library(posterior)

# ============================================================================
# ACCURACY, PER CHAIN
# ============================================================================
# RMSE and MAE compare one chain's posterior mean against the reference, so
# they are genuinely per-chain quantities and are computed inside the chain
# loop. ESS and R-hat are not: see compute_convergence() below.
compute_accuracy <- function(samples, reference_means, runtime) {

  tryCatch({
    sample_means <- colMeans(samples)

    list(
      rmse = sqrt(mean((sample_means - reference_means)^2)),
      mae = mean(abs(sample_means - reference_means)),
      runtime = runtime
    )
  }, error = function(e) {
    warning("Error computing accuracy: ", e$message)
    list(rmse = NA_real_, mae = NA_real_, runtime = runtime)
  })
}

# ============================================================================
# CONVERGENCE, ACROSS CHAINS
# ============================================================================
# Both diagnostics here are properties of a set of chains taken together, and
# both are reported per parameter.
#
# Two mistakes are easy to make and were both made in an earlier version of
# this script, so they are worth naming.
#
# 1. Handing a whole (draws x parameters) matrix to ess_bulk() or rhat().
#    Those functions reduce their input to a single number. With several
#    parameters sitting at different locations, the spread between parameters
#    swamps the autocorrelation within each one, and the estimate collapses
#    onto the parameter count: on independent, perfectly mixed draws a
#    3-parameter model reports ESS 3.7 and a 26-parameter model reports 26.4,
#    whatever the sampler did. The number that comes back is the dimension
#    wearing an ESS label.
#
# 2. Computing R-hat inside the chain loop. R-hat compares between-chain
#    variance against within-chain variance; given one chain there is no
#    between-chain term to form, and averaging four such values afterwards
#    does not reconstruct it. The diagnostic that detects non-mixing is then
#    never actually computed.
#
# The fix for both is the same: assemble an iterations x chains x parameters
# array and let posterior::summarise_draws() work over it.
compute_convergence <- function(chain_samples, param_names, total_runtime) {

  tryCatch({
    n_chains <- length(chain_samples)
    n_draws <- nrow(chain_samples[[1]])
    d <- ncol(chain_samples[[1]])

    if (n_chains < 2) {
      warning("R-hat needs at least 2 chains; got ", n_chains)
    }

    # iterations x chains x parameters, the layout posterior expects
    draws_array <- array(
      NA_real_,
      dim = c(n_draws, n_chains, d),
      dimnames = list(NULL, NULL, param_names[seq_len(d)])
    )
    for (i in seq_len(n_chains)) {
      draws_array[, i, ] <- chain_samples[[i]]
    }

    diagnostics <- posterior::summarise_draws(
      posterior::as_draws_array(draws_array),
      "ess_bulk", "ess_tail", "rhat"
    )

    ess_bulk <- diagnostics$ess_bulk
    ess_tail <- diagnostics$ess_tail
    rhat <- diagnostics$rhat

    list(
      ess_bulk_median = median(ess_bulk, na.rm = TRUE),
      ess_bulk_min = min(ess_bulk, na.rm = TRUE),
      ess_tail_min = min(ess_tail, na.rm = TRUE),
      ess_per_sec_median = median(ess_bulk, na.rm = TRUE) / total_runtime,
      ess_per_sec_min = min(ess_bulk, na.rm = TRUE) / total_runtime,
      rhat_max = max(rhat, na.rm = TRUE),
      n_params_unconverged = sum(!is.finite(rhat) | rhat > 1.01, na.rm = TRUE),
      n_params = d,
      total_draws = n_draws * n_chains,
      per_parameter = as.data.frame(diagnostics)
    )
  }, error = function(e) {
    warning("Error computing convergence diagnostics: ", e$message)
    list(
      ess_bulk_median = NA_real_, ess_bulk_min = NA_real_,
      ess_tail_min = NA_real_, ess_per_sec_median = NA_real_,
      ess_per_sec_min = NA_real_, rhat_max = NA_real_,
      n_params_unconverged = NA_integer_, n_params = NA_integer_,
      total_draws = NA_integer_, per_parameter = NULL
    )
  })
}
