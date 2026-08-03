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
#
# Two versions of each are reported.
#
# `rmse` and `mae` are in the parameters' own units. They are the right thing
# to read for a single model and meaningless to average across models, because
# the benchmark's posteriors do not share a scale: earnings-earn_height is on
# a raw dollar scale where a parameter's standard deviation is 9668, while
# every parameter in diamonds-diamonds sits below 0.33. An unweighted mean
# RMSE over models is therefore just earnings-earn_height, and the 61.8
# reported for RAM in the first run was exactly that.
#
# `rmse_normalised` and `mae_normalised` divide each parameter's error by that
# parameter's standard deviation under the reference posterior first, so the
# unit is "reference posterior standard deviations" and is the same unit in
# every model. Missing the posterior mean by one full standard deviation
# scores 1 whether the parameter is measured in dollars or in log-odds. These
# are the ones to aggregate.
compute_accuracy <- function(samples, reference_means, runtime,
                             reference_sds = NULL) {

  tryCatch({
    sample_means <- colMeans(samples)
    error <- sample_means - reference_means

    rmse_normalised <- NA_real_
    mae_normalised <- NA_real_

    if (!is.null(reference_sds)) {
      # A parameter that is constant under the reference posterior has no
      # scale to divide by. Rather than return Inf for the whole model, drop
      # those coordinates and say so.
      usable <- is.finite(reference_sds) & reference_sds > 0
      if (!all(usable)) {
        warning(sum(!usable), " of ", length(usable), " parameters have a ",
                "zero or non-finite reference SD and are excluded from the ",
                "normalised accuracy metrics")
      }
      if (any(usable)) {
        scaled <- error[usable] / reference_sds[usable]
        rmse_normalised <- sqrt(mean(scaled^2))
        mae_normalised <- mean(abs(scaled))
      }
    }

    list(
      rmse = sqrt(mean(error^2)),
      mae = mean(abs(error)),
      rmse_normalised = rmse_normalised,
      mae_normalised = mae_normalised,
      runtime = runtime
    )
  }, error = function(e) {
    warning("Error computing accuracy: ", e$message)
    list(rmse = NA_real_, mae = NA_real_, rmse_normalised = NA_real_,
         mae_normalised = NA_real_, runtime = runtime)
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
