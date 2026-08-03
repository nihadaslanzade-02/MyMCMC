# ============================================================================
# BENCHMARKING ADAPTIVE MCMC WITH POSTERIORDB
# Stage 2: build the targets, run the chains, measure them
# ============================================================================
# Stage 1 (benchmarking_posteriordb.R) has already selected the models and
# saved their reference draws into benchmark_config.rds. This stage reads them
# from there rather than going back to GitHub for the same data, so it runs
# offline and reproducibly. The posteriordb connection is opened lazily, and
# only when a target actually needs the Stan program - see stan_targets.R.

library(posterior)
library(dplyr)

source("adaptive_algorithms.R")
source("benchmark_metrics.R")
source("stan_targets.R")

# ============================================================================
# SETTINGS
# ============================================================================
# Everything that decides what gets run lives here, in the script that runs
# it. An earlier version also carried an algorithm settings block inside
# benchmark_config.rds - adaptation_start, target_acceptance, proposal_sd, a
# thinning factor and a seed - that nothing ever read. Settings buried in a
# 9 MB binary you cannot open in an editor, disagreeing with the code, are
# worse than no settings at all.
BENCHMARK_CONFIG <- list(
  # Total chain length. Every sampler here treats n_iterations as the whole
  # chain with burn_in taken out of it; see the note at the top of
  # adaptive_algorithms.R.
  n_iterations = 100000,
  burn_in_fraction = 0.5,
  n_chains = 4,

  # Skip anything wider than this. mcycle_gp-accel_gp has 66 parameters and
  # is the one selected model this excludes.
  max_dimension = 50,

  # "gaussian"   - fit a multivariate normal to the reference draws
  # "bridgestan" - compile and call the posterior's own Stan program
  # "auto"       - prefer bridgestan, fall back to the surrogate with a note
  target_method = "auto",

  # Chains are seeded deterministically from this, so a re-run reproduces the
  # numbers exactly. The seed used to be declared in stage 1 and never set.
  seed = 42
)

# One seed per (model, algorithm, chain), so that chains are independent of
# each other and none of them depends on how many ran before.
chain_seed <- function(base, model_index, algo_index, chain) {
  base * 1000000L + model_index * 10000L + algo_index * 100L + chain
}

# Each algorithm behind the same call, because they do not take the same
# arguments. The baseline is the only one that needs the target's scale: it
# never adapts, so it has to be told the size of a sensible step up front.
ALGORITHMS <- list(
  "AM" = function(target, init, n_iterations, burn_in) {
    adaptive_metropolis(target$log_density, init, n_iterations, burn_in)
  },
  "RAM" = function(target, init, n_iterations, burn_in) {
    robust_adaptive_metropolis(target$log_density, init, n_iterations, burn_in)
  },
  "RWM_baseline" = function(target, init, n_iterations, burn_in) {
    random_walk_baseline(target$log_density, init, n_iterations, burn_in,
                         proposal_scale = target$sampling_scale)
  }
)

# ============================================================================
# RUN BENCHMARK FOR ONE MODEL
# ============================================================================
benchmark_model <- function(posterior_name, model_entry, pdb, algorithms,
                            model_index, settings) {

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat("BENCHMARKING: ", posterior_name, "\n", sep = "")
  cat(strrep("=", 70), "\n", sep = "")

  ref_matrix <- as.matrix(model_entry$reference_draws)
  n_params <- ncol(ref_matrix)

  # Checked before building anything, because compiling a Stan model for a
  # posterior that is then skipped costs minutes for nothing.
  if (n_params > settings$max_dimension) {
    cat("  SKIPPING: dimension ", n_params, " exceeds maximum ",
        settings$max_dimension, "\n", sep = "")
    return(NULL)
  }

  target <- tryCatch(
    build_target(posterior_name, ref_matrix, pdb, method = settings$target_method),
    error = function(e) {
      cat("  SKIPPING: could not build a target:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (is.null(target)) return(NULL)

  n_iterations <- settings$n_iterations
  burn_in <- floor(n_iterations * settings$burn_in_fraction)

  cat("  Target: ", target$kind, ", ", n_params, " parameters",
      if (target$dimension != n_params)
        paste0(" (", target$dimension, " unconstrained)") else "",
      "\n", sep = "")

  results <- list()

  for (algo_index in seq_along(algorithms)) {
    algo_name <- names(algorithms)[algo_index]
    cat("\n  Running:", algo_name, "\n")

    algo_results <- list()
    chain_samples <- list()

    for (chain in seq_len(settings$n_chains)) {
      cat("    Chain ", chain, "...", sep = "")

      set.seed(chain_seed(settings$seed, model_index, algo_index, chain))
      init <- target$init()

      start_time <- Sys.time()
      algo_output <- tryCatch({
        algorithms[[algo_name]](target, init, n_iterations, burn_in)
      }, error = function(e) {
        cat(" ERROR:", conditionMessage(e), "\n")
        NULL
      })
      if (is.null(algo_output)) next
      runtime <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

      cat(" done (", round(runtime, 1), "s, acc=",
          round(algo_output$acceptance_rate, 3), ")\n", sep = "")

      # Everything is measured on the constrained scale, so the surrogate and
      # bridgestan runs report the same quantities and stay comparable.
      draws <- target$to_constrained(algo_output$samples)

      # Per-chain accuracy only. The convergence diagnostics need every chain
      # at once and are computed below, after the loop.
      accuracy <- compute_accuracy(draws, target$reference_mean, runtime,
                                   target$reference_sd)

      chain_samples[[length(chain_samples) + 1]] <- draws
      algo_results[[length(algo_results) + 1]] <- c(
        accuracy,
        acceptance_rate = algo_output$acceptance_rate
      )
    }

    if (length(algo_results) == 0) {
      cat("    No successful chains for", algo_name, "\n")
      next
    }

    per_chain <- data.frame(do.call(rbind, algo_results))

    # Chains run sequentially, so the wall-clock cost of the whole set is
    # their total. ESS is likewise an all-chains quantity, so ESS/second is
    # formed from the two totals rather than from one chain's share.
    total_runtime <- sum(unlist(per_chain$runtime), na.rm = TRUE)

    convergence <- compute_convergence(chain_samples, target$param_names,
                                       total_runtime)

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
  }

  # Metadata. target_kind travels with the result so that a summary can never
  # silently mix a run against the real posterior with one against a Gaussian
  # fitted to it.
  attr(results, "model_name") <- posterior_name
  attr(results, "dimension") <- n_params
  attr(results, "sampling_dimension") <- target$dimension
  attr(results, "target_kind") <- target$kind
  attr(results, "n_chains") <- settings$n_chains
  attr(results, "n_iterations") <- n_iterations
  attr(results, "burn_in") <- burn_in

  results
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

cat("\n")
cat(strrep("=", 70), "\n", sep = "")
cat("ADAPTIVE MCMC BENCHMARKING WITH POSTERIORDB\n")
cat(strrep("=", 70), "\n\n", sep = "")

cat("Loading configuration...\n")
config <- readRDS("benchmark_config.rds")
selected_models <- names(config$models)

if (length(selected_models) == 0) {
  stop("benchmark_config.rds holds no models. Run benchmarking_posteriordb.R first.")
}

cat("Found", length(selected_models), "models to benchmark\n")
cat("Testing", length(ALGORITHMS), "algorithms:",
    paste(names(ALGORITHMS), collapse = ", "), "\n")

cat("\nBenchmark settings:\n")
cat("  Iterations (total): ", BENCHMARK_CONFIG$n_iterations, "\n", sep = "")
cat("  Burn-in:            ",
    floor(BENCHMARK_CONFIG$n_iterations * BENCHMARK_CONFIG$burn_in_fraction),
    "\n", sep = "")
cat("  Chains:             ", BENCHMARK_CONFIG$n_chains, "\n", sep = "")
cat("  Max dimension:      ", BENCHMARK_CONFIG$max_dimension, "\n", sep = "")
cat("  Target:             ", BENCHMARK_CONFIG$target_method, "\n", sep = "")
cat("  Seed:               ", BENCHMARK_CONFIG$seed, "\n", sep = "")

# Only opened if a target needs the Stan source; the reference draws come from
# the config file.
pdb <- NULL
if (BENCHMARK_CONFIG$target_method != "gaussian") {
  pdb <- tryCatch(posteriordb::pdb_github(), error = function(e) {
    cat("\nCould not reach posteriordb (", conditionMessage(e), ").\n", sep = "")
    cat("Falling back to the Gaussian surrogate for every model.\n")
    BENCHMARK_CONFIG$target_method <<- "gaussian"
    NULL
  })
}

all_results <- list()
successful_models <- 0
failed_models <- 0

for (i in seq_along(selected_models)) {
  model_name <- selected_models[i]
  cat("\n[", i, "/", length(selected_models), "] ", sep = "")

  result <- tryCatch({
    benchmark_model(model_name, config$models[[model_name]], pdb, ALGORITHMS,
                    model_index = i, settings = BENCHMARK_CONFIG)
  }, error = function(e) {
    cat("FATAL ERROR with", model_name, ":", conditionMessage(e), "\n")
    NULL
  })

  if (!is.null(result) && length(result) > 0) {
    all_results[[model_name]] <- result
    successful_models <- successful_models + 1
  } else {
    failed_models <- failed_models + 1
  }

  # Checkpoint, in the same envelope as the final file so it can be inspected
  # with the same code.
  if (i %% 3 == 0) {
    saveRDS(list(schema = "chainwise-diagnostics-v2", results = all_results,
                 config = BENCHMARK_CONFIG, partial = TRUE,
                 models_done = i, models_total = length(selected_models)),
            "benchmark_results_partial.rds")
    cat("\n  [Saved intermediate results]\n")
  }
}

# ============================================================================
# SAVE FINAL RESULTS
# ============================================================================

cat("\n\n")
cat(strrep("=", 70), "\n", sep = "")
cat("BENCHMARKING COMPLETE\n")
cat(strrep("=", 70), "\n", sep = "")
cat("Successful models:", successful_models, "\n")
cat("Failed/skipped models:", failed_models, "\n")

target_kinds <- vapply(all_results, function(r) attr(r, "target_kind"), character(1))
cat("Targets used:", paste(names(table(target_kinds)), table(target_kinds),
                           sep = " x ", collapse = ", "), "\n")

# `schema` marks results produced by the corrected diagnostics. Files written
# before that fix carry per-chain ESS and R-hat values that are not what their
# names say, so analyze_results.R checks for this marker and refuses to plot a
# file without it rather than silently charting the old numbers.
final_output <- list(
  schema = "chainwise-diagnostics-v2",
  results = all_results,
  config = BENCHMARK_CONFIG,
  algorithms = names(ALGORITHMS),
  target_kinds = target_kinds,
  timestamp = Sys.time(),
  n_successful = successful_models,
  n_failed = failed_models,
  selected_models = selected_models,
  session = list(r_version = R.version.string,
                 posterior = as.character(utils::packageVersion("posterior")))
)

saveRDS(final_output, "benchmark_results.rds")

cat("\nResults saved to: benchmark_results.rds\n")
cat("\nDone!\n")
