# ============================================================================
# THE BENCHMARK SWEEP
# ============================================================================
# run_benchmarking.R ends in a call to run_benchmark_sweep(), so sourcing it
# used to start a run that takes half an hour. The option below suppresses
# that call, which is what lets the stage be tested at all.

options(mymcmc.source_only = TRUE)
source("run_benchmarking.R")

# Two tiny posteriors with known moments, in the shape stage 1 saves.
tiny_config <- function(path) {
  set.seed(31)
  models <- list()
  for (spec in list(list(nm = "tiny-a", d = 2, scale = c(1000, 0.001)),
                    list(nm = "tiny-b", d = 3, scale = c(1, 1, 1)))) {
    draws <- sweep(matrix(rnorm(4000 * spec$d), 4000, spec$d), 2, spec$scale, "*")
    colnames(draws) <- paste0("theta[", seq_len(spec$d), "]")
    models[[spec$nm]] <- list(name = spec$nm, reference_draws = draws)
  }
  saveRDS(list(models = models, selected = names(models)), path)
}

tiny_settings <- list(
  n_iterations = 3000, burn_in_fraction = 0.5, n_chains = 2,
  max_dimension = 50, target_method = "gaussian", seed = 42
)

# ---------------------------------------------------------------------------
test_that("chain seeds are distinct and depend on every coordinate", {
  # Two chains that share a seed are not two chains. The old harness never set
  # a seed at all, so a re-run could not reproduce a single number.
  grid <- expand.grid(model = 1:4, algo = 1:3, chain = 1:4)
  seeds <- mapply(chain_seed, 42, grid$model, grid$algo, grid$chain)

  expect_equal(length(unique(seeds)), nrow(grid))
  expect_true(chain_seed(42, 1, 1, 1) != chain_seed(43, 1, 1, 1))
  expect_true(chain_seed(42, 1, 1, 1) != chain_seed(42, 2, 1, 1))
  expect_true(chain_seed(42, 1, 1, 1) != chain_seed(42, 1, 2, 1))
  expect_true(chain_seed(42, 1, 1, 1) != chain_seed(42, 1, 1, 2))
})

# ---------------------------------------------------------------------------
test_that("a sweep produces a schema-marked file the analysis will accept", {
  dir <- file.path(tempdir(), "sweep")
  dir.create(dir, showWarnings = FALSE)
  old <- setwd(dir)
  on.exit(setwd(old), add = TRUE)

  tiny_config("benchmark_config.rds")
  out <- quietly(run_benchmark_sweep(settings = tiny_settings))

  expect_equal(out$schema, "chainwise-diagnostics-v2")
  expect_equal(out$n_successful, 2)
  expect_equal(sort(names(out$results)), c("tiny-a", "tiny-b"))

  # The settings actually used travel with the results, so a stale file cannot
  # be read as if it came from the current configuration.
  expect_equal(out$config$n_iterations, 3000)
  expect_true(all(out$target_kinds == "gaussian-surrogate"))

  for (model in out$results) {
    expect_equal(sort(names(model)), c("AM", "RAM", "RWM_baseline"))
    for (algo in model) {
      expect_equal(nrow(algo$per_chain), 2)
      expect_true(all(is.finite(unlist(algo$per_chain$rmse_normalised))))
      expect_equal(algo$n_chains_completed, 2L)
      expect_true(is.finite(algo$convergence$rhat_max))
      # Diagnostics are per parameter and across chains, so the row count is
      # the parameter count and the draw total covers both chains.
      expect_equal(nrow(algo$convergence$per_parameter), algo$convergence$n_params)
      expect_equal(algo$convergence$total_draws, 1500L * 2L)
    }
  }
})

# ---------------------------------------------------------------------------
test_that("the same seed reproduces the run exactly", {
  # The point of seeding it at all. Without this the committed results cannot
  # be checked by anyone, including their author.
  dir <- file.path(tempdir(), "sweep_repeat")
  dir.create(dir, showWarnings = FALSE)
  old <- setwd(dir)
  on.exit(setwd(old), add = TRUE)

  tiny_config("benchmark_config.rds")
  first <- quietly(run_benchmark_sweep(settings = tiny_settings))
  second <- quietly(run_benchmark_sweep(settings = tiny_settings))

  expect_equal(first$results[["tiny-a"]]$AM$per_chain$rmse,
               second$results[["tiny-a"]]$AM$per_chain$rmse)
  expect_equal(first$results[["tiny-b"]]$RAM$convergence$ess_bulk_median,
               second$results[["tiny-b"]]$RAM$convergence$ess_bulk_median)

  # A different seed must actually change something, or the seed is not
  # reaching the samplers.
  other <- quietly(run_benchmark_sweep(
    settings = modifyList(tiny_settings, list(seed = 7))
  ))
  expect_true(!isTRUE(all.equal(
    first$results[["tiny-a"]]$AM$per_chain$rmse,
    other$results[["tiny-a"]]$AM$per_chain$rmse
  )))
})

# ---------------------------------------------------------------------------
test_that("the baseline is handed the target's scale, the adaptive ones are not", {
  # tiny-a has coordinates whose SDs differ by a factor of a million. A
  # non-adaptive proposal with one absolute step size cannot move on it, which
  # is exactly the state the committed run's baseline was in.
  dir <- file.path(tempdir(), "sweep_scale")
  dir.create(dir, showWarnings = FALSE)
  old <- setwd(dir)
  on.exit(setwd(old), add = TRUE)

  tiny_config("benchmark_config.rds")
  out <- quietly(run_benchmark_sweep(settings = tiny_settings))

  baseline <- out$results[["tiny-a"]]$RWM_baseline
  expect_gt(mean(unlist(baseline$per_chain$acceptance_rate)), 0.05)
})

# ---------------------------------------------------------------------------
test_that("a model wider than max_dimension is skipped before anything is built", {
  dir <- file.path(tempdir(), "sweep_skip")
  dir.create(dir, showWarnings = FALSE)
  old <- setwd(dir)
  on.exit(setwd(old), add = TRUE)

  tiny_config("benchmark_config.rds")
  out <- quietly(run_benchmark_sweep(
    settings = modifyList(tiny_settings, list(max_dimension = 2))
  ))

  # tiny-b has 3 parameters and tiny-a has 2.
  expect_equal(names(out$results), "tiny-a")
  expect_equal(out$n_failed, 1)
})
