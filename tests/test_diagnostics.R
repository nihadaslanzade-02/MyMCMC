# ============================================================================
# CONVERGENCE DIAGNOSTICS
# ============================================================================
# Regression tests for the two diagnostics that were being computed wrongly.
# Every case below fails against the previous implementation.

source("benchmark_metrics.R")

# Four independent chains from a d-dimensional standard normal, each parameter
# shifted to its own location. Perfectly mixed by construction, so the right
# answers are known without running a sampler: ESS near the draw count, R-hat
# near 1.
independent_chains <- function(n_draws, d, n_chains = 4, seed = 1) {
  set.seed(seed)
  lapply(seq_len(n_chains), function(i) {
    sweep(matrix(rnorm(n_draws * d), n_draws, d), 2, seq_len(d) * 10, "+")
  })
}

param_names <- function(d) paste0("theta", seq_len(d))

# ---------------------------------------------------------------------------
test_that("ESS is per parameter and reflects the draws, not the dimension", {
  # The bug this pins: ess_bulk() handed a whole (draws x parameters) matrix
  # reduces it to one number, and with parameters at different locations that
  # number lands on the parameter count. On these chains the old code returned
  # 3.7 for d = 3 and 26.4 for d = 26, whatever the sampler had done.
  n_draws <- 1000

  for (d in c(3, 8, 26)) {
    conv <- compute_convergence(
      independent_chains(n_draws, d), param_names(d), total_runtime = 1
    )

    expect_equal(conv$n_params, d)
    expect_equal(conv$total_draws, n_draws * 4)
    expect_gt(conv$ess_bulk_median, 0.5 * conv$total_draws)
    expect_true(abs(conv$ess_bulk_median - d) > 100)
  }
})

# ---------------------------------------------------------------------------
test_that("ESS does not track the parameter count", {
  # The tell that exposed the bug in the committed results: ESS correlated
  # with dimension at r = 0.9991 across every model and algorithm.
  dims <- c(3, 4, 7, 8, 26)
  ess <- vapply(dims, function(d) {
    compute_convergence(
      independent_chains(1000, d), param_names(d), total_runtime = 1
    )$ess_bulk_median
  }, numeric(1))

  expect_lt(abs(cor(ess, dims)), 0.9)
  expect_true(all(ess > 500))
})

# ---------------------------------------------------------------------------
test_that("R-hat is near 1 when chains agree", {
  conv <- compute_convergence(independent_chains(1000, 5), param_names(5), 1)

  expect_lt(conv$rhat_max, 1.05)
  expect_equal(conv$n_params_unconverged, 0)
})

# ---------------------------------------------------------------------------
test_that("R-hat detects chains that never mixed", {
  # The bug this pins: R-hat compares between-chain against within-chain
  # variance. Computed inside the chain loop there is no between-chain term to
  # form, so four stuck chains looked no different from four healthy ones.
  set.seed(2)
  d <- 5
  stuck <- lapply(1:4, function(i) {
    matrix(rnorm(1000 * d), 1000, d) + i * 50   # each chain in its own place
  })

  conv <- compute_convergence(stuck, param_names(d), total_runtime = 1)

  expect_gt(conv$rhat_max, 2)
  expect_equal(conv$n_params_unconverged, d)
  expect_lt(conv$ess_bulk_median, 100)
})

# ---------------------------------------------------------------------------
test_that("a single chain warns rather than reporting a silent R-hat", {
  expect_warning(
    compute_convergence(independent_chains(500, 3, n_chains = 1), param_names(3), 1),
    "at least 2 chains"
  )
})

# ---------------------------------------------------------------------------
test_that("ESS per second divides by the runtime it was given", {
  chains <- independent_chains(1000, 4)

  fast <- compute_convergence(chains, param_names(4), total_runtime = 1)
  slow <- compute_convergence(chains, param_names(4), total_runtime = 10)

  expect_equal(fast$ess_bulk_median, slow$ess_bulk_median)
  expect_equal(fast$ess_per_sec_median / 10, slow$ess_per_sec_median, tolerance = 1e-8)
})

# ---------------------------------------------------------------------------
test_that("per-parameter diagnostics are returned, one row per parameter", {
  conv <- compute_convergence(independent_chains(500, 6), param_names(6), 1)

  expect_equal(nrow(conv$per_parameter), 6)
  expect_true(all(c("ess_bulk", "ess_tail", "rhat") %in% names(conv$per_parameter)))
  expect_equal(conv$per_parameter$variable, param_names(6))
})

# ---------------------------------------------------------------------------
test_that("accuracy is measured against the reference means", {
  set.seed(3)
  samples <- matrix(rnorm(2000 * 3), 2000, 3)
  reference <- c(0, 0, 0)

  acc <- compute_accuracy(samples, reference, runtime = 2.5)

  expect_lt(acc$rmse, 0.1)     # sample means of standard normals sit near 0
  expect_lt(acc$mae, 0.1)
  expect_equal(acc$runtime, 2.5)

  # A known offset must show up at its exact size.
  shifted <- compute_accuracy(samples + 5, reference, runtime = 1)
  expect_equal(shifted$rmse, 5, tolerance = 0.05)
})
