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

# ---------------------------------------------------------------------------
test_that("normalised accuracy is comparable across models on different scales", {
  # The bug this pins: RMSE was never scale-normalised, so averaging it over
  # models reported whichever model had the largest units. In the first run
  # that was earnings-earn_height, whose parameters are in dollars with
  # standard deviations up to 9668; every other model sat below 0.5, and the
  # 61.8 mean RMSE reported for RAM was essentially that one model.
  reference <- c(0, 0, 0)
  dollars <- c(9667.9, 144.2, 385.7)     # earnings-earn_height
  logs <- c(0.1962, 0.0029, 0.0080)      # earnings-log10earn_height

  # A chain whose posterior mean sits exactly half a reference SD high in
  # every coordinate, on each scale.
  off_by_half <- function(sds) matrix(rep(0.5 * sds, each = 100), nrow = 100)

  a <- compute_accuracy(off_by_half(dollars), reference, 1, reference_sds = dollars)
  b <- compute_accuracy(off_by_half(logs), reference, 1, reference_sds = logs)

  # Same error, so the same score - which is the whole point.
  expect_equal(a$rmse_normalised, 0.5, tolerance = 1e-10)
  expect_equal(b$rmse_normalised, 0.5, tolerance = 1e-10)
  expect_equal(a$mae_normalised, 0.5, tolerance = 1e-10)

  # On the raw scale the identical error differs by four orders of magnitude,
  # which is what made the cross-model average meaningless.
  expect_gt(a$rmse / b$rmse, 1e4)
})

# ---------------------------------------------------------------------------
test_that("a parameter with no spread is excluded rather than returning Inf", {
  samples <- matrix(rep(c(1, 1, 1), each = 50), nrow = 50)

  expect_warning(
    compute_accuracy(samples, c(0, 0, 0), 1, reference_sds = c(2, 0, 4)),
    "zero or non-finite reference SD"
  )

  acc <- suppressWarnings(
    compute_accuracy(samples, c(0, 0, 0), 1, reference_sds = c(2, 0, 4))
  )
  # Errors of 1/2 and 1/4 SDs on the two usable parameters.
  expect_equal(acc$rmse_normalised, sqrt(mean(c(0.5, 0.25)^2)), tolerance = 1e-10)
  expect_true(is.finite(acc$rmse_normalised))
})

# ---------------------------------------------------------------------------
test_that("normalised accuracy is absent, not wrong, when no SDs are supplied", {
  samples <- matrix(rnorm(300), 100, 3)
  acc <- compute_accuracy(samples, c(0, 0, 0), runtime = 1)

  expect_true(is.na(acc$rmse_normalised))
  expect_true(is.finite(acc$rmse))
})
