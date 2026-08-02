# ============================================================================
# SAMPLERS
# ============================================================================
# Targets with known answers, so a failure means the sampler is wrong rather
# than that the expected value was read off a previous run.

source("adaptive_algorithms.R")
source("metropolis_hastings.R")
source("basic_monte_carlo.R")

# Standard normal in d dimensions, up to a constant.
std_normal <- function(d) function(x) -0.5 * sum(x^2)

# ---------------------------------------------------------------------------
test_that("every adaptive sampler recovers a standard normal", {
  set.seed(11)
  d <- 3

  for (nm in c("AM", "RAM", "RWM")) {
    algo <- switch(nm,
      AM  = adaptive_metropolis,
      RAM = robust_adaptive_metropolis,
      RWM = random_walk_baseline
    )

    fit <- algo(std_normal(d), initial_value = rep(0, d),
                n_iterations = 20000, burn_in = 5000)

    expect_equal(ncol(fit$samples), d)
    expect_equal(nrow(fit$samples), 15000)
    expect_true(all(is.finite(fit$samples)))

    # Posterior mean is 0 and SD is 1 in every coordinate.
    expect_lt(max(abs(colMeans(fit$samples))), 0.35)
    expect_lt(max(abs(apply(fit$samples, 2, sd) - 1)), 0.35)
  }
})

# ---------------------------------------------------------------------------
test_that("RAM drives acceptance towards its target", {
  # The one result the committed benchmark supports: the Robbins-Monro scale
  # update is a control loop and it holds its setpoint.
  set.seed(12)

  rates <- vapply(c(3, 5, 8), function(d) {
    robust_adaptive_metropolis(
      std_normal(d), initial_value = rep(0, d),
      n_iterations = 20000, burn_in = 5000, target_acceptance = 0.234
    )$acceptance_rate
  }, numeric(1))

  expect_true(all(abs(rates - 0.234) < 0.1))
  expect_lt(sd(rates), 0.06)
})

# ---------------------------------------------------------------------------
test_that("a custom acceptance target is honoured, not hardcoded", {
  set.seed(13)

  low <- robust_adaptive_metropolis(std_normal(3), rep(0, 3), 20000, 5000,
                                    target_acceptance = 0.15)$acceptance_rate
  high <- robust_adaptive_metropolis(std_normal(3), rep(0, 3), 20000, 5000,
                                     target_acceptance = 0.45)$acceptance_rate

  expect_lt(low, high)
  expect_lt(abs(low - 0.15), 0.1)
  expect_lt(abs(high - 0.45), 0.1)
})

# ---------------------------------------------------------------------------
test_that("samplers return the full chain alongside post-burn-in samples", {
  set.seed(14)
  fit <- adaptive_metropolis(std_normal(2), c(0, 0), n_iterations = 3000, burn_in = 1000)

  expect_equal(nrow(fit$full_chain), 3000)
  expect_equal(nrow(fit$samples), 2000)
  # The retained samples must be the tail of the chain, not a copy of it.
  expect_equal(fit$samples, fit$full_chain[1001:3000, , drop = FALSE])
})

# ---------------------------------------------------------------------------
test_that("acceptance is decided on the log scale", {
  # A 26-parameter target puts log-density differences well past the point
  # where exp() underflows. A sampler comparing raw densities would reject
  # everything and freeze; one working on the log scale keeps moving.
  set.seed(15)
  d <- 26

  fit <- robust_adaptive_metropolis(
    function(x) -0.5 * sum(x^2) * 1000,   # very peaked, huge log differences
    initial_value = rep(0, d), n_iterations = 5000, burn_in = 1000
  )

  expect_gt(fit$acceptance_rate, 0)
  expect_true(all(is.finite(fit$samples)))
})

# ---------------------------------------------------------------------------
test_that("Monte Carlo integration reports an error bar with its estimate", {
  set.seed(16)

  # Integral of x^2 over [0, 1] is 1/3.
  est <- basic_monte_carlo(function(x) sum(x^2), lower_bounds = 0,
                           upper_bounds = 1, N = 100000)

  expect_true(all(c("estimate", "standard_error") %in% names(est)))
  expect_lt(abs(est$estimate - 1 / 3), 0.01)
  expect_gt(est$standard_error, 0)
  # The true value should sit inside a few standard errors.
  expect_lt(abs(est$estimate - 1 / 3), 5 * est$standard_error)
})

# ---------------------------------------------------------------------------
test_that("Monte Carlo integration scales by the volume of the domain", {
  set.seed(17)

  # Integral of 1 over the unit square is 1; over [0,2]^2 it is 4.
  unit <- basic_monte_carlo(function(x) 1, c(0, 0), c(1, 1), N = 20000)
  bigger <- basic_monte_carlo(function(x) 1, c(0, 0), c(2, 2), N = 20000)

  expect_equal(unit$estimate, 1, tolerance = 1e-8)
  expect_equal(bigger$estimate, 4, tolerance = 1e-8)
})

# ---------------------------------------------------------------------------
test_that("Random Walk Metropolis recovers a standard normal", {
  set.seed(18)

  fit <- random_walk_metropolis(std_normal(2), proposal_sd = 2.4,
                                initial_value = c(0, 0),
                                n_iterations = 20000, burn_in = 5000)

  expect_lt(max(abs(colMeans(fit$samples))), 0.35)
  expect_gt(fit$acceptance_rate, 0.1)
})

# ---------------------------------------------------------------------------
test_that("the two families read n_iterations differently, on purpose", {
  # Worth pinning, because the same argument name means two things here and
  # the difference is silent. metropolis_hastings.R follows the thesis
  # algorithms, where n_iterations is the number of draws you asked to keep
  # and burn_in is run on top of it. adaptive_algorithms.R treats
  # n_iterations as the total length of the chain, with burn_in taken out of
  # it, which is what the benchmark is calibrated against.
  #
  # Neither is wrong, but a test that fails if someone quietly unifies them
  # is cheaper than discovering it through a draw count that halved.
  set.seed(19)
  target <- std_normal(2)

  mh <- random_walk_metropolis(target, proposal_sd = 2.4, initial_value = c(0, 0),
                               n_iterations = 4000, burn_in = 1000)
  am <- adaptive_metropolis(target, initial_value = c(0, 0),
                            n_iterations = 4000, burn_in = 1000)

  expect_equal(nrow(mh$full_chain), 5000)   # 4000 requested + 1000 burn-in
  expect_equal(nrow(mh$samples), 4000)

  expect_equal(nrow(am$full_chain), 4000)   # 4000 total, burn-in included
  expect_equal(nrow(am$samples), 3000)
})
