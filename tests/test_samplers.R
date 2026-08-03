# ============================================================================
# SAMPLERS
# ============================================================================
# Targets with known answers, so a failure means the sampler is wrong rather
# than that the expected value was read off a previous run.

source("adaptive_algorithms.R")
source("metropolis_hastings.R")
source("basic_monte_carlo.R")
source("generic_mcmc.R")
source("hmc_leapfrog.R")

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
test_that("every sampler reads n_iterations as the total chain length", {
  # The two families used to disagree. metropolis_hastings.R and
  # generic_mcmc.R ran burn_in steps on top of n_iterations, so the same
  # arguments produced a longer chain there than in adaptive_algorithms.R.
  # The difference was silent and it broke the benchmark's timing columns:
  # two algorithms given identical arguments did different amounts of work.
  #
  # Now every sampler runs exactly n_iterations steps and returns
  # n_iterations - burn_in of them.
  set.seed(19)
  target <- std_normal(2)
  grad <- function(q) -q

  fits <- list(
    RWM_MH = random_walk_metropolis(target, proposal_sd = 2.4, initial_value = c(0, 0),
                                    n_iterations = 4000, burn_in = 1000),
    MH = metropolis_hastings(target,
                             proposal_sample = function(x) x + rnorm(length(x), 0, 2.4),
                             proposal_log_density = function(a, b) 0,
                             initial_value = c(0, 0),
                             n_iterations = 4000, burn_in = 1000),
    AM = adaptive_metropolis(target, initial_value = c(0, 0),
                             n_iterations = 4000, burn_in = 1000),
    RAM = robust_adaptive_metropolis(target, initial_value = c(0, 0),
                                     n_iterations = 4000, burn_in = 1000),
    RWM_base = random_walk_baseline(target, initial_value = c(0, 0),
                                    n_iterations = 4000, burn_in = 1000),
    HMC = quietly(hamiltonian_monte_carlo(target, grad, step_size = 0.2,
                                          n_leapfrog_steps = 5, initial_value = c(0, 0),
                                          n_iterations = 4000, burn_in = 1000)),
    GENERIC = generic_mcmc(target,
                           transition_kernel = function(x) {
                             y <- x + rnorm(length(x), 0, 2.4)
                             list(state = y, acceptance_prob = exp(min(0, target(y) - target(x))))
                           },
                           initial_value = c(0, 0),
                           n_iterations = 4000, burn_in = 1000)
  )

  for (nm in names(fits)) {
    expect_equal(nrow(fits[[nm]]$full_chain), 4000, info = nm)
    expect_equal(nrow(fits[[nm]]$samples), 3000, info = nm)
  }
})

# ---------------------------------------------------------------------------
test_that("AM's recursive covariance matches recomputing it outright", {
  # AM used to call cov(chain[1:(t-1), ]) at every adaptation, which costs
  # O(t*d^2) and so gets more expensive the longer the chain runs. Welford's
  # recursion is O(d^2) per iteration regardless. The two must agree, or the
  # speedup has quietly changed the algorithm.
  set.seed(21)
  d <- 4

  fit <- adaptive_metropolis(std_normal(d), initial_value = rep(0, d),
                             n_iterations = 1000, burn_in = 500,
                             adapt_interval = 50, adapt_start = 100,
                             adapt_stop = 1000)

  # The last adaptation fires at t = 1000 over chain[1:999, ].
  recomputed <- cov(fit$full_chain[1:999, ]) + diag(1e-6, d)
  expect_equal(fit$final_cov, recomputed, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
test_that("adaptation runs during burn-in and freezes before the kept draws", {
  # The old schedule was `t > burn_in`: fixed proposal for the whole of
  # burn-in, then adapting for every retained draw. That is backwards on both
  # counts. ESS and R-hat are defined for a fixed kernel, so the retained
  # stretch is exactly where adaptation must not still be running.
  set.seed(22)
  target <- std_normal(3)

  frozen <- adaptive_metropolis(target, rep(0, 3), n_iterations = 2000,
                                burn_in = 500, adapt_interval = 50,
                                adapt_start = 100)
  # t in {100, 150, ..., 500}
  expect_equal(frozen$n_adaptations, 9)

  # Haario's original algorithm adapts forever; one argument gets it back.
  forever <- adaptive_metropolis(target, rep(0, 3), n_iterations = 2000,
                                 burn_in = 500, adapt_interval = 50,
                                 adapt_start = 100, adapt_stop = 2000)
  expect_equal(forever$n_adaptations, 39)
})

# ---------------------------------------------------------------------------
test_that("RAM's shape window grows with the chain", {
  # It was hardcoded to 500, so a longer run gave the scale adaptation more
  # iterations while the shape estimate kept reading the same short tail.
  set.seed(23)
  target <- std_normal(2)

  short <- robust_adaptive_metropolis(target, c(0, 0), n_iterations = 4000,
                                      burn_in = 1000)
  long <- robust_adaptive_metropolis(target, c(0, 0), n_iterations = 40000,
                                     burn_in = 1000)

  expect_equal(short$shape_window, 500)    # floor, for short runs
  expect_equal(long$shape_window, 2000)    # n_iterations / 20
})

# ---------------------------------------------------------------------------
test_that("the baseline's proposal follows the scale it is given", {
  # The scale was the absolute constant (2.38 / sqrt(d)) * 0.1, which is a
  # sensible step only for a target that happens to be that wide. The
  # benchmark's posteriors span seven orders of magnitude in per-parameter SD,
  # so the baseline's acceptance rate was reporting that constant rather than
  # anything about random walk Metropolis.
  set.seed(24)
  sds <- c(1000, 0.001)
  target <- function(x) -0.5 * sum((x / sds)^2)

  informed <- random_walk_baseline(target, c(0, 0), n_iterations = 20000,
                                   burn_in = 5000, proposal_scale = sds)
  blind <- random_walk_baseline(target, c(0, 0), n_iterations = 20000,
                                burn_in = 5000)

  # Given the marginal scales it mixes and recovers them.
  expect_gt(informed$acceptance_rate, 0.15)
  recovered <- apply(informed$samples, 2, sd)
  expect_lt(max(abs(log10(recovered / sds))), 0.3)

  # Given none, one step size cannot fit both coordinates at once.
  expect_lt(blind$acceptance_rate, 0.05)

  expect_error(random_walk_baseline(target, c(0, 0), n_iterations = 100,
                                    burn_in = 10, proposal_scale = c(1, 1, 1)),
               "one entry per parameter")
})

# ---------------------------------------------------------------------------
test_that("a burn-in that swallows the chain is rejected, not silently wrong", {
  # chain[(burn_in + 1):n_iterations, ] with burn_in >= n_iterations does not
  # error in R: 1001:1000 counts backwards and returns two rows, one of them
  # out of bounds and therefore NA. Every sampler used to accept this and
  # hand back that garbage.
  set.seed(20)
  target <- std_normal(2)

  expect_error(adaptive_metropolis(target, c(0, 0), n_iterations = 1000, burn_in = 1000),
               "0 <= burn_in < n_iterations")
  expect_error(robust_adaptive_metropolis(target, c(0, 0), n_iterations = 500, burn_in = 900),
               "0 <= burn_in < n_iterations")
  expect_error(random_walk_baseline(target, c(0, 0), n_iterations = 500, burn_in = 900),
               "0 <= burn_in < n_iterations")
  expect_error(random_walk_metropolis(target, 2.4, c(0, 0), n_iterations = 100, burn_in = 100),
               "0 <= burn_in < n_iterations")
  expect_error(adaptive_metropolis(target, c(0, 0), n_iterations = 1, burn_in = 0),
               "n_iterations >= 2")
})
