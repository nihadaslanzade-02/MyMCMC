# ============================================================================
# TARGET DENSITIES
# ============================================================================
# The Gaussian surrogate has a closed-form answer, so every claim below is
# checked against arithmetic rather than against a previous run. The
# bridgestan tests need a C++ toolchain and the BridgeStan sources and skip
# without them.

source("stan_targets.R")
source("adaptive_algorithms.R")

# A reference posterior with known moments and a real correlation, on a scale
# nothing like 1 - which is the case that broke the old fixed proposal.
fake_reference <- function(n = 20000, seed = 5) {
  set.seed(seed)
  mu <- c(500, -3, 0.02)
  L <- matrix(c(1, 0, 0,
                0.8, 0.6, 0,
                -0.3, 0.2, 0.93), 3, 3, byrow = TRUE)
  scale <- c(200, 1.5, 0.004)
  z <- matrix(rnorm(n * 3), n, 3)
  draws <- sweep(z %*% t(L), 2, scale, "*")
  draws <- sweep(draws, 2, mu, "+")
  colnames(draws) <- c("alpha", "beta", "sigma")
  draws
}

# ---------------------------------------------------------------------------
test_that("the surrogate reports the reference moments it was built from", {
  ref <- fake_reference()
  target <- gaussian_surrogate_target(ref)

  expect_equal(target$kind, "gaussian-surrogate")
  expect_equal(target$dimension, 3)
  expect_equal(target$param_names, c("alpha", "beta", "sigma"))
  expect_equal(target$reference_mean, colMeans(ref))
  expect_equal(target$reference_sd, apply(ref, 2, sd))

  # The two spaces coincide, so mapping draws is a no-op and the proposal
  # scale is the reference scale.
  expect_equal(target$sampling_scale, target$reference_sd)
  expect_equal(target$to_constrained(ref[1:10, ]), ref[1:10, ])
})

# ---------------------------------------------------------------------------
test_that("the surrogate's log density is the multivariate normal it claims", {
  ref <- fake_reference()
  target <- gaussian_surrogate_target(ref)

  mu <- colMeans(ref)
  sigma <- cov(ref) + diag(1e-6, 3)

  # Computed independently of the implementation, straight from the formula.
  manual <- function(x) {
    delta <- x - mu
    -0.5 * (determinant(sigma, logarithm = TRUE)$modulus[1] +
              as.numeric(t(delta) %*% solve(sigma) %*% delta) +
              3 * log(2 * pi))
  }

  set.seed(6)
  for (i in 1:5) {
    x <- mu + apply(ref, 2, sd) * rnorm(3)
    expect_equal(target$log_density(x), manual(x), tolerance = 1e-8)
  }

  # The mode sits at the mean, and nothing evaluable returns NaN.
  expect_gt(target$log_density(mu), target$log_density(mu + apply(ref, 2, sd)))
  expect_equal(target$log_density(c(NA, 0, 0)), -Inf)
  expect_equal(target$log_density(c(1, 2)), -Inf)     # wrong length
})

# ---------------------------------------------------------------------------
test_that("each chain starts at its own draw from the reference posterior", {
  # The old harness used init <- reference_mean + rnorm(d, 0, 0.1) whatever
  # the scale, which is simultaneously too small on a posterior with SDs in
  # the hundreds - all four chains land on the same point and R-hat goes
  # blind - and far too large on one with SDs near 0.001.
  ref <- fake_reference()
  target <- gaussian_surrogate_target(ref)

  set.seed(7)
  starts <- t(replicate(2000, target$init()))

  # Dispersed exactly like the posterior, on every coordinate whatever its
  # scale, and centred where the posterior is centred.
  expect_lt(max(abs(apply(starts, 2, sd) / target$reference_sd - 1)), 0.1)
  expect_lt(max(abs(colMeans(starts) - target$reference_mean) / target$reference_sd), 0.1)

  # Independent between chains: two starts must not be the same point.
  # Drawing 2000 rows with replacement from 20000 leaves about 1903 distinct
  # ones, so the bound allows for the collisions and still rules out a
  # constant start.
  expect_gt(length(unique(starts[, 1])), 1800)

  # And every start is a genuine draw, so a Stan model's constraints hold by
  # construction rather than by luck.
  expect_true(all(starts[1:50, 1] %in% ref[, 1]))
})

# ---------------------------------------------------------------------------
test_that("a sampler run against the surrogate recovers the reference", {
  ref <- fake_reference()
  target <- gaussian_surrogate_target(ref)

  set.seed(8)
  fit <- robust_adaptive_metropolis(target$log_density, target$init(),
                                    n_iterations = 40000, burn_in = 20000)

  recovered_mean <- colMeans(fit$samples)
  recovered_sd <- apply(fit$samples, 2, sd)

  # Within a quarter of a reference SD on the mean, and 20% on the spread.
  expect_lt(max(abs(recovered_mean - target$reference_mean) / target$reference_sd), 0.25)
  expect_lt(max(abs(recovered_sd / target$reference_sd - 1)), 0.2)
})

# ---------------------------------------------------------------------------
test_that("build_target asks for the Gaussian surrogate without a connection", {
  ref <- fake_reference()
  # pdb = NULL would fail immediately if this path tried to reach posteriordb.
  target <- build_target("whatever", ref, pdb = NULL, method = "gaussian")
  expect_equal(target$kind, "gaussian-surrogate")
})

# ---------------------------------------------------------------------------
test_that("auto falls back to the surrogate, bridgestan insists", {
  ref <- fake_reference()

  # "not-a-real-posterior" has no cached Stan program and there is no
  # connection to fetch one, so the bridgestan path fails here whether or not
  # the toolchain exists. That is the fallback under test, and it means this
  # reads the same on a machine with bridgestan and one without.
  auto <- suppressMessages(
    build_target("not-a-real-posterior", ref, pdb = NULL, method = "auto")
  )
  expect_equal(auto$kind, "gaussian-surrogate")

  # Asking for bridgestan specifically must fail loudly instead of quietly
  # handing back a Gaussian and letting it be reported as the real posterior.
  expect_error(
    suppressMessages(
      build_target("not-a-real-posterior", ref, pdb = NULL, method = "bridgestan")
    ),
    "bridgestan was required"
  )
})

# ---------------------------------------------------------------------------
test_that("bridgestan availability is reported with a reason, not a bare FALSE", {
  status <- bridgestan_available()

  expect_true(is.logical(status$ok))
  expect_true(is.character(status$why))
  # Whichever way it lands, an unavailable toolchain has to say what to do.
  if (!status$ok) expect_gt(nchar(status$why), 20)
})

# ---------------------------------------------------------------------------
test_that("a compiled Stan model round trips between the two spaces", {
  status <- bridgestan_available()
  skip_if(!status$ok, paste("bridgestan unavailable:", status$why))
  skip_if(!file.exists("stan_cache/arma-arma11.stan"),
          "no compiled model in stan_cache/; run the benchmark once first")

  ref <- readRDS("benchmark_config.rds")$models[["arma-arma11"]]$reference_draws
  target <- bridgestan_target("arma-arma11", ref, pdb = NULL)
  skip_if(is.null(target), "the cached Stan model could not be loaded")

  expect_equal(target$kind, "bridgestan")

  # Unconstrain a genuine draw and constrain it back: the parameters must
  # survive the trip, or every accuracy number is measured in the wrong space.
  one <- as.numeric(as.matrix(ref)[1, target$param_names])
  back <- target$to_constrained(matrix(target$to_unconstrained(one), nrow = 1))
  expect_equal(as.numeric(back), one, tolerance = 1e-8)

  # The real posterior must prefer real draws to arbitrary points.
  refm <- as.matrix(ref)[1:100, target$param_names, drop = FALSE]
  at_draws <- apply(refm, 1, function(r) target$log_density(target$to_unconstrained(r)))
  set.seed(9)
  at_random <- replicate(100, target$log_density(rnorm(target$dimension) * 5))
  expect_gt(median(at_draws), median(at_random))
})
