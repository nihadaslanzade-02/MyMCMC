# ============================================================================
# TARGET DENSITIES
# ============================================================================
# Two ways of giving the samplers something to sample, behind one interface.
#
# The Gaussian surrogate fits a multivariate normal to the reference draws and
# samples that. Its ground truth is exact - the true mean and covariance are
# known in closed form, so RMSE is an exact accuracy measure rather than one
# finite sample compared against another - and the dimension and correlation
# structure still come from a real posterior. What it cannot test is
# non-Gaussian geometry: funnels, multimodality, heavy tails. That is
# precisely where adaptive methods are supposed to earn their keep, so a
# benchmark that only ever sees Gaussians is a statement about adaptive MCMC
# on Gaussian targets and not about adaptive MCMC.
#
# The bridgestan target compiles the posterior's actual Stan program and calls
# its log density. That removes the caveat, at the cost of needing a C++
# toolchain and the BridgeStan sources, and of sampling in the unconstrained
# space that Stan transforms to. Draws come back through param_constrain()
# before anything is measured, so both paths report on the same scale and the
# accuracy and convergence columns stay comparable.
#
# A target is a list with:
#
#   kind            "gaussian-surrogate" or "bridgestan"
#   log_density     function(theta) -> numeric(1), on the SAMPLING space
#   dimension       length of theta, i.e. the sampling space's dimension
#   param_names     names of the constrained parameters
#   reference_mean  posterior mean of each constrained parameter
#   reference_sd    posterior SD of each constrained parameter
#   sampling_scale  per-coordinate SD in the SAMPLING space, which is what a
#                   non-adaptive proposal has to be scaled by
#   condition       condition number of the reference posterior's CORRELATION
#                   matrix - see below
#   to_constrained  function(draws matrix) -> matrix on the constrained scale
#   to_unconstrained function(constrained vector) -> sampling-space vector
#   init            function() -> one starting value, drawn independently
#
# ----------------------------------------------------------------------------
# Where the chains start
# ----------------------------------------------------------------------------
# Both targets start each chain at an independent draw from the reference
# posterior - a randomly chosen row, mapped into the sampling space.
#
# The old harness used reference_mean + rnorm(d, 0, 0.1) whatever the model's
# scale. On a posterior whose parameters have standard deviations in the
# thousands that puts all four chains at effectively the same point, and R-hat
# cannot detect non-mixing among chains that were never apart. On one whose
# parameters sit near 0.001 the same 0.1 is a wild displacement instead. On
# diamonds-diamonds it was the second: over 40,000 iterations Adaptive
# Metropolis reached an acceptance rate of 0.019, R-hat of 3.21 and a median
# ESS of 6, against 0.254, 1.03 and 912 from a reference draw.
#
# Scaling the perturbation by each parameter's own SD is not enough either.
# These posteriors are strongly correlated - earnings-earn_height has a
# condition number of 1.4 million - so a displacement that is two SDs on every
# axis at once lands far out in the joint density even though it looks modest
# one coordinate at a time. Sampling whole reference draws respects the
# correlation structure for free, and for a Stan model it is also the only
# construction guaranteed to satisfy every declared constraint.
#
# What this costs: chains start dispersed like the posterior rather than
# overdispersed relative to it, so this measures mixing from a good start and
# not recovery from a bad one. R-hat still catches a stuck sampler, because
# four chains frozen at four different draws have almost no within-chain
# variance and R-hat goes up, not down.
#
# For the surrogate the two spaces coincide and to_constrained is the
# identity. For bridgestan they do not, and dimension is the unconstrained
# count while param_names describes the constrained output.
#
# ----------------------------------------------------------------------------
# Why the condition number is recorded
# ----------------------------------------------------------------------------
# The correlation matrix, not the covariance. The non-adaptive baseline is
# given each parameter's marginal scale, so scale is not what it is missing;
# what it cannot represent is the correlation between parameters. The
# condition number of the correlation matrix is exactly how much of that is
# left, and it is the variable that predicts what adaptation is worth. Across
# these posteriors it has nothing to do with dimension: arma-arma11 has 4
# parameters and a correlation condition number of 1.7, while
# earnings-earn_height has 3 and 1,214.
reference_condition <- function(ref) {
  if (ncol(ref) < 2) return(1)
  suppressWarnings(tryCatch(kappa(cor(ref), exact = TRUE),
                            error = function(e) NA_real_))
}

# ============================================================================
# GAUSSIAN SURROGATE
# ============================================================================
gaussian_surrogate_target <- function(reference_draws) {
  ref <- as.matrix(reference_draws)
  d <- ncol(ref)

  mu <- colMeans(ref)
  sds <- apply(ref, 2, sd)

  # Small ridge so the covariance stays invertible and usable as a proposal.
  sigma <- cov(ref) + diag(1e-6, d)

  sigma_inv <- tryCatch(solve(sigma), error = function(e) {
    warning("Singular reference covariance; using a diagonal approximation")
    diag(1 / (diag(sigma) + 1e-6))
  })
  log_det <- determinant(sigma, logarithm = TRUE)$modulus[1]

  list(
    kind = "gaussian-surrogate",
    log_density = function(theta) {
      # Coerced, not just length checked. A 1-row matrix - which is what
      # indexing a draws_matrix hands back - passes a length test and then
      # turns the quadratic form into a non-conformable multiply.
      theta <- as.numeric(theta)
      if (length(theta) != d || any(!is.finite(theta))) return(-Inf)
      delta <- theta - mu
      -0.5 * (log_det + sum(delta * (sigma_inv %*% delta)) + d * log(2 * pi))
    },
    dimension = d,
    param_names = colnames(ref),
    reference_mean = mu,
    reference_sd = sds,
    sampling_scale = sds,          # the two spaces coincide here
    condition = reference_condition(ref),
    to_constrained = function(draws) draws,
    to_unconstrained = function(theta) as.numeric(theta),
    init = function() as.numeric(ref[sample.int(nrow(ref), 1L), ])
  )
}

# ============================================================================
# BRIDGESTAN
# ============================================================================
# Compiles the posterior's Stan program to a shared library and wraps its log
# density. Returns NULL, with a reason, whenever that is not possible on this
# machine, so the caller can fall back rather than abort the sweep.

# BridgeStan names container elements the way Stan's CSV output does, with dot
# separated indices: beta.1, beta.2, Omega.1.2. The reference draws come
# through the posterior package, which uses brackets: beta[1], beta[2],
# Omega[1,2]. Same parameters, two spellings, and comparing them raw makes
# every model with a vector in it look like a mismatch.
bridgestan_to_posterior_names <- function(x) {
  vapply(x, function(nm) {
    parts <- strsplit(nm, ".", fixed = TRUE)[[1]]
    if (length(parts) == 1) return(nm)
    idx <- parts[-1]
    # Anything that is not a plain index is left alone rather than mangled.
    if (!all(grepl("^[0-9]+$", idx))) return(nm)
    paste0(parts[1], "[", paste(idx, collapse = ","), "]")
  }, character(1), USE.NAMES = FALSE)
}

bridgestan_available <- function() {
  if (!requireNamespace("bridgestan", quietly = TRUE)) {
    return(list(ok = FALSE, why = "the bridgestan R package is not installed"))
  }
  if (!nzchar(Sys.which("make"))) {
    return(list(ok = FALSE, why = "no `make` on PATH (install Rtools and add its usr/bin)"))
  }
  path <- tryCatch(bridgestan:::get_bridgestan_path(download = FALSE),
                   error = function(e) NULL)
  if (is.null(path) || !dir.exists(path)) {
    return(list(ok = FALSE,
                why = "the BridgeStan sources are not downloaded; run bridgestan:::get_bridgestan_path(download = TRUE)"))
  }
  list(ok = TRUE, why = "", path = path)
}

bridgestan_target <- function(posterior_name, reference_draws, pdb,
                              cache_dir = "stan_cache", seed = 1) {
  ready <- bridgestan_available()
  if (!ready$ok) {
    message("  bridgestan unavailable: ", ready$why)
    return(NULL)
  }

  ref <- as.matrix(reference_draws)

  built <- tryCatch({
    dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

    # posteriordb hands back paths inside a temporary directory, so the model
    # and its data are copied into the cache first. Compiling writes the
    # shared library next to the .stan file, and keeping all three here means
    # a re-run reuses the compiled model rather than rebuilding it - which
    # matters, because the first build of the Stan library takes minutes.
    safe <- gsub("[^A-Za-z0-9_.-]", "_", posterior_name)
    stan_file <- file.path(cache_dir, paste0(safe, ".stan"))
    data_file <- file.path(cache_dir, paste0(safe, ".data.json"))

    # posteriordb is only consulted for what the cache does not already hold,
    # so a repeat run needs no network at all.
    if (!file.exists(stan_file) || !file.exists(data_file)) {
      if (is.null(pdb)) {
        stop("need a posteriordb connection to fetch the Stan program for ",
             posterior_name, "; nothing cached in ", cache_dir)
      }
      po <- posteriordb::posterior(posterior_name, pdb)
      if (!file.exists(stan_file)) {
        file.copy(posteriordb::stan_code_file_path(po), stan_file, overwrite = TRUE)
      }
      if (!file.exists(data_file)) {
        file.copy(posteriordb::stan_data_file_path(po), data_file, overwrite = TRUE)
      }
    }

    lib <- bridgestan::compile_model(stan_file)
    model <- bridgestan::StanModel$new(lib, data_file, seed)
    list(model = model, lib = lib)
  }, error = function(e) {
    message("  bridgestan could not build ", posterior_name, ": ", conditionMessage(e))
    NULL
  })

  if (is.null(built)) return(NULL)
  model <- built$model

  d_unc <- model$param_unc_num()
  constrained_names <- bridgestan_to_posterior_names(
    model$param_names(include_tp = FALSE, include_gq = FALSE)
  )

  # The reference draws and the Stan program must describe the same
  # parameters, or the accuracy columns would be comparing different things.
  if (!setequal(constrained_names, colnames(ref))) {
    only_stan <- setdiff(constrained_names, colnames(ref))
    only_ref <- setdiff(colnames(ref), constrained_names)
    message("  bridgestan parameter names for ", posterior_name,
            " do not match the reference draws; falling back",
            if (length(only_stan)) paste0("\n    only in the Stan program: ",
                                          paste(head(only_stan, 6), collapse = ", ")),
            if (length(only_ref)) paste0("\n    only in the reference: ",
                                         paste(head(only_ref, 6), collapse = ", ")))
    return(NULL)
  }
  ref <- ref[, constrained_names, drop = FALSE]

  # The spread of the reference posterior mapped into the unconstrained space,
  # which is the space the samplers actually move in. A subsample is enough
  # for a scale and keeps the setup cheap.
  unc_sample <- t(apply(ref[seq_len(min(1000L, nrow(ref))), , drop = FALSE], 1,
                        function(row) model$param_unconstrain(as.numeric(row))))
  unc_sd <- apply(unc_sample, 2, sd)

  list(
    kind = "bridgestan",
    log_density = function(theta) {
      theta <- as.numeric(theta)
      if (length(theta) != d_unc || any(!is.finite(theta))) return(-Inf)
      # A Stan program returns -Inf, or throws, on parameters its constraints
      # reject. Either way the proposal must simply be rejected.
      out <- tryCatch(model$log_density(theta, propto = TRUE, jacobian = TRUE),
                      error = function(e) -Inf)
      if (!is.finite(out)) -Inf else out
    },
    dimension = d_unc,
    param_names = constrained_names,
    reference_mean = colMeans(ref),
    reference_sd = apply(ref, 2, sd),
    sampling_scale = unc_sd,
    # Measured in the unconstrained space, because that is where the sampler
    # moves and where the baseline's diagonal proposal has to cope. It is also
    # the only one that is finite here: a simplex makes the constrained
    # covariance exactly rank deficient, which is why the two bball models
    # report a constrained condition number around 1e17 and 8 parameters
    # against 6 unconstrained ones.
    condition = reference_condition(unc_sample),
    to_constrained = function(draws) {
      out <- t(apply(draws, 1, function(row) {
        model$param_constrain(row, include_tp = FALSE, include_gq = FALSE)
      }))
      colnames(out) <- constrained_names
      out
    },
    to_unconstrained = function(theta) {
      model$param_unconstrain(as.numeric(theta))
    },
    # The same rule as the surrogate: one independent reference draw per
    # chain, mapped into the space the sampler moves in. Unconstraining a real
    # draw also guarantees the start satisfies every constraint the model
    # declares, which a perturbed one does not.
    init = function() {
      model$param_unconstrain(as.numeric(ref[sample.int(nrow(ref), 1L), ]))
    }
  )
}

# ============================================================================
# DISPATCH
# ============================================================================
# `method` is "gaussian" for the surrogate, "bridgestan" to insist on the real
# thing and fail if it is unavailable, or "auto" to prefer bridgestan and fall
# back with a message. The kind that was actually used is recorded on every
# result, so a summary can never silently mix the two.
build_target <- function(posterior_name, reference_draws, pdb,
                         method = c("auto", "gaussian", "bridgestan"),
                         cache_dir = "stan_cache") {
  method <- match.arg(method)

  if (method == "gaussian") {
    return(gaussian_surrogate_target(reference_draws))
  }

  target <- bridgestan_target(posterior_name, reference_draws, pdb, cache_dir)

  if (is.null(target)) {
    if (method == "bridgestan") {
      stop("bridgestan was required for ", posterior_name, " but is unavailable")
    }
    message("  falling back to the Gaussian surrogate for ", posterior_name)
    return(gaussian_surrogate_target(reference_draws))
  }

  target
}
