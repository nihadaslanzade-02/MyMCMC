# ============================================================================
# Benchmarking Adaptive MCMC with posteriordb
# Uses GitHub connection
# ============================================================================

# Required packages
required_packages <- c("posteriordb", "posterior", "jsonlite", "parallel", 
                       "ggplot2", "dplyr", "tidyr", "knitr")

# Install posteriordb if not available
if (!requireNamespace("posteriordb", quietly = TRUE)) {
  remotes::install_github("stan-dev/posteriordb-r", subdir = "rpackage")
}

# Load packages silently
invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}))
cat("✓ All required packages loaded\n\n")

# ============================================================================
# 1. POSTERIORDB SETUP
# ============================================================================

# Use GitHub connection to access reference posteriors
pdb <- posteriordb::pdb_github()
cat("✓ Posteriordb GitHub connection initialized\n\n")

# ============================================================================
# 2. DISCOVER AVAILABLE POSTERIORS
# ============================================================================

# Get all available posteriors
all_posteriors <- posteriordb::posterior_names(pdb)
cat("Total posteriors in database:", length(all_posteriors), "\n")

# Display first 20
cat("\nFirst 20 posteriors:\n")
print(head(all_posteriors, 20))

# ============================================================================
# 3. IDENTIFY POSTERIORS WITH REFERENCE POSTERIORS
# ============================================================================

cat("\n\nChecking which posteriors have reference posteriors...This will take some minutes.\n")

posteriors_with_refs <- list()
posteriors_without_refs <- c()

for (i in seq_along(all_posteriors)) {
  pname <- all_posteriors[i]
  
  if (i %% 10 == 0) {
    cat("Checked", i, "of", length(all_posteriors), "posteriors...\n")
  }
  
  tryCatch({
    po <- posteriordb::posterior(pname, pdb)
    
    # Try to load reference posterior draws directly
    has_ref <- tryCatch({
      draws <- posteriordb::reference_posterior_draws(po)
      !is.null(draws) && length(draws) > 0
    }, error = function(e) FALSE)
    
    if (has_ref) {
      # Get model info
      model_inf <- tryCatch({
        posteriordb::model_info(po)
      }, error = function(e) list(dimensions = list(parameters = NA)))
      
      posteriors_with_refs[[pname]] <- list(
        name = pname,
        dimension = model_inf$dimensions$parameters,
        has_reference = TRUE
      )
      cat("  ✓", pname, "(dim:", model_inf$dimensions$parameters, ")\n")
    } else {
      posteriors_without_refs <- c(posteriors_without_refs, pname)
    }
  }, error = function(e) {
    posteriors_without_refs <- c(posteriors_without_refs, pname)
  })
}

cat("\n\nSummary:\n")
cat("Posteriors with reference posteriors:", length(posteriors_with_refs), "\n")
cat("Posteriors without reference posteriors:", length(posteriors_without_refs), "\n")

# Save discovery results
saveRDS(list(
  with_refs = names(posteriors_with_refs),
  without_refs = posteriors_without_refs,
  details = posteriors_with_refs
), "posteriordb_discovery.rds")

# ============================================================================
# 4. CATEGORIZE BY DIMENSION - EXTRACT FROM REFERENCE DRAWS
# ============================================================================

cat("\n\nExtracting dimensions from reference draws...\n")
cat("(This is the only reliable source for dimensions)\n\n")

dims <- vapply(names(posteriors_with_refs), function(pname) {
  tryCatch({
    po <- posteriordb::posterior(pname, pdb)
    
    # Get dimension from actual reference draws
    draws <- posteriordb::reference_posterior_draws(po)
    draws_matrix <- posterior::as_draws_matrix(draws)
    
    # Get parameter names and count (excluding metadata columns)
    param_names <- colnames(draws_matrix)
    param_names <- param_names[!param_names %in% c(".chain", ".iteration", ".draw")]
    n_params <- length(param_names)
    
    if (n_params > 0) {
      cat("  ✓", pname, ":", n_params, "parameters\n")
    } else {
      cat("  ✗", pname, ": No parameters found\n")
    }
    
    return(as.integer(n_params))
  }, error = function(e) {
    cat("  ✗", pname, ": ERROR -", e$message, "\n")
    return(NA_integer_)
  })
}, FUN.VALUE = integer(1))

# Remove posteriors with unknown dimensions
valid_dims <- !is.na(dims) & dims > 0
dims_valid <- dims[valid_dims]
names_valid <- names(dims)[valid_dims]

cat("\n", rep("=", 70), "\n", sep = "")
cat("DIMENSION EXTRACTION SUMMARY\n")
cat(rep("=", 70), "\n", sep = "")
cat("Total posteriors with references:", length(posteriors_with_refs), "\n")
cat("Successfully extracted dimensions:", length(dims_valid), "\n")
cat("Failed to extract:", sum(!valid_dims), "\n")

if (length(dims_valid) > 0) {
  cat("\nDimension statistics:\n")
  cat("  Min:", min(dims_valid), "\n")
  cat("  Max:", max(dims_valid), "\n")
  cat("  Median:", median(dims_valid), "\n")
  cat("  Mean:", round(mean(dims_valid), 1), "\n")
  
  cat("\nDimension distribution:\n")
  dim_table <- table(cut(dims_valid, 
                         breaks = c(0, 5, 10, 20, 50, 100, Inf), 
                         labels = c("1-5", "6-10", "11-20", "21-50", "51-100", "100+")))
  print(dim_table)
}

# Categorize by dimension
low_dim <- names_valid[dims_valid >= 2 & dims_valid <= 5]
medium_dim <- names_valid[dims_valid > 5 & dims_valid <= 20]
high_dim <- names_valid[dims_valid > 20]

cat("\n", rep("=", 70), "\n", sep = "")
cat("CATEGORIZATION BY DIMENSION\n")
cat(rep("=", 70), "\n", sep = "")
cat("Low dimensional (2-5 params):", length(low_dim), "posteriors\n")
cat("Medium dimensional (6-20 params):", length(medium_dim), "posteriors\n")
cat("High dimensional (20+ params):", length(high_dim), "posteriors\n")

# Show examples with actual dimensions
if (length(low_dim) > 0) {
  cat("\n--- Low Dimensional Examples (up to 5) ---\n")
  for (name in head(low_dim, 5)) {
    cat(sprintf("  • %-50s (%2d params)\n", name, dims_valid[names_valid == name]))
  }
}

if (length(medium_dim) > 0) {
  cat("\n--- Medium Dimensional Examples (up to 5) ---\n")
  for (name in head(medium_dim, 5)) {
    cat(sprintf("  • %-50s (%2d params)\n", name, dims_valid[names_valid == name]))
  }
}

if (length(high_dim) > 0) {
  cat("\n--- High Dimensional Examples (up to 5) ---\n")
  for (name in head(high_dim, 5)) {
    cat(sprintf("  • %-50s (%2d params)\n", name, dims_valid[names_valid == name]))
  }
}

# Update posteriors_with_refs with correct dimensions
cat("\n\nUpdating posterior metadata with correct dimensions...\n")
for (pname in names_valid) {
  posteriors_with_refs[[pname]]$dimension <- dims_valid[names_valid == pname]
}
cat("✓ Updated", length(names_valid), "posterior entries\n")

# Save updated discovery results
saveRDS(list(
  with_refs = names(posteriors_with_refs),
  without_refs = posteriors_without_refs,
  details = posteriors_with_refs,
  dimensions = setNames(dims_valid, names_valid)
), "posteriordb_discovery.rds")
cat("✓ Saved to posteriordb_discovery.rds\n")
# ============================================================================
# 5. SELECT MODELS FOR BENCHMARKING
# ============================================================================

# Select diverse models based on availability
selected_models <- list(
  low_dim = head(low_dim, min(3, length(low_dim))),
  medium_dim = head(medium_dim, min(3, length(medium_dim))),
  high_dim = head(high_dim, min(3, length(high_dim)))
)

# Flatten list
all_selected_models <- unlist(selected_models)

cat("\n\nSelected models for benchmarking:\n")
for (cat_name in names(selected_models)) {
  cat("\n", cat_name, ":\n", sep = "")
  print(selected_models[[cat_name]])
}

# ============================================================================
# 6. LOAD REFERENCE POSTERIORS
# ============================================================================

load_reference_posterior <- function(posterior_name, pdb) {
  tryCatch({
    # Get posterior object
    po <- posteriordb::posterior(posterior_name, pdb)
    
    # Get model info
    model_info <- posteriordb::model_info(po)
    
    # Load reference draws
    ref_draws <- tryCatch({
      draws <- posteriordb::reference_posterior_draws(po)
      # Convert to matrix format for easier handling
      posterior::as_draws_matrix(draws)
    }, error = function(e) {
      message(sprintf("Could not load reference draws for %s: %s", 
                      posterior_name, e$message))
      NULL
    })
    
    if (is.null(ref_draws)) {
      message(sprintf("No reference posterior for %s", posterior_name))
      return(NULL)
    }
    
    return(list(
      name = posterior_name,
      dimension = model_info$dimensions$parameters,
      n_draws = nrow(ref_draws),
      n_params = ncol(ref_draws),
      reference_draws = ref_draws,
      model_info = model_info
    ))
    
  }, error = function(e) {
    message(sprintf("Error loading %s: %s", posterior_name, e$message))
    return(NULL)
  })
}

# ============================================================================
# 7. LOAD ALL SELECTED REFERENCE POSTERIORS
# ============================================================================

reference_posteriors <- list()

cat("\n\nLoading reference posteriors...\n")
cat("=" , rep("=", 60), "\n", sep = "")

for (model in all_selected_models) {
  cat("\nLoading:", model, "\n")
  ref_post <- load_reference_posterior(model, pdb)
  
  if (!is.null(ref_post)) {
    reference_posteriors[[model]] <- ref_post
    cat("  ✓ SUCCESS:", ref_post$n_draws, "draws ×", 
        ref_post$n_params, "parameters\n")
  } else {
    cat("  ✗ FAILED\n")
  }
}

cat("\n", rep("=", 60), "\n", sep = "")
cat("Successfully loaded:", length(reference_posteriors), 
    "of", length(all_selected_models), "reference posteriors\n")

# ============================================================================
# 8. BENCHMARK CONFIGURATION
# ============================================================================

benchmark_config <- list(
  # Algorithm settings
  n_iterations = 10000,
  burn_in = 1000,
  n_chains = 4,
  
  # Adaptive algorithm settings
  am_config = list(
    initial_cov_scale = 0.1,
    adaptation_start = 100,
    target_acceptance = 0.234
  ),
  
  ram_config = list(
    initial_scale = 1.0,
    target_acceptance = 0.234,
    gamma = 0.6
  ),
  
  rwm_config = list(
    proposal_sd = 2.38
  ),
  
  # Metrics to compute
  metrics = c("ess", "rhat", "rmse", "acceptance_rate", "time"),
  
  # Thinning
  thin = 1,
  
  # Random seed
  seed = 42
)

# ============================================================================
# 9. SAVE CONFIGURATION
# ============================================================================

saveRDS(list(
  models = reference_posteriors,
  config = benchmark_config,
  all_available = names(posteriors_with_refs),
  selected = all_selected_models,
  dimensions = list(
    low = low_dim,
    medium = medium_dim,
    high = high_dim
  ),
  timestamp = Sys.time()
), file = "benchmark_config.rds")

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================

cat("\n\n")
cat("=" , rep("=", 70), "\n", sep = "")
cat("BENCHMARK CONFIGURATION COMPLETE\n")
cat("=" , rep("=", 70), "\n", sep = "")

cat("\nDatabase Statistics:\n")
cat("  Total posteriors:", length(all_posteriors), "\n")
cat("  With reference posteriors:", length(posteriors_with_refs), "\n")
cat("  Without reference posteriors:", length(posteriors_without_refs), "\n")

cat("\nSelected for Benchmarking:\n")
cat("  Low dimensional (2-5):", length(selected_models$low_dim), "models\n")
cat("  Medium dimensional (6-20):", length(selected_models$medium_dim), "models\n")
cat("  High dimensional (20+):", length(selected_models$high_dim), "models\n")
cat("  TOTAL:", length(all_selected_models), "models\n")

cat("\nSuccessfully Loaded:\n")
cat("  Reference posteriors:", length(reference_posteriors), "\n")

cat("\nConfiguration saved to: benchmark_config.rds\n")
cat("Discovery results saved to: posteriordb_discovery.rds\n")

cat("\n", rep("=", 70), "\n", sep = "")
cat("Ready for benchmarking!\n")
cat(rep("=", 70), "\n", sep = "")

