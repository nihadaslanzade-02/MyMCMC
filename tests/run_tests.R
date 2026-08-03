# ============================================================================
# TEST RUNNER
# ============================================================================
# A small testthat-shaped harness built on base R. The core samplers in this
# repository deliberately depend on nothing beyond base R and MASS, and the
# test suite keeps that property: the only package it needs is `posterior`,
# which the diagnostics under test already require.
#
# Run from the repository root:
#
#   Rscript tests/run_tests.R
#
# Exits non-zero if anything fails, so CI picks it up.

TESTS <- new.env()
TESTS$passed <- 0L
TESTS$failed <- 0L
TESTS$failures <- character(0)
TESTS$current <- NA_character_

# ---------------------------------------------------------------------------
# Assertions, named to match testthat so the tests read the same either way
# ---------------------------------------------------------------------------

fail <- function(msg) stop(structure(
  class = c("expectation_failure", "error", "condition"),
  list(message = msg, call = NULL)
))

expect_true <- function(object, info = NULL) {
  if (!isTRUE(all(object))) {
    fail(paste0("expected TRUE, got ", paste(format(object), collapse = ", "),
                if (!is.null(info)) paste0(" (", info, ")")))
  }
  invisible(TRUE)
}

expect_equal <- function(object, expected, tolerance = 1e-8, info = NULL) {
  cmp <- all.equal(object, expected, tolerance = tolerance)
  if (!isTRUE(cmp)) {
    fail(paste0("not equal: ", paste(cmp, collapse = "; "),
                if (!is.null(info)) paste0(" (", info, ")")))
  }
  invisible(TRUE)
}

expect_lt <- function(object, expected) {
  if (!isTRUE(object < expected)) {
    fail(sprintf("expected %s < %s", format(object), format(expected)))
  }
  invisible(TRUE)
}

expect_gt <- function(object, expected) {
  if (!isTRUE(object > expected)) {
    fail(sprintf("expected %s > %s", format(object), format(expected)))
  }
  invisible(TRUE)
}

expect_warning <- function(expr, regexp = NULL) {
  seen <- character(0)
  withCallingHandlers(
    expr,
    warning = function(w) {
      seen <<- c(seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  if (length(seen) == 0) fail("expected a warning, none raised")
  if (!is.null(regexp) && !any(grepl(regexp, seen))) {
    fail(sprintf("no warning matched %s; got: %s", regexp,
                 paste(seen, collapse = " | ")))
  }
  invisible(TRUE)
}

expect_error <- function(expr, regexp = NULL) {
  caught <- tryCatch({ force(expr); NULL }, error = function(e) e)
  if (is.null(caught)) fail("expected an error, none raised")
  if (!is.null(regexp) && !grepl(regexp, conditionMessage(caught))) {
    fail(sprintf("error did not match %s; got: %s", regexp,
                 conditionMessage(caught)))
  }
  invisible(TRUE)
}

# Several samplers print tuning advice with cat() when the acceptance rate
# sits outside their healthy band. That is useful at the console and noise in
# a test log, so tests that deliberately run a sampler off-target wrap it.
quietly <- function(expr) {
  result <- NULL
  invisible(utils::capture.output(result <- expr))
  result
}

test_that <- function(desc, code) {
  TESTS$current <- desc
  result <- tryCatch({ force(code); "pass" }, error = function(e) e)

  if (identical(result, "pass")) {
    TESTS$passed <- TESTS$passed + 1L
    cat(sprintf("  PASS  %s\n", desc))
  } else {
    TESTS$failed <- TESTS$failed + 1L
    msg <- sprintf("%s\n          %s", desc, conditionMessage(result))
    TESTS$failures <- c(TESTS$failures, msg)
    cat(sprintf("  FAIL  %s\n          %s\n", desc, conditionMessage(result)))
  }
  invisible(NULL)
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if (!file.exists("benchmark_metrics.R")) {
  stop("Run this from the repository root: Rscript tests/run_tests.R")
}

test_files <- sort(Sys.glob(file.path("tests", "test_*.R")))

cat(strrep("=", 70), "\n", sep = "")
cat("MyMCMC test suite\n")
cat(strrep("=", 70), "\n", sep = "")

for (f in test_files) {
  cat("\n", basename(f), "\n", sep = "")
  source(f, local = new.env(parent = globalenv()))
}

cat("\n", strrep("=", 70), "\n", sep = "")
cat(sprintf("%d passed, %d failed\n", TESTS$passed, TESTS$failed))
cat(strrep("=", 70), "\n", sep = "")

if (TESTS$failed > 0) {
  cat("\nFailures:\n")
  for (f in TESTS$failures) cat("  - ", f, "\n", sep = "")
  quit(status = 1)
}
