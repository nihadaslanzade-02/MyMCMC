# ============================================================================
# ANALYSIS STAGE
# ============================================================================
# analyze_results.R reads a file and writes figures, so it is exercised by
# handing it fixtures rather than by importing functions out of it.

# The script prints four summary tables on every run. Useful at the console,
# unreadable in a test log, so the output is swallowed and only the files it
# writes are inspected.
run_analysis_in <- function(dir) {
  script <- normalizePath("analyze_results.R")
  old <- setwd(dir)
  on.exit(setwd(old), add = TRUE)
  quietly(source(script, local = new.env(parent = globalenv())))
}

# A results file in the current schema, small enough to plot quickly.
# Five models across five dimensions, which is what loess in the figures
# needs to fit without complaining. Two would still exercise every code
# path but would fill the test output with smoother warnings.
fake_results <- function(models = paste0("model-", 1:5),
                         dims = c(3, 4, 8, 15, 26),
                         schema = "chainwise-diagnostics-v2",
                         raw_scale = rep(1, length(models))) {
  set.seed(42)
  results <- list()

  for (i in seq_along(models)) {
    d <- dims[i]
    per_algo <- list()

    for (algo in c("AM", "RAM", "RWM_baseline")) {
      # raw_scale lets a model be put on huge units while its normalised
      # error stays where every other model's is.
      normalised <- runif(4, 0.01, 0.5)
      per_algo[[algo]] <- list(
        per_chain = data.frame(
          rmse = normalised * raw_scale[i],
          mae = runif(4, 0.01, 0.4) * raw_scale[i],
          rmse_normalised = normalised,
          mae_normalised = runif(4, 0.01, 0.4),
          runtime = runif(4, 0.1, 1),
          acceptance_rate = runif(4, 0.2, 0.3)
        ),
        convergence = list(
          ess_bulk_median = runif(1, 500, 3000),
          ess_bulk_min = runif(1, 100, 500),
          ess_tail_min = runif(1, 100, 500),
          ess_per_sec_median = runif(1, 100, 900),
          ess_per_sec_min = runif(1, 10, 90),
          rhat_max = runif(1, 1.0, 1.02),
          n_params_unconverged = 0L,
          n_params = d,
          total_draws = 20000L,
          per_parameter = NULL
        ),
        total_runtime = 2,
        n_chains_completed = 4L
      )
    }

    attr(per_algo, "dimension") <- d
    attr(per_algo, "target_kind") <- "gaussian-surrogate"
    attr(per_algo, "condition") <- 10^i
    results[[models[i]]] <- per_algo
  }

  out <- list(
    results = results, config = list(), algorithms = c("AM", "RAM", "RWM_baseline"),
    timestamp = Sys.time(), n_successful = length(models), n_failed = 0L,
    selected_models = models
  )
  if (!is.null(schema)) out$schema <- schema
  out
}

# ---------------------------------------------------------------------------
test_that("results written before the diagnostics fix are refused", {
  # The old file is still on disk and still loads. Charting it would put the
  # parameter count on an axis labelled ESS, which is what this guard exists
  # to stop.
  dir <- file.path(tempdir(), "old_schema")
  dir.create(dir, showWarnings = FALSE)
  saveRDS(fake_results(schema = NULL), file.path(dir, "benchmark_results.rds"))

  expect_error(run_analysis_in(dir), "before the convergence diagnostics were fixed")
})

# ---------------------------------------------------------------------------
test_that("a missing results file names the script that produces it", {
  dir <- file.path(tempdir(), "no_results")
  dir.create(dir, showWarnings = FALSE)
  unlink(file.path(dir, "benchmark_results.rds"))

  expect_error(run_analysis_in(dir), "Run run_benchmarking.R first")
})

# ---------------------------------------------------------------------------
test_that("the analysis reads the complete run, not the checkpoint", {
  # The bug this pins: analyze_results.R read benchmark_results_partial.rds,
  # a checkpoint written every 3 models, so the 7th model was missing from
  # every figure and CSV. Here the checkpoint holds one model and the
  # completed file holds two; the output must show two.
  dir <- file.path(tempdir(), "full_vs_partial")
  dir.create(dir, showWarnings = FALSE)

  saveRDS(fake_results(), file.path(dir, "benchmark_results.rds"))
  saveRDS(fake_results(models = "model-1", dims = 3),
          file.path(dir, "benchmark_results_partial.rds"))

  run_analysis_in(dir)

  summary_csv <- read.csv(file.path(dir, "benchmark_summary_detailed.csv"))
  expect_equal(length(unique(summary_csv$Model)), 5)
  expect_true("model-5" %in% summary_csv$Model)
  expect_equal(max(summary_csv$Dimension), 26)
})

# ---------------------------------------------------------------------------
test_that("the model count is not divided by the chain count", {
  # summary_df holds one row per model and algorithm, with chains already
  # averaged. An earlier version reported n() / 4, which turned 7 models into
  # 1.75.
  dir <- file.path(tempdir(), "model_count")
  dir.create(dir, showWarnings = FALSE)
  saveRDS(fake_results(), file.path(dir, "benchmark_results.rds"))

  run_analysis_in(dir)

  algo_csv <- read.csv(file.path(dir, "benchmark_summary_algorithms.csv"))
  expect_equal(sort(unique(algo_csv$N_models)), 5)
  expect_true(all(algo_csv$N_models == as.integer(algo_csv$N_models)))
})

# ---------------------------------------------------------------------------
test_that("one model on huge units cannot dominate the cross-model average", {
  # The bug this pins: the algorithm summary averaged raw RMSE across models.
  # Here model-2 is on a scale 10,000 times the others while its normalised
  # error is drawn from the same distribution as theirs, which is exactly the
  # earnings-earn_height situation. The aggregate must not notice.
  dir <- file.path(tempdir(), "rmse_scale")
  dir.create(dir, showWarnings = FALSE)
  saveRDS(fake_results(raw_scale = c(1, 1e4, 1, 1, 1)),
          file.path(dir, "benchmark_results.rds"))

  run_analysis_in(dir)

  algo_csv <- read.csv(file.path(dir, "benchmark_summary_algorithms.csv"))
  expect_true("Mean_RMSE_normalised" %in% names(algo_csv))
  expect_true(!"Mean_RMSE" %in% names(algo_csv))
  # Every per-chain normalised error was drawn from runif(0.01, 0.5), so the
  # mean of five models' worth has to land inside that range. Averaging the
  # raw column instead would put it in the hundreds.
  expect_true(all(algo_csv$Mean_RMSE_normalised > 0.01))
  expect_true(all(algo_csv$Mean_RMSE_normalised < 0.5))

  # The raw column survives per model, where it is the readable one.
  summary_csv <- read.csv(file.path(dir, "benchmark_summary_detailed.csv"))
  expect_gt(max(summary_csv$RMSE), 100)
  expect_lt(max(summary_csv$RMSE_normalised), 0.5)
})

# ---------------------------------------------------------------------------
test_that("figures are written and their subtitles follow the data", {
  dir <- file.path(tempdir(), "figures_out")
  dir.create(dir, showWarnings = FALSE)
  saveRDS(fake_results(), file.path(dir, "benchmark_results.rds"))

  run_analysis_in(dir)

  for (f in c("acceptance_rates", "ess_comparison", "rmse_dimension",
              "acceptance_stability", "ess_conditioning")) {
    expect_true(file.exists(file.path(dir, "figures", paste0(f, ".png"))))
  }

  # Which target produced each model, and how correlated that posterior is,
  # both travel into the tables. A figure that mixed real Stan posteriors with
  # Gaussian surrogates and did not say so would be the easiest mistake here.
  summary_csv <- read.csv(file.path(dir, "benchmark_summary_detailed.csv"))
  expect_true(all(summary_csv$Target == "gaussian-surrogate"))
  expect_equal(sort(unique(summary_csv$Condition)), 10^(1:5))
})
