# ============================================================================
# BENCHMARK RESULTS ANALYSIS
# ============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)

# Load results.
#
# This reads the completed run, not the checkpoint. An earlier version read
# benchmark_results_partial.rds, which is written every 3 models and stopped at
# 6 of the 7 that finished. The missing one was diamonds-diamonds at 26
# parameters, the only high-dimensional model in the set, so every figure and
# CSV lost the entire top of the dimension range while their subtitles still
# announced it.
RESULTS_FILE <- "benchmark_results.rds"

if (!file.exists(RESULTS_FILE)) {
  stop(RESULTS_FILE, " not found. Run run_benchmarking.R first.")
}

benchmark <- readRDS(RESULTS_FILE)

# Refuse results that predate the diagnostics fix rather than charting numbers
# whose labels do not match what they measure. See run_benchmarking.R.
if (!identical(benchmark$schema, "chainwise-diagnostics-v2")) {
  stop(
    RESULTS_FILE, " was written before the convergence diagnostics were fixed.\n",
    "  Its ESS and R-hat columns were computed per chain over a whole\n",
    "  multi-parameter matrix, so they report the parameter count rather than\n",
    "  sampling quality. Re-run run_benchmarking.R to regenerate it."
  )
}

results <- benchmark$results
cat("Loaded", length(results), "models from", RESULTS_FILE, "\n")

`%||%` <- function(x, y) if (is.null(x)) y else x

# ============================================================================
# 1. CREATE SUMMARY TABLE
# ============================================================================

summary_rows <- list()

for (model_name in names(results)) {
  model_res <- results[[model_name]]

  for (algo_name in names(model_res)) {
    entry <- model_res[[algo_name]]
    per_chain <- entry$per_chain
    conv <- entry$convergence

    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      Model = model_name,
      Algorithm = algo_name,
      Dimension = attr(model_res, "dimension"),
      # Whether this model was sampled through its own Stan program or
      # through a Gaussian fitted to its reference draws. Carried into every
      # table so the two can never be read as the same experiment.
      Target = attr(model_res, "target_kind") %||% "unknown",
      # Condition number of the reference posterior's correlation matrix.
      # The baseline is given the marginal scales, so this is what is left for
      # adaptation to earn, and it is unrelated to dimension.
      Condition = attr(model_res, "condition") %||% NA_real_,
      # Averaged over chains, because accuracy and acceptance are per-chain.
      RMSE = mean(unlist(per_chain$rmse), na.rm = TRUE),
      MAE = mean(unlist(per_chain$mae), na.rm = TRUE),
      # In reference posterior standard deviations, so it means the same
      # thing in every model. See benchmark_metrics.R.
      RMSE_normalised = mean(unlist(per_chain$rmse_normalised), na.rm = TRUE),
      MAE_normalised = mean(unlist(per_chain$mae_normalised), na.rm = TRUE),
      Acceptance = mean(unlist(per_chain$acceptance_rate), na.rm = TRUE),
      Runtime = mean(unlist(per_chain$runtime), na.rm = TRUE),
      # Taken as computed, because these are already all-chains quantities.
      # Averaging them over chains is what made the R-hat column meaningless.
      ESS_median = conv$ess_bulk_median,
      ESS_min = conv$ess_bulk_min,
      ESS_per_sec = conv$ess_per_sec_median,
      Rhat_max = conv$rhat_max,
      Params_unconverged = conv$n_params_unconverged,
      N_params = conv$n_params,
      Total_draws = conv$total_draws,
      stringsAsFactors = FALSE
    )
  }
}

summary_df <- bind_rows(summary_rows)
summary_df <- as.data.frame(summary_df)

# Dimension comes from the run itself. A hardcoded lookup table used to sit
# here, which meant the plots could disagree with the models actually loaded.
summary_df$Dimension <- summary_df$N_params

# Subtitles are derived for the same reason: they used to be written by hand
# and kept announcing 7 models and a 3 to 26 parameter range while the partial
# results file supplied 6 models topping out at 8.
n_models <- length(unique(summary_df$Model))
n_chains_run <- max(unlist(lapply(results, function(m) {
  max(vapply(m, function(a) a$n_chains_completed, integer(1)))
})))
dim_range <- range(summary_df$Dimension, na.rm = TRUE)
n_iterations <- benchmark$config$n_iterations %||% NA

# Which target each model was sampled through goes into the subtitle too. A
# figure mixing real Stan posteriors with Gaussian surrogates has to say so on
# its face, not in a caption somewhere else.
target_mix <- table(summary_df$Target[!duplicated(summary_df$Model)])
target_note <- paste(names(target_mix), target_mix, sep = ": ", collapse = ", ")

coverage <- sprintf("%d models, %d chains of %s iterations, %d to %d parameters (%s)",
                    n_models, n_chains_run,
                    if (is.na(n_iterations)) "?" else format(n_iterations, big.mark = ","),
                    dim_range[1], dim_range[2], target_note)

# Add dimension category
summary_df$DimCategory <- cut(as.numeric(summary_df$Dimension), 
                              breaks = c(0, 5, 10, Inf),
                              labels = c("Low (2-5)", "Medium (6-10)", "High (11+)"))

cat("\n")
cat(rep("=", 80), "\n", sep = "")
cat("BENCHMARK RESULTS SUMMARY\n")
cat(rep("=", 80), "\n\n", sep = "")

print(knitr::kable(summary_df %>%
                     select(Model, Algorithm, Target, Dimension, Condition,
                            Acceptance, ESS_median, ESS_per_sec, Rhat_max,
                            RMSE, RMSE_normalised, Runtime) %>%
                     arrange(Model, Algorithm),
                   digits = 3, format = "markdown"))

# ============================================================================
# 2. AGGREGATE BY ALGORITHM
# ============================================================================

cat("\n\n")
cat(rep("=", 80), "\n", sep = "")
cat("ALGORITHM PERFORMANCE SUMMARY\n")
cat(rep("=", 80), "\n\n", sep = "")

# summary_df already holds one row per model and algorithm, with the chains
# averaged in section 1. Dividing the row count by the chain count, as an
# earlier version did, reported 1.75 models where there are 7.
algo_summary <- summary_df %>%
  group_by(Algorithm) %>%
  summarise(
    N_models = n(),
    Mean_Acceptance = mean(Acceptance, na.rm = TRUE),
    SD_Acceptance = sd(Acceptance, na.rm = TRUE),
    Mean_ESS = mean(ESS_median, na.rm = TRUE),
    Mean_ESS_per_sec = mean(ESS_per_sec, na.rm = TRUE),
    Worst_Rhat = max(Rhat_max, na.rm = TRUE),
    # Only the normalised error is averaged across models. The raw one is on
    # each model's own units, and averaging it reports whichever model has the
    # largest numbers - here earnings-earn_height, whose parameters are in
    # dollars. It stays in the per-model table above, where it is readable.
    Mean_RMSE_normalised = mean(RMSE_normalised, na.rm = TRUE),
    Worst_RMSE_normalised = max(RMSE_normalised, na.rm = TRUE),
    Mean_Runtime = mean(Runtime, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(Mean_ESS_per_sec))

print(knitr::kable(algo_summary, digits = 3, format = "markdown"))

# ============================================================================
# 3. PERFORMANCE BY DIMENSION
# ============================================================================

cat("\n\n")
cat(rep("=", 80), "\n", sep = "")
cat("PERFORMANCE BY DIMENSION CATEGORY\n")
cat(rep("=", 80), "\n\n", sep = "")

dim_summary <- summary_df %>%
  group_by(Algorithm, DimCategory) %>%
  summarise(
    N_models = n(),
    Mean_Acceptance = mean(Acceptance, na.rm = TRUE),
    Mean_ESS_per_sec = mean(ESS_per_sec, na.rm = TRUE),
    Worst_Rhat = max(Rhat_max, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(DimCategory, desc(Mean_ESS_per_sec))

print(knitr::kable(dim_summary, digits = 3, format = "markdown"))

# ============================================================================
# 4. IDENTIFY BEST ALGORITHM PER MODEL
# ============================================================================

cat("\n\n")
cat(rep("=", 80), "\n", sep = "")
cat("BEST ALGORITHM PER MODEL (by ESS/sec)\n")
cat(rep("=", 80), "\n\n", sep = "")

best_per_model <- summary_df %>%
  group_by(Model) %>%
  filter(ESS_per_sec == max(ESS_per_sec, na.rm = TRUE)) %>%
  select(Model, Algorithm, Dimension, ESS_per_sec, Acceptance) %>%
  arrange(Dimension)

print(knitr::kable(best_per_model, digits = 3, format = "markdown"))

# ============================================================================
# 5. CREATE VISUALIZATIONS
# ============================================================================

# Ensure figures directory exists
if (!dir.exists("figures")) dir.create("figures")

# Figure 1: Acceptance rates by model and algorithm
p1 <- ggplot(summary_df, aes(x = reorder(Model, Dimension), y = Acceptance, 
                             color = Algorithm, group = Algorithm)) +
  geom_point(size = 3) +
  geom_line() +
  geom_hline(yintercept = 0.234, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("text", x = 1, y = 0.26, label = "Optimal (0.234)", 
           vjust = 0, color = "gray50", size = 3.5) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_color_manual(values = c("AM" = "#E41A1C", "RAM" = "#377EB8", "RWM_baseline" = "#4DAF4A")) +
  labs(title = "Acceptance Rates Across Models",
       subtitle = paste("Models ordered by dimension.", coverage),
       y = "Acceptance Rate",
       x = "Model",
       color = "Algorithm") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
        legend.position = "top",
        plot.title = element_text(face = "bold"))

ggsave("figures/acceptance_rates.pdf", p1, width = 10, height = 6)
ggsave("figures/acceptance_rates.png", p1, width = 10, height = 6, dpi = 300)

# Figure 2: ESS per second comparison (boxplot)
p2 <- ggplot(summary_df, aes(x = Algorithm, y = ESS_per_sec, fill = Algorithm)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_log10(labels = scales::comma) +
  scale_fill_manual(values = c("AM" = "#E41A1C", "RAM" = "#377EB8", "RWM_baseline" = "#4DAF4A")) +
  labs(title = "Sampling Efficiency: Effective Sample Size per Second",
       subtitle = paste0("Across ", coverage, ", log scale"),
       y = "ESS per Second (log₁₀ scale)",
       x = "Algorithm") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))

ggsave("figures/ess_comparison.pdf", p2, width = 8, height = 6)
ggsave("figures/ess_comparison.png", p2, width = 8, height = 6, dpi = 300)

# Figure 3: accuracy vs dimension.
#
# Normalised, because this figure puts every model on one pair of axes. On the
# raw scale the y axis is dominated by whichever model has the largest units
# and the comparison across dimensions is unreadable.
p3 <- ggplot(summary_df, aes(x = Dimension, y = RMSE_normalised,
                             color = Algorithm, shape = Algorithm)) +
  geom_point(size = 3, alpha = 0.7) +
  # A loess band used to sit here. With one point per model there are only
  # a handful per algorithm, and loess replies "span too small, fewer data
  # values than degrees of freedom" and falls back to a pseudoinverse. The
  # curve it drew was an artefact, and Figure 4 leaned on it for a claim
  # about RAM holding its target. Connecting the observed points states
  # exactly what was measured and nothing more.
  geom_line(alpha = 0.5) +
  scale_y_log10() +
  scale_color_manual(values = c("AM" = "#E41A1C", "RAM" = "#377EB8", "RWM_baseline" = "#4DAF4A")) +
  labs(title = "Accuracy vs Model Dimension",
       subtitle = paste("Posterior mean error in reference posterior SDs.", coverage),
       y = "RMSE (reference posterior SDs, log₁₀ scale)",
       x = "Number of Parameters",
       color = "Algorithm",
       shape = "Algorithm") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"))

ggsave("figures/rmse_dimension.pdf", p3, width = 10, height = 6)
ggsave("figures/rmse_dimension.png", p3, width = 10, height = 6, dpi = 300)

# Figure 4: Acceptance rate stability across dimensions
p4 <- ggplot(summary_df, aes(x = Dimension, y = Acceptance, 
                             color = Algorithm, shape = Algorithm)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_line(alpha = 0.5) +   # see note on the loess band above
  geom_hline(yintercept = 0.234, linetype = "dashed", color = "gray50", linewidth = 0.8) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.20, ymax = 0.27, 
           alpha = 0.1, fill = "blue") +
  # Placed from the data rather than at a hardcoded x = 25, which only sat
  # inside the panel while the widest model happened to have 26 parameters.
  annotate("text", x = max(summary_df$Dimension, na.rm = TRUE) * 0.85, y = 0.30,
           label = "Optimal range", color = "gray40", size = 3.5) +
  scale_color_manual(values = c("AM" = "#E41A1C", "RAM" = "#377EB8", "RWM_baseline" = "#4DAF4A")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(title = "Acceptance Rate Stability Across Dimensions",
       subtitle = paste("Dashed line marks the 0.234 optimum.", coverage),
       y = "Acceptance Rate",
       x = "Number of Parameters",
       color = "Algorithm",
       shape = "Algorithm") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"))

ggsave("figures/acceptance_stability.pdf", p4, width = 10, height = 6)
ggsave("figures/acceptance_stability.png", p4, width = 10, height = 6, dpi = 300)

# Figure 5: what the adaptation is actually worth.
#
# The baseline is handed each parameter's marginal scale, so the only thing it
# cannot represent is the correlation between parameters. Plotting ESS against
# the correlation matrix's condition number puts the comparison against the
# variable that explains it. Dimension does not: arma-arma11 has 4 parameters
# and a condition number of 1.7, earnings-earn_height has 3 and 1,214.
p5 <- ggplot(summary_df, aes(x = Condition, y = ESS_median,
                             color = Algorithm, shape = Algorithm)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_line(alpha = 0.5) +
  scale_x_log10(labels = scales::comma) +
  scale_y_log10(labels = scales::comma) +
  scale_color_manual(values = c("AM" = "#E41A1C", "RAM" = "#377EB8",
                                "RWM_baseline" = "#4DAF4A")) +
  labs(title = "What Adaptation Buys, Against How Correlated the Posterior Is",
       subtitle = paste("Condition number of the reference correlation matrix.", coverage),
       y = "Median ESS (log₁₀ scale)",
       x = "Correlation condition number (log₁₀ scale)",
       color = "Algorithm",
       shape = "Algorithm") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"))

ggsave("figures/ess_conditioning.pdf", p5, width = 10, height = 6)
ggsave("figures/ess_conditioning.png", p5, width = 10, height = 6, dpi = 300)

# ============================================================================
# 6. SAVE OUTPUTS
# ============================================================================

write.csv(summary_df, "benchmark_summary_detailed.csv", row.names = FALSE)
write.csv(algo_summary, "benchmark_summary_algorithms.csv", row.names = FALSE)
write.csv(dim_summary, "benchmark_summary_by_dimension.csv", row.names = FALSE)
write.csv(best_per_model, "benchmark_best_per_model.csv", row.names = FALSE)

cat("\n\n")
cat(rep("=", 80), "\n", sep = "")
cat("ANALYSIS COMPLETE\n")
cat(rep("=", 80), "\n", sep = "")
cat("\nOutputs saved:\n")
cat("  - figures/acceptance_rates.pdf/png\n")
cat("  - figures/ess_comparison.pdf/png\n")
cat("  - figures/rmse_dimension.pdf/png\n")
cat("  - figures/acceptance_stability.pdf/png\n")
cat("  - figures/ess_conditioning.pdf/png\n")
cat("  - benchmark_summary_detailed.csv\n")
cat("  - benchmark_summary_algorithms.csv\n")
cat("  - benchmark_summary_by_dimension.csv\n")
cat("  - benchmark_best_per_model.csv\n")
cat(rep("=", 80), "\n", sep = "")
