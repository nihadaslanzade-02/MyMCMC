# ============================================================================
# BENCHMARK RESULTS ANALYSIS
# ============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)

# Load results
results <- readRDS("benchmark_results_partial.rds")

# ============================================================================
# 1. CREATE SUMMARY TABLE
# ============================================================================

summary_rows <- list()

for (model_name in names(results)) {
  model_res <- results[[model_name]]
  
  for (algo_name in names(model_res)) {
    df <- model_res[[algo_name]]
    
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      Model = model_name,
      Algorithm = algo_name,
      Dimension = attr(model_res, "dimension"),
      ESS_median = mean(unlist(df$ess_median), na.rm = TRUE),
      ESS_per_sec = mean(unlist(df$ess_per_sec_median), na.rm = TRUE),
      RMSE = mean(unlist(df$rmse), na.rm = TRUE),
      MAE = mean(unlist(df$mae), na.rm = TRUE),
      Acceptance = mean(unlist(df$acceptance_rate), na.rm = TRUE),
      Runtime = mean(unlist(df$runtime), na.rm = TRUE),
      Rhat_max = mean(unlist(df$rhat_max), na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }
}

summary_df <- bind_rows(summary_rows)
summary_df <- as.data.frame(summary_df)

# Dimension
dimension_map <- c(
  "arma-arma11" = 4,
  "earnings-earn_height" = 3,
  "earnings-log10earn_height" = 3,
  "arK-arK" = 7,
  "bball_drive_event_0-hmm_drive_0" = 8,
  "bball_drive_event_1-hmm_drive_1" = 8,
  "diamonds-diamonds" = 26
)

summary_df$Dimension <- dimension_map[summary_df$Model]

# Add dimension category
summary_df$DimCategory <- cut(as.numeric(summary_df$Dimension), 
                              breaks = c(0, 5, 10, Inf),
                              labels = c("Low (2-5)", "Medium (6-10)", "High (11+)"))

cat("\n")
cat(rep("=", 80), "\n", sep = "")
cat("BENCHMARK RESULTS SUMMARY\n")
cat(rep("=", 80), "\n\n", sep = "")

print(knitr::kable(summary_df %>% 
                     select(Model, Algorithm, Dimension, Acceptance, 
                            ESS_per_sec, RMSE, Runtime) %>%
                     arrange(Model, Algorithm), 
                   digits = 3, format = "markdown"))

# ============================================================================
# 2. AGGREGATE BY ALGORITHM
# ============================================================================

cat("\n\n")
cat(rep("=", 80), "\n", sep = "")
cat("ALGORITHM PERFORMANCE SUMMARY\n")
cat(rep("=", 80), "\n\n", sep = "")

algo_summary <- summary_df %>%
  group_by(Algorithm) %>%
  summarise(
    N_models = n() / 4,  # 4 chains per model
    Mean_Acceptance = mean(Acceptance, na.rm = TRUE),
    SD_Acceptance = sd(Acceptance, na.rm = TRUE),
    Mean_ESS_per_sec = mean(ESS_per_sec, na.rm = TRUE),
    Mean_RMSE = mean(RMSE, na.rm = TRUE),
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
    N_models = n() / 4,
    Mean_Acceptance = mean(Acceptance, na.rm = TRUE),
    Mean_ESS_per_sec = mean(ESS_per_sec, na.rm = TRUE),
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
  geom_hline(yintercept = 0.234, linetype = "dashed", color = "gray50", size = 0.8) +
  annotate("text", x = 1, y = 0.26, label = "Optimal (0.234)", 
           vjust = 0, color = "gray50", size = 3.5) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_color_manual(values = c("AM" = "#E41A1C", "RAM" = "#377EB8", "RWM_baseline" = "#4DAF4A")) +
  labs(title = "Acceptance Rates Across Models",
       subtitle = "Models ordered by dimension (left to right: 3 to 26 parameters)",
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
       subtitle = "Distribution across 7 models × 4 chains (log scale)",
       y = "ESS per Second (log₁₀ scale)",
       x = "Algorithm") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold"))

ggsave("figures/ess_comparison.pdf", p2, width = 8, height = 6)
ggsave("figures/ess_comparison.png", p2, width = 8, height = 6, dpi = 300)

# Figure 3: RMSE vs Dimension
p3 <- ggplot(summary_df, aes(x = Dimension, y = RMSE, color = Algorithm, shape = Algorithm)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.15) +
  scale_y_log10() +
  scale_color_manual(values = c("AM" = "#E41A1C", "RAM" = "#377EB8", "RWM_baseline" = "#4DAF4A")) +
  labs(title = "Accuracy vs Model Dimension",
       subtitle = "Root Mean Squared Error compared to reference posterior",
       y = "RMSE (log₁₀ scale)",
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
  geom_smooth(method = "loess", se = TRUE, alpha = 0.15) +
  geom_hline(yintercept = 0.234, linetype = "dashed", color = "gray50", size = 0.8) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.20, ymax = 0.27, 
           alpha = 0.1, fill = "blue") +
  annotate("text", x = 25, y = 0.28, label = "Optimal range", 
           color = "gray40", size = 3.5) +
  scale_color_manual(values = c("AM" = "#E41A1C", "RAM" = "#377EB8", "RWM_baseline" = "#4DAF4A")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(title = "Acceptance Rate Stability Across Dimensions",
       subtitle = "RAM maintains near-optimal rates consistently",
       y = "Acceptance Rate",
       x = "Number of Parameters",
       color = "Algorithm",
       shape = "Algorithm") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"))

ggsave("figures/acceptance_stability.pdf", p4, width = 10, height = 6)
ggsave("figures/acceptance_stability.png", p4, width = 10, height = 6, dpi = 300)

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
cat("  - benchmark_summary_detailed.csv\n")
cat("  - benchmark_summary_algorithms.csv\n")
cat("  - benchmark_summary_by_dimension.csv\n")
cat("  - benchmark_best_per_model.csv\n")
cat(rep("=", 80), "\n", sep = "")
