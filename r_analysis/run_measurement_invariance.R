#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(lavaan)
})

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
# This script is integrated with CENTAUR infrastructure for path management.
# Data paths point to the new data/processed/ directory structure.

base_dir <- normalizePath(file.path(dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])), "..", ".."))

# Updated paths for CENTAUR-integrated data structure
data_path <- file.path(base_dir, "data", "processed", "regression", "regression_dataset_with_bert_themes.csv")
results_dir <- file.path(base_dir, "results", "measurement_invariance")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

likert_map <- c(
  "Strongly disagree" = 1,
  "Somewhat disagree" = 2,
  "Neither agree or disagree" = 3,
  "Neither agree nor disagree" = 3,
  "Somewhat agree" = 4,
  "Strongly agree" = 5
)

items <- c(
  "3.1_1", "3.1_2", "3.1_3", "3.1_4", "3.1_5", "3.1_7", "3.1_8", "3.1_9", "3.2_1", "3.2_3",
  "3.1_6", "3.2_2", "3.2_4", "3.3_1", "3.3_3",
  "3.2_5", "3.2_6", "3.2_7", "3.3_2",
  "4.1_1", "4.1_2", "4.1_3", "4.1_4", "4.1_5",
  "6.1_1", "6.1_2", "6.1_3", "6.1_4", "6.1_5"
)

df_raw <- readr::read_csv(data_path, show_col_types = FALSE)

data_prepped <- df_raw %>%
  filter(htype %in% c("Homeowner", "Renter")) %>%
  mutate(
    tenure = factor(htype, levels = c("Homeowner", "Renter"))
  ) %>%
  mutate(across(all_of(items), ~ likert_map[.x])) %>%
  drop_na(all_of(items)) %>%
  select(tenure, all_of(items))

ordered_items <- items

measurement_model <- '
attachment_belonging =~ `3.1_1` + `3.1_2` + `3.1_3` + `3.1_4` + `3.1_5` + `3.1_7` + `3.1_8` + `3.1_9` + `3.2_1` + `3.2_3`
move_intent_disamenity =~ `3.1_6` + `3.2_2` + `3.2_4` + `3.3_1` + `3.3_3`
trust_safety =~ `3.2_5` + `3.2_6` + `3.2_7` + `3.3_2`
social_cohesion =~ `4.1_1` + `4.1_2` + `4.1_3` + `4.1_4` + `4.1_5`
quality_of_life =~ `6.1_1` + `6.1_2` + `6.1_3` + `6.1_4` + `6.1_5`
'

fit_configural <- cfa(
  measurement_model,
  data = data_prepped,
  group = "tenure",
  estimator = "WLSMV",
  ordered = ordered_items,
  std.lv = TRUE,
  parameterization = "theta"
)

fit_metric <- cfa(
  measurement_model,
  data = data_prepped,
  group = "tenure",
  estimator = "WLSMV",
  ordered = ordered_items,
  std.lv = TRUE,
  group.equal = c("loadings"),
  parameterization = "theta"
)

fit_scalar <- cfa(
  measurement_model,
  data = data_prepped,
  group = "tenure",
  estimator = "WLSMV",
  ordered = ordered_items,
  std.lv = TRUE,
  group.equal = c("loadings", "thresholds"),
  parameterization = "theta"
)

fit_measures <- function(fit, label) {
  stats <- fitMeasures(fit, c("chisq", "df", "cfi", "tli", "rmsea", "rmsea.ci.lower", "rmsea.ci.upper", "srmr"))
  tibble(
    model = label,
    chisq = stats["chisq"],
    df = stats["df"],
    cfi = stats["cfi"],
    tli = stats["tli"],
    rmsea = stats["rmsea"],
    rmsea_lower = stats["rmsea.ci.lower"],
    rmsea_upper = stats["rmsea.ci.upper"],
    srmr = stats["srmr"]
  )
}

fits_summary <- bind_rows(
  fit_measures(fit_configural, "configural"),
  fit_measures(fit_metric, "metric"),
  fit_measures(fit_scalar, "scalar")
)

deltas <- fits_summary %>%
  arrange(match(model, c("configural", "metric", "scalar"))) %>%
  mutate(
    delta_cfi = c(NA, diff(cfi)),
    delta_tli = c(NA, diff(tli)),
    delta_rmsea = c(NA, diff(rmsea)),
    delta_srmr = c(NA, diff(srmr))
  )

readr::write_csv(fits_summary, file.path(results_dir, "invariance_fit_indices.csv"))
readr::write_csv(deltas, file.path(results_dir, "invariance_fit_deltas.csv"))

lrt <- lavTestLRT(fit_configural, fit_metric, fit_scalar)
readr::write_csv(as_tibble(lrt), file.path(results_dir, "invariance_chisq_tests.csv"))

saveRDS(list(
  configural = fit_configural,
  metric = fit_metric,
  scalar = fit_scalar
), file = file.path(results_dir, "invariance_models.rds"))

cat("Measurement invariance analysis complete. See results/measurement_invariance for outputs.\n")
