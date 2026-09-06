#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(mice)
  library(broom)
})

timing_mode <- Sys.getenv("TIMING_MODE", unset = "optionB")
timing_var <- dplyr::case_when(
  timing_mode == "optionA" ~ "days_since_event_optionA",
  timing_mode == "optionB" ~ "days_since_event_optionB",
  timing_mode == "optionC" ~ "time_since_event_bin",
  TRUE ~ "days_since_event_optionB"
)

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
# This script is integrated with CENTAUR infrastructure for path management.
# Data paths point to the new data/processed/ directory structure.

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  script_path <- "src/analysis/run_missingness_and_mi.R"
}
base_dir <- normalizePath(file.path(dirname(script_path), "..", ".."))

# Updated paths for CENTAUR-integrated data structure
regression_path <- file.path(base_dir, "data", "processed", "regression", "recovery_model_dataset.csv")
coverage_path <- file.path(base_dir, "data", "processed", "insurance_model_dataset.csv")
results_dir <- file.path(base_dir, "results", "cluster_inference")
missing_dir <- file.path(base_dir, "results", "insurance")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(missing_dir, recursive = TRUE, showWarnings = FALSE)

read_dataset <- function() {
  survey <- readr::read_csv(regression_path, show_col_types = FALSE)
  coverage <- readr::read_csv(coverage_path, show_col_types = FALSE) %>%
    select(response_id, coverage_peril_aligned)

  survey %>%
    left_join(coverage, by = c("ResponseId" = "response_id")) %>%
    mutate(
      coverage_peril_aligned = as.numeric(coverage_peril_aligned),
      renter_flag = if_else(htype == "Renter", 1, 0),
      structure_multifamily = suppressWarnings(as.numeric(structure_multifamily)),
      mfr_flag = if_else(replace_na(structure_multifamily, 0) > 0, 1, 0)
    )
}

missingness_summary <- function(df) {
  vars <- c(
    "coverage_peril_aligned", "aid_flag", "displacement_flag", "income_ord",
    "damage_ord", "employed", "age_years", "household_size", "attachment_factor",
    timing_var
  )
  df %>%
    mutate(htype = factor(htype)) %>%
    group_by(htype) %>%
    summarise(across(all_of(vars), ~ mean(is.na(.x)), .names = "missing_{.col}"), .groups = "drop") %>%
    pivot_longer(-htype, names_to = "variable", values_to = "missing_rate")
}

prep_mi_data <- function(df) {
  timing_sym <- rlang::sym(timing_var)
  data <- df %>%
    select(
      displacement_flag, renter_flag, mfr_flag, income_ord, employed, age_years,
      household_size, damage_ord, attachment_factor, aid_flag, coverage_peril_aligned,
      all_of(timing_var)
    )

  data <- data %>%
    mutate(across(
      c(displacement_flag, renter_flag, mfr_flag, employed, aid_flag),
      ~ suppressWarnings(as.numeric(.x))
    )) %>%
    mutate(
      income_ord = as.numeric(income_ord),
      age_years = as.numeric(age_years),
      household_size = as.numeric(household_size),
      damage_ord = as.numeric(damage_ord),
      attachment_factor = as.numeric(attachment_factor),
      coverage_peril_aligned = as.numeric(coverage_peril_aligned)
    )

  if (timing_mode == "optionC") {
    data <- data %>%
      mutate(!!timing_sym := factor(!!timing_sym, levels = c("0-3 months", "3-6 months", "6-9 months", "9-12 months", ">12 months", "Unknown")))
  } else {
    data <- data %>%
      mutate(!!timing_sym := as.numeric(!!timing_sym))
  }

  data %>%
    mutate(
      displacement_flag = factor(if_else(replace_na(displacement_flag, 0) > 0, 1, 0)),
      renter_flag = factor(if_else(replace_na(renter_flag, 0) > 0, 1, 0)),
      mfr_flag = factor(if_else(replace_na(mfr_flag, 0) > 0, 1, 0)),
      employed = factor(if_else(replace_na(employed, 0) > 0, 1, 0)),
      aid_flag = factor(if_else(replace_na(aid_flag, 0) > 0, 1, 0))
    )
}

data <- read_dataset()
missing_tbl <- missingness_summary(data)
readr::write_csv(missing_tbl, file.path(missing_dir, "missingness_by_tenure_detailed.csv"))

mi_data <- prep_mi_data(data)
methods <- make.method(mi_data)
methods[] <- "pmm"
methods[c("displacement_flag", "aid_flag")] <- "logreg"
methods[c("renter_flag", "mfr_flag", "employed")] <- ""
if (timing_mode == "optionC") {
  methods[timing_var] <- ""
}

predictors <- make.predictorMatrix(mi_data)
predictors[,] <- 1
predictors[colnames(predictors) == "displacement_flag", "displacement_flag"] <- 0
predictors[colnames(predictors) == "aid_flag", "aid_flag"] <- 0
zero_method_vars <- names(methods[methods == ""])
predictors[zero_method_vars, ] <- 0
predictors[, zero_method_vars] <- 0

set.seed(20251026)
mi_fit <- mice(mi_data, m = 20, method = methods, predictorMatrix = predictors, maxit = 20, printFlag = FALSE)

mediator_formula_str <- paste(
  "displacement_flag ~ renter_flag + mfr_flag + income_ord + employed + age_years +",
  "household_size + damage_ord + attachment_factor +",
  timing_var,
  "+ coverage_peril_aligned + aid_flag"
)

total_formula_str <- paste(
  "displacement_flag ~ renter_flag + mfr_flag + income_ord + employed + age_years +",
  "household_size + damage_ord + attachment_factor +",
  timing_var
)

model_mediator <- with(
  mi_fit,
  glm(
    stats::as.formula(mediator_formula_str),
    family = binomial()
  )
)

model_total <- with(
  mi_fit,
  glm(
    stats::as.formula(total_formula_str),
    family = binomial()
  )
)

results_mediator <- summary(pool(model_mediator), conf.int = TRUE, exponentiate = TRUE) %>%
  mutate(spec = "with_mediators")
results_total <- summary(pool(model_total), conf.int = TRUE, exponentiate = TRUE) %>%
  mutate(spec = "total_effect")

combined <- bind_rows(results_mediator, results_total)
readr::write_csv(combined, file.path(results_dir, "displacement_mi_results.csv"))
