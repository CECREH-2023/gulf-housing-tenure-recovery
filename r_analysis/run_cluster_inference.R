#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(fixest)
  library(fwildclusterboot)
  library(lme4)
  library(broom)
library(broom.mixed)
library(glue)
library(dqrng)
})

timing_mode <- Sys.getenv("TIMING_MODE", unset = "optionB")
timing_predictor <- dplyr::case_when(
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
if (identical(script_path, character(0))) {
  script_path <- "src/analysis/run_cluster_inference.R"
}
base_dir <- normalizePath(file.path(dirname(script_path), "..", ".."))

# Updated paths for CENTAUR-integrated data structure
data_path <- file.path(base_dir, "data", "processed", "regression", "recovery_model_dataset.csv")
results_dir <- file.path(base_dir, "results", "cluster_inference")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

read_dataset <- function(path) {
  readr::read_csv(path, show_col_types = FALSE) %>%
    mutate(
      hazard_cluster = factor(hazard_cluster),
      renter_flag = as.numeric(renter_flag),
      income_ord = as.numeric(income_ord),
      attachment_factor = as.numeric(attachment_factor),
      employed = as.numeric(employed),
      age_years = as.numeric(age_years),
      household_size = as.numeric(household_size),
      damage_ord = as.numeric(damage_ord),
      mfr_flag = as.numeric(mfr_flag),
      insurance_yes = as.numeric(insurance_yes),
      aid_flag = as.numeric(aid_flag),
      displacement_flag = as.numeric(displacement_flag),
      days_since_event_optionA = as.numeric(days_since_event_optionA),
      days_since_event_optionB = as.numeric(days_since_event_optionB),
      time_since_event_bin = factor(
        time_since_event_bin,
        levels = c("0-3 months", "3-6 months", "6-9 months", "9-12 months", ">12 months", "Unknown")
      )
    )
}

prepare_model_data <- function(df, outcome, predictors, cluster) {
  vars <- unique(c(outcome, predictors, cluster))
  df %>%
    select(all_of(vars)) %>%
    drop_na() %>%
    mutate("{cluster}" := forcats::fct_drop(.data[[cluster]]))
}

build_formula <- function(outcome, predictors) {
  stats::as.formula(glue("{outcome} ~ {paste(predictors, collapse = ' + ')}"))
}

to_summary_row <- function(model_name, term, estimate, std_error, conf_low, conf_high, cluster_name) {
  tibble(
    model = model_name,
    term = term,
    log_odds = estimate,
    std_error = std_error,
    conf_low = conf_low,
    conf_high = conf_high,
    odds_ratio = exp(estimate),
    or_low = exp(conf_low),
    or_high = exp(conf_high),
    cluster_var = cluster_name
  )
}

run_wcb <- function(model, model_name, cluster_formula, term = "renter_flag", n_clusters = NULL) {
  if (is.null(n_clusters)) {
    B_value <- 9999L
  } else {
    max_draws <- as.integer(2^n_clusters)
    B_value <- min(9999L, max_draws)
  }
  boot <- boottest(
    object = model,
    param = term,
    B = B_value,
    clustid = cluster_formula,
    type = "rademacher",
    bootstrap_type = "fnw11",
    impose_null = TRUE,
    p_val_type = "two-tailed"
  )
  tidy_boot <- broom::tidy(boot)
  tibble(
    model = model_name,
    n_clusters = n_clusters,
    B = B_value,
    term = term,
    wcb_estimate = tidy_boot$estimate,
    wcb_stat = tidy_boot$statistic,
    wcb_p_value = tidy_boot$p.value,
    wcb_conf_low = tidy_boot$conf.low,
    wcb_conf_high = tidy_boot$conf.high
  )
}

extract_icc <- function(glmer_fit, cluster_name) {
  vc <- as.data.frame(VarCorr(glmer_fit))
  tau2 <- vc %>%
    filter(grp == cluster_name) %>%
    pull(vcov)
  if (length(tau2) == 0) {
    return(NA_real_)
  }
  tau2 / (tau2 + (pi ^ 2) / 3)
}

dataset <- read_dataset(data_path)

coverage_path <- file.path(base_dir, "data", "processed", "insurance_model_dataset.csv")
coverage_df <- readr::read_csv(coverage_path, show_col_types = FALSE) %>%
  select(response_id, coverage_peril_aligned)

dataset <- dataset %>%
  left_join(coverage_df, by = c("ResponseId" = "response_id")) %>%
  mutate(coverage_peril_aligned = as.numeric(coverage_peril_aligned))

cluster_counts <- dataset %>%
  count(hazard_cluster, name = "n") %>%
  arrange(desc(n))

small_clusters <- cluster_counts %>%
  filter(n < 5) %>%
  pull(hazard_cluster)

dataset <- dataset %>%
  mutate(
    hazard_cluster_collapsed = if_else(
      hazard_cluster %in% small_clusters,
      "Other (≤4 cases)",
      as.character(hazard_cluster)
    ),
    hazard_cluster_collapsed = factor(hazard_cluster_collapsed)
  )

collapsed_summary <- dataset %>%
  count(hazard_cluster_collapsed, name = "n") %>%
  arrange(desc(n))

min_cluster_size <- min(collapsed_summary$n)
median_cluster_size <- stats::median(collapsed_summary$n)

cluster_meta <- collapsed_summary %>%
  mutate(
    min_cluster_size = min_cluster_size,
    median_cluster_size = median_cluster_size
  )

readr::write_csv(cluster_counts, file.path(results_dir, "hazard_cluster_raw_counts.csv"))
readr::write_csv(cluster_meta, file.path(results_dir, "hazard_cluster_collapsed_counts.csv"))

specs <- list(
  list(
    name = "insurance",
    outcome = "coverage_peril_aligned",
    predictors = c("renter_flag", "income_ord", "attachment_factor", "employed", "age_years", "household_size", "damage_ord"),
    cluster = "hazard_cluster_collapsed"
  ),
  list(
    name = "aid",
    outcome = "aid_flag",
    predictors = c("renter_flag", "mfr_flag", "income_ord", "employed", "age_years", "household_size", "damage_ord", "attachment_factor", timing_predictor),
    cluster = "hazard_cluster_collapsed"
  ),
  list(
    name = "displacement_with_mediators",
    outcome = "displacement_flag",
    predictors = c("renter_flag", "mfr_flag", "income_ord", "employed", "age_years", "household_size", "damage_ord", "attachment_factor", timing_predictor, "coverage_peril_aligned", "aid_flag"),
    cluster = "hazard_cluster_collapsed"
  ),
  list(
    name = "displacement_without_mediators",
    outcome = "displacement_flag",
    predictors = c("renter_flag", "mfr_flag", "income_ord", "employed", "age_years", "household_size", "damage_ord", "attachment_factor", timing_predictor),
    cluster = "hazard_cluster_collapsed"
  )
)

conventional_rows <- list()
wcb_rows <- list()
random_intercept_rows <- list()

set.seed(20251026)
dqrng::dqset.seed(20251026)

for (spec in specs) {
  model_name <- spec$name
  outcome <- spec$outcome
  predictors <- spec$predictors
  cluster_var <- spec$cluster

  model_data <- prepare_model_data(dataset, outcome, predictors, cluster_var)

  if (nrow(model_data) == 0) {
    warning(glue("Skipping {model_name}: no complete cases."))
    next
  }

  if (n_distinct(model_data[[cluster_var]]) < 2) {
    warning(glue("Skipping {model_name}: insufficient clusters after filtering."))
    next
  }

  formula <- build_formula(outcome, predictors)

  logit_model <- fixest::feglm(
    formula,
    data = model_data,
    family = binomial(link = "logit")
  )

  vcov_formula <- reformulate(cluster_var)
  fe_summary <- summary(logit_model, vcov = vcov_formula)
  coef_table <- as.data.frame(fe_summary$coeftable) %>%
    rownames_to_column(var = "term")

  renter_row <- coef_table %>%
    filter(term == "renter_flag")

  if (nrow(renter_row) == 1) {
    crit <- qnorm(0.975)
    renter_low <- renter_row$Estimate - crit * renter_row$`Std. Error`
    renter_high <- renter_row$Estimate + crit * renter_row$`Std. Error`
    conventional_rows[[model_name]] <- to_summary_row(
      model_name = model_name,
      term = renter_row$term,
      estimate = renter_row$Estimate,
      std_error = renter_row$`Std. Error`,
      conf_low = renter_low,
      conf_high = renter_high,
      cluster_name = cluster_var
    )
  } else {
    warning(glue("Term renter_flag not found in {model_name} conventional output."))
  }

  lpm_model <- fixest::feols(
    formula,
    data = model_data
  )

  cluster_formula <- reformulate(cluster_var)
  n_clusters <- dplyr::n_distinct(model_data[[cluster_var]])
  wcb_rows[[model_name]] <- run_wcb(
    model = lpm_model,
    model_name = model_name,
    cluster_formula = cluster_formula,
    n_clusters = n_clusters
  )

  random_effect_term <- glue("(1 | {cluster_var})")
  glmer_formula <- stats::as.formula(glue(
    "{outcome} ~ {paste(c(predictors, random_effect_term), collapse = ' + ')}"
  ))
  glmer_fit <- tryCatch(
    glmer(
      glmer_formula,
      data = model_data,
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
    ),
    error = function(e) {
      warning(glue("Random intercept model failed for {model_name}: {e$message}"))
      return(NULL)
    }
  )

  if (!is.null(glmer_fit)) {
    renter_tidy <- broom.mixed::tidy(
      glmer_fit,
      effects = "fixed",
      conf.int = TRUE,
      conf.method = "Wald"
    ) %>%
      filter(term == "renter_flag")
    if (nrow(renter_tidy) == 1) {
      icc <- extract_icc(glmer_fit, cluster_var)
      random_intercept_rows[[model_name]] <- tibble(
        model = model_name,
        term = renter_tidy$term,
        log_odds = renter_tidy$estimate,
        std_error = renter_tidy$std.error,
        conf_low = renter_tidy$conf.low,
        conf_high = renter_tidy$conf.high,
        odds_ratio = exp(renter_tidy$estimate),
        or_low = exp(renter_tidy$conf.low),
        or_high = exp(renter_tidy$conf.high),
        icc = icc,
        cluster_var = cluster_var,
        n_clusters = n_distinct(model_data[[cluster_var]]),
        n_obs = nrow(model_data)
      )
    }
  }
}

conventional_df <- bind_rows(conventional_rows)
wcb_df <- bind_rows(wcb_rows)
random_intercept_df <- bind_rows(random_intercept_rows)

readr::write_csv(conventional_df, file.path(results_dir, "renter_conventional_clustered.csv"))
readr::write_csv(wcb_df, file.path(results_dir, "renter_wild_cluster_bootstrap.csv"))
readr::write_csv(random_intercept_df, file.path(results_dir, "renter_random_intercepts.csv"))

message("Cluster inference analyses completed. Outputs saved to ", results_dir)
