#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(clubSandwich)
  library(broom)
  library(mice)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  script_path <- "src/analysis/analyze_insurance_models.R"
}
base_dir <- normalizePath(file.path(dirname(script_path), "..", ".."))
data_path <- file.path(base_dir, "data", "processed", "insurance_model_dataset.csv")
results_dir <- file.path(base_dir, "results", "insurance")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

read_dataset <- function(path) {
  df <- readr::read_csv(path, show_col_types = FALSE)
  df %>%
    mutate(
      renter_flag = if_else(tenure == "Renter", 1, 0),
      homeowner_flag = if_else(tenure == "Homeowner", 1, 0),
      hazard_category = factor(hazard_category),
      state_abbr = if_else(is.na(state_abbr) | state_abbr == "", "Unknown", state_abbr),
      state_abbr = factor(state_abbr),
      hazard_class = factor(hazard_class, levels = c("flood", "wind", "winter", "other", "fire")),
      coastal_county_flag = as.numeric(coastal_county_flag),
      aid_flag = as.numeric(aid_flag),
      displacement_flag = as.numeric(displacement_flag),
      employed = as.numeric(employed),
      female = as.numeric(female),
      nonwhite = as.numeric(nonwhite),
      hispanic = as.numeric(hispanic),
      married = as.numeric(married),
      mfr_flag = as.numeric(mfr_flag)
    )
}

form_string <- function(outcome, include_renter = TRUE) {
  base_terms <- c(
    if (include_renter) "renter_flag" else NULL,
    "income_ord", "damage_ord", "aid_flag", "displacement_flag",
    "age_years", "household_size", "employed",
    "attachment_belonging", "move_intent_disamenity", "trust_safety",
    "social_cohesion", "quality_of_life", "female", "nonwhite",
    "hispanic", "education_ord", "married", "mfr_flag",
    "distance_to_gulf_km", "coastal_county_flag",
    "factor(state_abbr)"
  )
  stats::as.formula(paste(outcome, "~", paste(base_terms, collapse = " + ")))
}

fit_glm_cr2 <- function(formula, data, cluster_var) {
  required_vars <- unique(c(all.vars(formula), cluster_var))
  clean <- data %>%
    select(any_of(required_vars)) %>%
    drop_na()
  model <- glm(formula, data = clean, family = binomial())
  cluster_vals <- clean[[cluster_var]]
  vcov_cr2 <- clubSandwich::vcovCR(model, cluster = cluster_vals, type = "CR2")
  list(model = model, vcov = vcov_cr2, data = clean)
}

tidy_or <- function(model, vcov_mat, term) {
  tt <- clubSandwich::coef_test(model, vcov = vcov_mat, test = "Satterthwaite")
  term_col <- if ("coef" %in% names(tt)) "coef" else if ("Coef" %in% names(tt)) "Coef" else stop("Coefficient column not found.")
  row <- tt[tt[[term_col]] == term, , drop = FALSE]
  if (nrow(row) == 0) {
    return(tibble(term = term, estimate = NA_real_, std_error = NA_real_, df = NA_real_, p_value = NA_real_, or = NA_real_, or_lwr = NA_real_, or_upr = NA_real_))
  }
  df_val <- row$df_Satt
  crit <- qt(0.975, df = df_val)
  tibble(
    term = term,
    estimate = row$beta,
    std_error = row$SE,
    df = df_val,
    p_value = row$p_Satt,
    or = exp(row$beta),
    or_lwr = exp(row$beta - crit * row$SE),
    or_upr = exp(row$beta + crit * row$SE)
  )
}

predicted_probabilities <- function(fit, renter_values = c(0, 1)) {
  base_data <- fit$data
  if (!("renter_flag" %in% names(base_data))) {
    return(tibble())
  }
  map_dfr(
    renter_values,
    function(val) {
      newdata <- base_data
      newdata$renter_flag <- val
      preds <- predict(fit$model, newdata = newdata, type = "response")
      tibble(renter_flag = val, predicted_prob = mean(preds))
    }
  )
}

hazard_levels <- c("flood", "wind", "winter")

dataset <- read_dataset(data_path)

# Pooled models
pooled_data <- dataset %>%
  filter(!is.na(coverage_peril_aligned), !is.na(coverage_original))

pooled_formula_current <- form_string("coverage_original", include_renter = TRUE)
pooled_formula_aligned <- form_string("coverage_peril_aligned", include_renter = TRUE)

pooled_current <- fit_glm_cr2(pooled_formula_current, pooled_data, "hazard_category")
pooled_aligned <- fit_glm_cr2(pooled_formula_aligned, pooled_data, "hazard_category")

pooled_results <- bind_rows(
  tidy_or(pooled_current$model, pooled_current$vcov, "renter_flag") %>%
    mutate(definition = "current"),
  tidy_or(pooled_aligned$model, pooled_aligned$vcov, "renter_flag") %>%
    mutate(definition = "peril_aligned")
) %>%
  mutate(
    pooled_renter_prob = map_dbl(definition, function(defn) {
      fit <- if (defn == "current") pooled_current else pooled_aligned
      predicted_probabilities(fit) %>%
        filter(renter_flag == 1) %>%
        pull(predicted_prob)
    }),
    pooled_owner_prob = map_dbl(definition, function(defn) {
      fit <- if (defn == "current") pooled_current else pooled_aligned
      predicted_probabilities(fit) %>%
        filter(renter_flag == 0) %>%
        pull(predicted_prob)
    }),
    risk_difference = pooled_renter_prob - pooled_owner_prob
  )

readr::write_csv(pooled_results, file.path(results_dir, "pooled_insurance_results.csv"))

full_aligned <- clubSandwich::coef_test(pooled_aligned$model, vcov = pooled_aligned$vcov, test = "Satterthwaite") %>%
  mutate(
    odds_ratio = exp(beta),
    or_lwr = exp(beta - qt(0.975, df_Satt) * SE),
    or_upr = exp(beta + qt(0.975, df_Satt) * SE)
  )
readr::write_csv(full_aligned, file.path(results_dir, "pooled_peril_aligned_coefficients.csv"))

# Hazard-specific models
hazard_results <- map_dfr(
  hazard_levels,
  function(hazard) {
    subset <- pooled_data %>%
      filter(hazard_class == hazard, !is.na(coverage_peril_aligned))
    if (nrow(subset) < 20) {
      return(tibble())
    }
    formula <- form_string("coverage_peril_aligned", include_renter = TRUE)
    cluster_var <- if (hazard == "wind") "hazard_category" else "state_abbr"
    fit <- fit_glm_cr2(formula, subset, cluster_var)
    probs <- predicted_probabilities(fit)
    tidy_or(fit$model, fit$vcov, "renter_flag") %>%
      mutate(
        hazard_class = hazard,
        renter_prob = probs %>% filter(renter_flag == 1) %>% pull(predicted_prob),
        owner_prob = probs %>% filter(renter_flag == 0) %>% pull(predicted_prob),
        risk_difference = renter_prob - owner_prob,
        cluster_var = cluster_var
      )
  }
)

readr::write_csv(hazard_results, file.path(results_dir, "hazard_class_results.csv"))

# Tenure-specific models
owner_flood <- dataset %>%
  filter(tenure == "Homeowner", hazard_class == "flood")

renter_wind_winter <- dataset %>%
  filter(tenure == "Renter", hazard_class %in% c("wind", "winter"))

renter_flood <- dataset %>%
  filter(tenure == "Renter", hazard_class == "flood")

fit_owner <- fit_glm_cr2(
  form_string("owner_flood_policy", include_renter = FALSE),
  owner_flood,
  "state_abbr"
)

fit_renter_standard <- fit_glm_cr2(
  form_string("renter_policy_indicator", include_renter = FALSE),
  renter_wind_winter,
  "hazard_category"
)

fit_renter_flood <- fit_glm_cr2(
  form_string("renter_flood_policy", include_renter = FALSE),
  renter_flood,
  "state_abbr"
)

tenure_probs <- bind_rows(
  tibble(
    group = "owners_flood",
    coverage = "owner_flood_policy",
    n = nrow(owner_flood),
    predicted = mean(predict(fit_owner$model, type = "response"))
  ),
  tibble(
    group = "renters_wind_winter",
    coverage = "renter_policy_indicator",
    n = nrow(renter_wind_winter),
    predicted = mean(predict(fit_renter_standard$model, type = "response"))
  ),
  tibble(
    group = "renters_flood",
    coverage = "renter_flood_policy",
    n = nrow(renter_flood),
    predicted = mean(predict(fit_renter_flood$model, type = "response"))
  )
)

readr::write_csv(tenure_probs, file.path(results_dir, "tenure_specific_predictions.csv"))

# Missingness summary by tenure
missing_vars <- c(
  "coverage_peril_aligned", "income_ord", "damage_ord", "aid_flag",
  "displacement_flag", "age_years", "household_size", "employed",
  "attachment_belonging", "move_intent_disamenity", "trust_safety",
  "social_cohesion", "quality_of_life", "female", "nonwhite",
  "hispanic", "education_ord", "married", "mfr_flag",
  "distance_to_gulf_km", "coastal_county_flag"
)

missingness <- dataset %>%
  mutate(tenure = factor(tenure)) %>%
  group_by(tenure) %>%
  summarise(across(all_of(missing_vars), ~ mean(is.na(.x))), .groups = "drop") %>%
  pivot_longer(-tenure, names_to = "variable", values_to = "missing_rate")

readr::write_csv(missingness, file.path(results_dir, "missingness_by_tenure.csv"))

# Multiple imputation with MICE
imputation_vars <- c(
  "coverage_peril_aligned", "renter_flag",
  "income_ord", "damage_ord", "aid_flag", "displacement_flag",
  "age_years", "household_size", "employed",
  "attachment_belonging", "move_intent_disamenity", "trust_safety",
  "social_cohesion", "quality_of_life", "female", "nonwhite",
  "hispanic", "education_ord", "married", "mfr_flag",
  "distance_to_gulf_km", "coastal_county_flag", "state_abbr"
)

mi_data <- dataset %>%
  select(all_of(imputation_vars)) %>%
  mutate(across(c(renter_flag, aid_flag, displacement_flag, employed, female, nonwhite, hispanic, married, mfr_flag, coastal_county_flag), as.integer))

methods <- make.method(mi_data)
methods[c("coverage_peril_aligned", "renter_flag", "aid_flag", "displacement_flag", "employed", "female", "nonwhite", "hispanic", "married", "mfr_flag", "coastal_county_flag")] <- "logreg"
methods["state_abbr"] <- "polyreg"
methods[c("income_ord", "damage_ord", "age_years", "household_size", "attachment_belonging", "move_intent_disamenity", "trust_safety", "social_cohesion", "quality_of_life", "distance_to_gulf_km", "education_ord")] <- "pmm"

predictors <- make.predictorMatrix(mi_data)
predictors[, "coverage_peril_aligned"] <- 1
predictors["coverage_peril_aligned", ] <- 1
predictors["state_abbr", "state_abbr"] <- 0
diag(predictors) <- 0

mice_fit <- mice(mi_data, m = 20, method = methods, predictorMatrix = predictors, maxit = 10, printFlag = FALSE)

mi_formula <- form_string("coverage_peril_aligned", include_renter = TRUE)
mi_models <- lapply(seq_len(mice_fit$m), function(i) {
  data_i <- complete(mice_fit, action = i)
  glm(mi_formula, data = data_i, family = binomial())
})
mi_pool <- summary(pool(as.mira(mi_models)), conf.int = TRUE, exponentiate = TRUE)

readr::write_csv(mi_pool, file.path(results_dir, "mi_pooled_results.csv"))

cat("Insurance analyses completed. Outputs stored in results/insurance.\n")
