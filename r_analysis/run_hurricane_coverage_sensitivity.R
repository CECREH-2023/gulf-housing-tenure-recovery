#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(broom)
})

`%||%` <- function(a, b) if (!is.na(a) && length(a) > 0) a else b

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  script_path <- "src/analysis/run_hurricane_coverage_sensitivity.R"
}
base_dir <- normalizePath(file.path(dirname(script_path), "..", ".."))
data_path <- file.path(base_dir, "data", "processed", "insurance_model_dataset.csv")
results_dir <- file.path(base_dir, "results", "insurance")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

logit_summary <- function(df, outcome) {
  model <- glm(as.formula(paste(outcome, "~ renter_flag + income_ord + attachment_belonging + employed + age_years + household_size + damage_ord")),
               data = df, family = binomial())
  tidy(model, conf.int = TRUE, exponentiate = TRUE) %>%
    filter(term == "renter_flag") %>%
    transmute(odds_ratio = estimate, conf_low = conf.low, conf_high = conf.high)
}

df <- readr::read_csv(data_path, show_col_types = FALSE) %>%
  mutate(
    renter_flag = if_else(tenure == "Renter", 1, 0),
    hurricane_event = str_detect(tolower(hazard_category), "hurricane"),
    inland_hurricane = hurricane_event & coastal_county_flag == 0 & damage_ord >= 2
  )

mixed_cov <- df %>%
  mutate(
    coverage_mixed = case_when(
      hurricane_event & tenure == "Homeowner" ~ as.numeric(pmax(owner_home_policy, owner_flood_policy, na.rm = TRUE)),
      hurricane_event & tenure == "Renter" ~ as.numeric(pmax(renter_policy_indicator, renter_flood_policy, na.rm = TRUE)),
      TRUE ~ coverage_peril_aligned
    )
  )

inland_cov <- df %>%
  mutate(
    coverage_inland = case_when(
      inland_hurricane & tenure == "Homeowner" ~ owner_flood_policy,
      inland_hurricane & tenure == "Renter" ~ renter_flood_policy,
      TRUE ~ coverage_peril_aligned
    )
  )

scenario_frames <- list(
  baseline = df %>% mutate(coverage_variant = coverage_peril_aligned),
  hurricane_mixed = mixed_cov %>% mutate(coverage_variant = coverage_mixed),
  inland_reclassified = inland_cov %>% mutate(coverage_variant = coverage_inland)
)

records <- map_df(names(scenario_frames), function(name) {
  frame <- scenario_frames[[name]]
  rates <- frame %>%
    group_by(tenure) %>%
    summarise(coverage_rate = mean(coverage_variant, na.rm = TRUE), n = n(), .groups = "drop")
  renter_rate <- rates %>% filter(tenure == "Renter") %>% pull(coverage_rate)
  owner_rate <- rates %>% filter(tenure == "Homeowner") %>% pull(coverage_rate)
  gap <- renter_rate - owner_rate
  logit <- logit_summary(frame %>% drop_na(coverage_variant), "coverage_variant")
  tibble(
    scenario = name,
    renter_rate = renter_rate,
    owner_rate = owner_rate,
    gap = gap,
    renter_or = logit$odds_ratio,
    renter_or_low = logit$conf_low,
    renter_or_high = logit$conf_high
  )
})

readr::write_csv(records, file.path(results_dir, "hurricane_coverage_sensitivity.csv"))
