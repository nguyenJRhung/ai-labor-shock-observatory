# ==============================================================================
# Cross-Check Causal Econometric Estimates in R (fixest::feols)
# Compares R fixest::feols against Python pyfixest results on identical data.
# ==============================================================================

suppressPackageStartupMessages({
  library(fixest)
  library(dplyr)
  library(readr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) > 0) dirname(sub("^--file=", "", file_arg[1])) else "src/causal"
root_dir <- normalizePath(file.path(script_dir, "..", ".."), winslash = "/", mustWork = FALSE)

panel_path <- file.path(root_dir, "data", "processed", "mart_panel_causal_lfs.csv")
py_json_path <- file.path(root_dir, "data", "probes", "causal_results.json")
out_json_path <- file.path(root_dir, "data", "probes", "r_fixest_verification.json")

stopifnot(file.exists(panel_path))

df_total <- read_csv(panel_path, show_col_types = FALSE) %>%
  filter(sex == "SEX_T", !is.na(exposure_2025), !is.na(ln_emp)) %>%
  mutate(exposure = as.numeric(exposure_2025))

# 1. Event Study (ref = 2022) with unit_fe and iso3 clustering
es_unit <- feols(ln_emp ~ i(year, exposure, ref = 2022) | unit_fe + country_year_fe,
                 data = df_total, cluster = ~unit_fe)
es_iso  <- feols(ln_emp ~ i(year, exposure, ref = 2022) | unit_fe + country_year_fe,
                 data = df_total, cluster = ~iso3)

# 2. Pooled Post-2022 DiD
did_unit <- feols(ln_emp ~ post_2022:exposure | unit_fe + country_year_fe,
                  data = df_total, cluster = ~unit_fe)
did_iso  <- feols(ln_emp ~ post_2022:exposure | unit_fe + country_year_fe,
                  data = df_total, cluster = ~iso3)

es_ct <- coeftable(es_unit)
es_rows <- lapply(seq_len(nrow(es_ct)), function(i) {
  nm <- rownames(es_ct)[i]
  yr <- as.integer(sub("year::([0-9]+):exposure", "\\1", nm))
  list(
    year = yr,
    term = nm,
    estimate = unname(es_ct[i, "Estimate"]),
    se_unit = unname(es_ct[i, "Std. Error"]),
    pvalue_unit = unname(es_ct[i, "Pr(>|t|)"]),
    se_iso = unname(se(es_iso)[i])
  )
})

did_ct_u <- coeftable(did_unit)
did_ct_c <- coeftable(did_iso)

comparison <- list(available = FALSE)
if (file.exists(py_json_path)) {
  py_res <- fromJSON(py_json_path, simplifyVector = FALSE)
  py_es <- py_res$main_lfs_ilo2025$event_study_unit_cluster
  py_did <- py_res$main_lfs_ilo2025$pooled_did_unit_cluster

  max_coef_diff <- 0.0
  max_se_diff <- 0.0
  for (r_row in es_rows) {
    for (p_row in py_es) {
      if (p_row$year == r_row$year) {
        max_coef_diff <- max(max_coef_diff, abs(r_row$estimate - p_row$estimate))
        max_se_diff   <- max(max_se_diff,   abs(r_row$se_unit - p_row$se))
      }
    }
  }
  did_coef_diff <- abs(unname(did_ct_u[1, "Estimate"]) - py_did$estimate)
  did_se_diff   <- abs(unname(did_ct_u[1, "Std. Error"]) - py_did$se)
  max_coef_diff <- max(max_coef_diff, did_coef_diff)
  max_se_diff   <- max(max_se_diff, did_se_diff)

  comparison <- list(
    available = TRUE,
    n_obs_r = nobs(es_unit),
    n_obs_python = py_res$main_lfs_ilo2025$n_obs,
    max_abs_coef_diff = max_coef_diff,
    max_abs_se_diff = max_se_diff,
    did_coef_diff = did_coef_diff,
    did_se_diff = did_se_diff,
    exact_match_pass = (max_coef_diff < 1e-5) && (max_se_diff < 1e-3)
  )
}

out_obj <- list(
  verified_at_utc = format(Sys.time(), "%Y-%m-%d %H:%M:%S UTC", tz = "UTC"),
  r_version = R.version.string,
  fixest_version = as.character(packageVersion("fixest")),
  n_obs = nobs(es_unit),
  event_study = es_rows,
  pooled_did = list(
    estimate = unname(did_ct_u[1, "Estimate"]),
    se_unit_cluster = unname(did_ct_u[1, "Std. Error"]),
    pvalue_unit_cluster = unname(did_ct_u[1, "Pr(>|t|)"]),
    se_country_cluster = unname(did_ct_c[1, "Std. Error"]),
    pvalue_country_cluster = unname(did_ct_c[1, "Pr(>|t|)"])
  ),
  python_vs_r_comparison = comparison
)

write_json(out_obj, out_json_path, auto_unbox = TRUE, pretty = TRUE, digits = 10)
cat("[R fixest] Verification written to:", out_json_path, "\n")
if (isTRUE(comparison$available)) {
  cat(sprintf("[R vs Python] max |coef diff| = %.3e | max |se diff| = %.3e | PASS = %s\n",
              comparison$max_abs_coef_diff, comparison$max_abs_se_diff, comparison$exact_match_pass))
}
