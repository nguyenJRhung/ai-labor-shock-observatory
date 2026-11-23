# ==============================================================================
# Cross-Check 5 Causal Econometric Models in R (fixest::feols) vs Python pyfixest
# Models: (1) Event Study, (2) Pooled DiD, (3) Occupation-Detrended DiD,
#         (4) Binary Clerical vs Manual DiD, (5) Triple-Diff Gender
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

df_all <- read_csv(panel_path, show_col_types = FALSE) %>%
  mutate(
    isco1 = as.character(isco1),
    exposure = as.numeric(exposure_2025),
    year_c = year - 2022L
  )

df_total <- df_all %>%
  filter(sex == "SEX_T", !is.na(exposure), !is.na(ln_emp))

# 1. Event Study (ref = 2022)
es_unit <- feols(ln_emp ~ i(year, exposure, ref = 2022) | unit_fe + country_year_fe,
                 data = df_total, cluster = ~unit_fe)
es_iso  <- feols(ln_emp ~ i(year, exposure, ref = 2022) | unit_fe + country_year_fe,
                 data = df_total, cluster = ~iso3)

# 2. Pooled Post-2022 DiD
did_unit <- feols(ln_emp ~ post_2022:exposure | unit_fe + country_year_fe,
                  data = df_total, cluster = ~unit_fe)
did_iso  <- feols(ln_emp ~ post_2022:exposure | unit_fe + country_year_fe,
                  data = df_total, cluster = ~iso3)

# 3. Occupation-Detrended DiD (ref ISCO-9)
det_unit <- feols(ln_emp ~ post_2022:exposure + i(isco1, year_c, ref = "9") | unit_fe + country_year_fe,
                  data = df_total, cluster = ~unit_fe)

# 4. Binary DiD: Clerical (ISCO-4) vs Manual (ISCO-7,8,9)
df_bin <- df_total %>%
  filter(isco1 %in% c("4", "7", "8", "9")) %>%
  mutate(is_clerical = as.integer(isco1 == "4"))
bin_unit <- feols(ln_emp ~ post_2022:is_clerical | unit_fe + country_year_fe,
                  data = df_bin, cluster = ~unit_fe)

# 5. Triple-Diff Gender (balanced Male & Female cell pairs)
df_mf <- df_all %>%
  filter(sex %in% c("SEX_M", "SEX_F"), !is.na(exposure), !is.na(ln_emp)) %>%
  group_by(iso3, isco1, year) %>%
  filter(n_distinct(sex) == 2) %>%
  ungroup()
td_unit <- feols(ln_emp ~ post_2022:exposure + post_2022:exposure:is_female | country_occ_sex_fe + country_year_sex_fe,
                 data = df_mf, cluster = ~country_occ_sex_fe)

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
det_ct_u <- coeftable(det_unit)
bin_ct_u <- coeftable(bin_unit)
td_ct_u  <- coeftable(td_unit)

comparison <- list(available = FALSE)
if (file.exists(py_json_path)) {
  py_res <- fromJSON(py_json_path, simplifyVector = FALSE)
  py_es <- py_res$main_lfs_ilo2025$event_study_unit_cluster
  py_did <- py_res$main_lfs_ilo2025$pooled_did_unit_cluster
  py_det <- py_res$robustness_checks$`5_occupation_linear_trends`$ln_emp_unit_cluster
  py_bin <- py_res$robustness_checks$`8_binary_clerical_vs_manual`$baseline
  py_td_m <- py_res$triple_diff_gender$unit_cluster$male_post_exposure
  py_td_f <- py_res$triple_diff_gender$unit_cluster$female_differential

  es_coef_diff <- 0.0
  es_se_diff <- 0.0
  for (r_row in es_rows) {
    for (p_row in py_es) {
      if (p_row$year == r_row$year) {
        es_coef_diff <- max(es_coef_diff, abs(r_row$estimate - p_row$estimate))
        es_se_diff   <- max(es_se_diff,   abs(r_row$se_unit - p_row$se))
      }
    }
  }

  did_coef_diff <- abs(unname(did_ct_u[1, "Estimate"]) - py_did$estimate)
  did_se_diff   <- abs(unname(did_ct_u[1, "Std. Error"]) - py_did$se)

  det_coef_diff <- abs(unname(det_ct_u["post_2022:exposure", "Estimate"]) - py_det$estimate)
  det_se_diff   <- abs(unname(det_ct_u["post_2022:exposure", "Std. Error"]) - py_det$se)

  bin_coef_diff <- abs(unname(bin_ct_u["post_2022:is_clerical", "Estimate"]) - py_bin$estimate)
  bin_se_diff   <- abs(unname(bin_ct_u["post_2022:is_clerical", "Std. Error"]) - py_bin$se)

  td_coef_diff <- max(
    abs(unname(td_ct_u["post_2022:exposure", "Estimate"]) - py_td_m$estimate),
    abs(unname(td_ct_u["post_2022:exposure:is_female", "Estimate"]) - py_td_f$estimate)
  )
  td_se_diff <- max(
    abs(unname(td_ct_u["post_2022:exposure", "Std. Error"]) - py_td_m$se),
    abs(unname(td_ct_u["post_2022:exposure:is_female", "Std. Error"]) - py_td_f$se)
  )

  max_coef_diff <- max(es_coef_diff, did_coef_diff, det_coef_diff, bin_coef_diff, td_coef_diff)
  max_se_diff   <- max(es_se_diff,   did_se_diff,   det_se_diff,   bin_se_diff,   td_se_diff)

  comparison <- list(
    available = TRUE,
    models_verified_count = 5L,
    n_obs_r = nobs(es_unit),
    n_obs_python = py_res$main_lfs_ilo2025$n_obs,
    per_model_diffs = list(
      event_study = list(max_coef_diff = es_coef_diff, max_se_diff = es_se_diff),
      pooled_did = list(max_coef_diff = did_coef_diff, max_se_diff = did_se_diff),
      occ_linear_trends = list(max_coef_diff = det_coef_diff, max_se_diff = det_se_diff),
      binary_clerical_vs_manual = list(max_coef_diff = bin_coef_diff, max_se_diff = bin_se_diff),
      triple_diff_gender = list(max_coef_diff = td_coef_diff, max_se_diff = td_se_diff)
    ),
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
  occ_linear_trends = list(
    estimate = unname(det_ct_u["post_2022:exposure", "Estimate"]),
    se_unit_cluster = unname(det_ct_u["post_2022:exposure", "Std. Error"]),
    pvalue_unit_cluster = unname(det_ct_u["post_2022:exposure", "Pr(>|t|)"])
  ),
  binary_clerical_vs_manual = list(
    n_obs = nobs(bin_unit),
    estimate = unname(bin_ct_u["post_2022:is_clerical", "Estimate"]),
    se_unit_cluster = unname(bin_ct_u["post_2022:is_clerical", "Std. Error"]),
    pvalue_unit_cluster = unname(bin_ct_u["post_2022:is_clerical", "Pr(>|t|)"])
  ),
  triple_diff_gender = list(
    n_obs = nobs(td_unit),
    male_estimate = unname(td_ct_u["post_2022:exposure", "Estimate"]),
    male_se = unname(td_ct_u["post_2022:exposure", "Std. Error"]),
    female_diff_estimate = unname(td_ct_u["post_2022:exposure:is_female", "Estimate"]),
    female_diff_se = unname(td_ct_u["post_2022:exposure:is_female", "Std. Error"])
  ),
  python_vs_r_comparison = comparison
)

write_json(out_obj, out_json_path, auto_unbox = TRUE, pretty = TRUE, digits = 12)
cat("[R fixest] 5-model verification written to:", out_json_path, "\n")
if (isTRUE(comparison$available)) {
  cat(sprintf("[R vs Python 5/5] max |coef diff| = %.3e | max |se diff| = %.3e | PASS = %s\n",
              comparison$max_abs_coef_diff, comparison$max_abs_se_diff, comparison$exact_match_pass))
}
