-- =============================================================================
-- 01_staging_harmonize.sql
-- Lớp Staging: Chuẩn hóa danh mục LMICs, khử trùng ILOSTAT LFS/Modeled,
-- tính biến trễ WDI (t-1) và chỉ số sẵn sàng công nghệ (GitHub IG + Anthropic AEI).
-- =============================================================================

CREATE OR REPLACE TABLE stg_countries_lmics AS
SELECT
    UPPER(TRIM(iso3)) AS iso3,
    UPPER(TRIM(iso2)) AS iso2,
    TRIM(country_name) AS country_name,
    UPPER(TRIM(income_level)) AS income_level,
    TRIM(income_name) AS income_name,
    TRIM(region) AS region
FROM raw_countries_lmics
WHERE iso3 IS NOT NULL AND LENGTH(TRIM(iso3)) = 3;

CREATE OR REPLACE TABLE stg_ilo_exposure_isco1 AS
SELECT
    CAST(isco1 AS VARCHAR) AS isco1,
    TRIM(label1d) AS occupation_group,
    CAST(occ_count AS INTEGER) AS occ_count,
    CAST(task_count AS INTEGER) AS task_count,
    CAST(exposure_2025 AS DOUBLE) AS exposure_2025,
    CAST(exposure_2025_median AS DOUBLE) AS exposure_2025_median,
    CAST(exposure_2025_p25 AS DOUBLE) AS exposure_2025_p25,
    CAST(exposure_2025_p75 AS DOUBLE) AS exposure_2025_p75,
    CAST(exposure_2025_sd AS DOUBLE) AS exposure_2025_sd,
    CAST(task_score_2025_mean AS DOUBLE) AS task_score_2025_mean,
    CAST(exposure_2025_gpt4o AS DOUBLE) AS exposure_2025_gpt4o,
    CAST(exposure_2025_gemini AS DOUBLE) AS exposure_2025_gemini,
    CAST(exposure_2023 AS DOUBLE) AS exposure_2023,
    CAST(share_exposed_grad34 AS DOUBLE) AS share_exposed_grad34,
    CAST(share_exposed_grad12 AS DOUBLE) AS share_exposed_grad12,
    CAST(share_minimal_25 AS DOUBLE) AS share_minimal_25,
    CAST(share_not_exposed AS DOUBLE) AS share_not_exposed,
    CAST(share_automation_23 AS DOUBLE) AS share_automation_23,
    CAST(share_augmentation_23 AS DOUBLE) AS share_augmentation_23
FROM raw_ilo_exposure_isco1
WHERE CAST(isco1 AS VARCHAR) BETWEEN '1' AND '9';

-- Khử trùng tất định khóa (iso3, isco1, sex, year) trên Actual LFS:
-- Ưu tiên mã nguồn (source) có độ dài chuỗi năm lớn nhất trong cùng quốc gia.
CREATE OR REPLACE TABLE stg_ilostat_lfs AS
WITH parsed AS (
    SELECT
        UPPER(TRIM(iso3)) AS iso3,
        REPLACE(TRIM(classif1), 'OCU_ISCO08_', '') AS isco1,
        UPPER(TRIM(sex)) AS sex,
        CAST(year AS INTEGER) AS year,
        TRIM(source) AS source,
        CAST(obs_value AS DOUBLE) AS emp_thousands,
        obs_status
    FROM raw_ilostat_lfs
    WHERE regexp_matches(TRIM(classif1), '^OCU_ISCO08_[1-9]$')
      AND UPPER(TRIM(sex)) IN ('SEX_T', 'SEX_M', 'SEX_F')
      AND year BETWEEN 2015 AND 2025
      AND obs_value > 0
),
source_span AS (
    SELECT
        iso3,
        source,
        COUNT(DISTINCT year) AS span_years
    FROM parsed
    WHERE sex = 'SEX_T'
    GROUP BY iso3, source
),
ranked AS (
    SELECT
        p.*,
        COALESCE(s.span_years, 0) AS span_years,
        ROW_NUMBER() OVER (
            PARTITION BY p.iso3, p.isco1, p.sex, p.year
            ORDER BY COALESCE(s.span_years, 0) DESC, p.source ASC
        ) AS rn
    FROM parsed p
    LEFT JOIN source_span s
      ON p.iso3 = s.iso3 AND p.source = s.source
)
SELECT
    iso3,
    isco1,
    sex,
    year,
    source,
    emp_thousands,
    obs_status,
    1 AS is_actual_lfs
FROM ranked
WHERE rn = 1;

-- Khử trùng tất định cho ILO Modeled Estimates (dùng đối chiếu độ nhạy)
CREATE OR REPLACE TABLE stg_ilostat_modeled AS
WITH parsed AS (
    SELECT
        UPPER(TRIM(iso3)) AS iso3,
        REPLACE(TRIM(classif1), 'OCU_ISCO08_', '') AS isco1,
        UPPER(TRIM(sex)) AS sex,
        CAST(year AS INTEGER) AS year,
        TRIM(source) AS source,
        CAST(obs_value AS DOUBLE) AS emp_thousands,
        obs_status
    FROM raw_ilostat_modeled
    WHERE regexp_matches(TRIM(classif1), '^OCU_ISCO08_[1-9]$')
      AND UPPER(TRIM(sex)) IN ('SEX_T', 'SEX_M', 'SEX_F')
      AND year BETWEEN 2015 AND 2025
      AND obs_value > 0
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY iso3, isco1, sex, year
            ORDER BY source ASC
        ) AS rn
    FROM parsed
)
SELECT
    iso3,
    isco1,
    sex,
    year,
    source,
    emp_thousands,
    obs_status,
    0 AS is_actual_lfs
FROM ranked
WHERE rn = 1;

-- World Bank WDI kèm biến trễ (t-1) bằng Window Function LAG() đồng bộ NCKH
CREATE OR REPLACE TABLE stg_wdi_macro AS
WITH base_wdi AS (
    SELECT
        UPPER(TRIM(iso3)) AS iso3,
        CAST(year AS INTEGER) AS year,
        unemployment,
        gdp_growth,
        inflation_cpi,
        fdi_net_inflows_pct_gdp,
        trade_pct_gdp,
        female_lfpr,
        credit_private_pct_gdp,
        gdp_per_capita_const,
        LN(NULLIF(gdp_per_capita_const, 0)) AS log_gdp_pc,
        working_age_pop
    FROM raw_wdi_macro
    WHERE year BETWEEN 2014 AND 2025
),
with_lags AS (
    SELECT
        *,
        LAG(unemployment) OVER (PARTITION BY iso3 ORDER BY year) AS lag_unemployment,
        LAG(gdp_growth) OVER (PARTITION BY iso3 ORDER BY year) AS lag_gdp_growth,
        LAG(inflation_cpi) OVER (PARTITION BY iso3 ORDER BY year) AS lag_inflation_cpi,
        LAG(fdi_net_inflows_pct_gdp) OVER (PARTITION BY iso3 ORDER BY year) AS lag_fdi_pct_gdp,
        LAG(trade_pct_gdp) OVER (PARTITION BY iso3 ORDER BY year) AS lag_trade_pct_gdp,
        LAG(female_lfpr) OVER (PARTITION BY iso3 ORDER BY year) AS lag_female_lfpr,
        LAG(credit_private_pct_gdp) OVER (PARTITION BY iso3 ORDER BY year) AS lag_credit_private_pct_gdp,
        LAG(log_gdp_pc) OVER (PARTITION BY iso3 ORDER BY year) AS lag_log_gdp_pc
    FROM base_wdi
)
SELECT *
FROM with_lags
WHERE year BETWEEN 2015 AND 2025;

-- Chỉ số sẵn sàng công nghệ/AI cấp quốc gia LMICs (GitHub Innovation Graph + Anthropic AEI)
CREATE OR REPLACE TABLE stg_tech_readiness AS
WITH latest_wap AS (
    SELECT
        iso3,
        working_age_pop,
        ROW_NUMBER() OVER (PARTITION BY iso3 ORDER BY year DESC) AS rn
    FROM stg_wdi_macro
    WHERE working_age_pop IS NOT NULL AND working_age_pop > 0
),
wap_clean AS (
    SELECT
        iso3,
        working_age_pop,
        working_age_pop * 100.0 / SUM(working_age_pop) OVER () AS lmic_wap_share_pct
    FROM latest_wap
    WHERE rn = 1
),
world_wap AS (
    SELECT working_age_pop AS world_working_age_pop
    FROM raw_wdi_world_wap
    WHERE working_age_pop IS NOT NULL AND working_age_pop > 0
    ORDER BY year DESC
    LIMIT 1
),
gh_latest AS (
    SELECT
        iso3,
        year AS gh_year,
        developers_year_end,
        git_pushes_annual,
        dev_yoy_growth,
        ROW_NUMBER() OVER (PARTITION BY iso3 ORDER BY year DESC) AS rn
    FROM raw_github_annual
    WHERE year <= 2025 AND developers_year_end IS NOT NULL
),
gh_2022 AS (
    SELECT iso3, developers_year_end AS developers_2022
    FROM raw_github_annual
    WHERE year = 2022
),
aei_with_lmic_share AS (
    SELECT
        *,
        COALESCE(usage_count, 0.0) * 100.0 / NULLIF(SUM(COALESCE(usage_count, 0.0)) OVER (), 0) AS aei_lmic_usage_share_pct
    FROM raw_aei_country
),
combined AS (
    SELECT
        c.iso3,
        c.iso2,
        c.country_name,
        c.income_level,
        w.working_age_pop,
        w.lmic_wap_share_pct,
        w.working_age_pop * 100.0 / NULLIF(ww.world_working_age_pop, 0) AS world_wap_share_pct,
        g.gh_year,
        g.developers_year_end AS gh_developers,
        g.git_pushes_annual AS gh_git_pushes,
        g.dev_yoy_growth AS gh_dev_yoy_growth,
        CASE
            WHEN g22.developers_2022 > 0 AND g.developers_year_end IS NOT NULL
            THEN (g.developers_year_end - g22.developers_2022) * 1.0 / g22.developers_2022
            ELSE NULL
        END AS gh_dev_growth_since_2022,
        g.developers_year_end * 10000.0 / NULLIF(w.working_age_pop, 0) AS devs_per_10k_wap,
        COALESCE(a.usage_count, 0.0) AS aei_usage_count,
        COALESCE(a.usage_pct, 0.0) AS aei_usage_pct,
        COALESCE(a.aei_lmic_usage_share_pct, 0.0) AS aei_lmic_usage_share_pct,
        a.automation_pct AS aei_automation_pct,
        a.augmentation_pct AS aei_augmentation_pct,
        CASE
            WHEN a.usage_pct IS NOT NULL AND w.working_age_pop > 0 AND ww.world_working_age_pop > 0
            THEN a.usage_pct / (w.working_age_pop * 100.0 / ww.world_working_age_pop)
            ELSE NULL
        END AS aui_global,
        CASE
            WHEN a.aei_lmic_usage_share_pct IS NOT NULL AND w.lmic_wap_share_pct > 0
            THEN a.aei_lmic_usage_share_pct / w.lmic_wap_share_pct
            ELSE NULL
        END AS aui_lmic_relative
    FROM stg_countries_lmics c
    CROSS JOIN world_wap ww
    LEFT JOIN wap_clean w ON c.iso3 = w.iso3
    LEFT JOIN gh_latest g ON c.iso3 = g.iso3 AND g.rn = 1
    LEFT JOIN gh_2022 g22 ON c.iso3 = g22.iso3
    LEFT JOIN aei_with_lmic_share a ON c.iso3 = a.iso3
),
med AS (
    SELECT MEDIAN(devs_per_10k_wap) AS median_devs_per_10k
    FROM combined
    WHERE devs_per_10k_wap IS NOT NULL
)
SELECT
    cb.*,
    m.median_devs_per_10k,
    CASE
        WHEN cb.devs_per_10k_wap IS NULL THEN NULL
        WHEN cb.devs_per_10k_wap >= m.median_devs_per_10k THEN 1
        ELSE 0
    END AS high_tech_readiness
FROM combined cb
CROSS JOIN med m;

-- Khử trùng tất định bảng Nhóm tuổi x Nghề ISCO-08 (Youth 15-24 vs Prime-Age 25-54)
CREATE OR REPLACE TABLE stg_ilostat_age_ocu AS
WITH parsed AS (
    SELECT
        UPPER(TRIM(iso3)) AS iso3,
        TRIM(age_group) AS age_group,
        REPLACE(TRIM(classif1), 'OCU_ISCO08_', '') AS isco1,
        UPPER(TRIM(sex)) AS sex,
        CAST(year AS INTEGER) AS year,
        TRIM(source) AS source,
        CAST(obs_value AS DOUBLE) AS emp_thousands
    FROM raw_ilostat_age_ocu
    WHERE regexp_matches(TRIM(classif1), '^OCU_ISCO08_[1-9]$')
      AND UPPER(TRIM(sex)) = 'SEX_T'
      AND TRIM(age_group) IN ('AGE_AGGREGATE_Y15-24', 'AGE_AGGREGATE_Y25-54', 'AGE_AGGREGATE_Y55-64')
      AND year BETWEEN 2015 AND 2025
      AND obs_value > 0
),
source_span AS (
    SELECT iso3, source, COUNT(DISTINCT year) AS span_years
    FROM parsed
    GROUP BY iso3, source
),
ranked AS (
    SELECT
        p.*,
        ROW_NUMBER() OVER (
            PARTITION BY p.iso3, p.age_group, p.isco1, p.year
            ORDER BY COALESCE(s.span_years, 0) DESC, p.source ASC
        ) AS rn
    FROM parsed p
    LEFT JOIN source_span s ON p.iso3 = s.iso3 AND p.source = s.source
)
SELECT iso3, age_group, isco1, sex, year, source, emp_thousands
FROM ranked
WHERE rn = 1;

-- Khử trùng tất định bảng Giờ làm việc trung bình / tuần theo Nghề ISCO-08
CREATE OR REPLACE TABLE stg_ilostat_hours_ocu AS
WITH parsed AS (
    SELECT
        UPPER(TRIM(iso3)) AS iso3,
        REPLACE(TRIM(classif1), 'OCU_ISCO08_', '') AS isco1,
        UPPER(TRIM(sex)) AS sex,
        CAST(year AS INTEGER) AS year,
        TRIM(source) AS source,
        CAST(obs_value AS DOUBLE) AS weekly_hours
    FROM raw_ilostat_hours_ocu
    WHERE regexp_matches(TRIM(classif1), '^OCU_ISCO08_[1-9]$')
      AND UPPER(TRIM(sex)) = 'SEX_T'
      AND year BETWEEN 2015 AND 2025
      AND obs_value > 0
),
source_span AS (
    SELECT iso3, source, COUNT(DISTINCT year) AS span_years
    FROM parsed
    GROUP BY iso3, source
),
ranked AS (
    SELECT
        p.*,
        ROW_NUMBER() OVER (
            PARTITION BY p.iso3, p.isco1, p.year
            ORDER BY COALESCE(s.span_years, 0) DESC, p.source ASC
        ) AS rn
    FROM parsed p
    LEFT JOIN source_span s ON p.iso3 = s.iso3 AND p.source = s.source
)
SELECT iso3, isco1, sex, year, source, weekly_hours
FROM ranked
WHERE rn = 1;

-- Khử trùng tất định bảng Tiền lương danh nghĩa trung bình tháng (LCU) theo Nghề ISCO-08
CREATE OR REPLACE TABLE stg_ilostat_earnings_ocu AS
WITH parsed AS (
    SELECT
        UPPER(TRIM(iso3)) AS iso3,
        REPLACE(TRIM(classif1), 'OCU_ISCO08_', '') AS isco1,
        UPPER(TRIM(sex)) AS sex,
        CAST(year AS INTEGER) AS year,
        TRIM(source) AS source,
        CAST(obs_value AS DOUBLE) AS monthly_earnings_lcu
    FROM raw_ilostat_earnings_ocu
    WHERE regexp_matches(TRIM(classif1), '^OCU_ISCO08_[1-9]$')
      AND UPPER(TRIM(sex)) = 'SEX_T'
      AND year BETWEEN 2015 AND 2025
      AND obs_value > 0
),
source_span AS (
    SELECT iso3, source, COUNT(DISTINCT year) AS span_years
    FROM parsed
    GROUP BY iso3, source
),
ranked AS (
    SELECT
        p.*,
        ROW_NUMBER() OVER (
            PARTITION BY p.iso3, p.isco1, p.year
            ORDER BY COALESCE(s.span_years, 0) DESC, p.source ASC
        ) AS rn
    FROM parsed p
    LEFT JOIN source_span s ON p.iso3 = s.iso3 AND p.source = s.source
)
SELECT iso3, isco1, sex, year, source, monthly_earnings_lcu
FROM ranked
WHERE rn = 1;
