-- =============================================================================
-- 02_mart_panel_causal.sql
-- Lớp Mart Nhân quả: Dựng bảng dữ liệu bảng Quốc gia x Nghề ISCO-08 x Giới tính x Năm
-- kèm kiểm tra điều kiện đủ chuỗi trước & sau 2022 trên Actual LFS và Modeled.
-- =============================================================================

CREATE OR REPLACE TABLE mart_lfs_country_eligibility AS
WITH stats AS (
    SELECT
        iso3,
        COUNT(DISTINCT CASE WHEN year BETWEEN 2015 AND 2022 THEN year END) AS pre_2022_years,
        COUNT(DISTINCT CASE WHEN year BETWEEN 2023 AND 2025 THEN year END) AS post_2022_years,
        COUNT(DISTINCT isco1) AS isco1_groups_count,
        MIN(year) AS min_year,
        MAX(year) AS max_year,
        COUNT(*) AS rows_total_sex
    FROM stg_ilostat_lfs
    WHERE sex = 'SEX_T'
    GROUP BY iso3
)
SELECT
    c.iso3,
    c.country_name,
    c.income_level,
    c.region,
    COALESCE(s.pre_2022_years, 0) AS pre_2022_years,
    COALESCE(s.post_2022_years, 0) AS post_2022_years,
    COALESCE(s.isco1_groups_count, 0) AS isco1_groups_count,
    s.min_year,
    s.max_year,
    COALESCE(s.rows_total_sex, 0) AS rows_total_sex,
    CASE
        WHEN COALESCE(s.pre_2022_years, 0) >= 3
         AND COALESCE(s.post_2022_years, 0) >= 1
         AND COALESCE(s.isco1_groups_count, 0) >= 7
        THEN 1 ELSE 0
    END AS is_eligible
FROM stg_countries_lmics c
LEFT JOIN stats s ON c.iso3 = s.iso3;

CREATE OR REPLACE TABLE mart_panel_causal_lfs AS
SELECT
    l.iso3,
    c.country_name,
    c.income_level,
    c.region,
    l.isco1,
    e.occupation_group,
    l.sex,
    l.year,
    l.source,
    l.emp_thousands,
    LN(l.emp_thousands) AS ln_emp,
    l.emp_thousands / NULLIF(SUM(l.emp_thousands) OVER (PARTITION BY l.iso3, l.sex, l.year), 0) AS emp_share,
    e.exposure_2025,
    e.exposure_2025_median,
    e.exposure_2025_p25,
    e.exposure_2025_p75,
    e.task_score_2025_mean,
    e.exposure_2025_gpt4o,
    e.exposure_2025_gemini,
    e.exposure_2023,
    e.share_exposed_grad34,
    e.share_exposed_grad12,
    e.share_automation_23,
    e.share_augmentation_23,
    CASE WHEN l.year >= 2023 THEN 1 ELSE 0 END AS post_2022,
    l.year - 2022 AS event_time,
    CASE WHEN l.year IN (2020, 2021) THEN 1 ELSE 0 END AS is_covid_2020_2021,
    CASE WHEN l.sex = 'SEX_F' THEN 1 ELSE 0 END AS is_female,
    l.iso3 || '_' || l.isco1 AS unit_fe,
    l.iso3 || '_' || CAST(l.year AS VARCHAR) AS country_year_fe,
    l.iso3 || '_' || l.isco1 || '_' || l.sex AS country_occ_sex_fe,
    l.iso3 || '_' || CAST(l.year AS VARCHAR) || '_' || l.sex AS country_year_sex_fe,
    w.lag_unemployment,
    w.lag_gdp_growth,
    w.lag_inflation_cpi,
    w.lag_fdi_pct_gdp,
    w.lag_trade_pct_gdp,
    w.lag_female_lfpr,
    w.lag_credit_private_pct_gdp,
    w.lag_log_gdp_pc,
    t.devs_per_10k_wap,
    t.high_tech_readiness,
    t.aei_usage_pct,
    t.aei_automation_pct,
    t.aei_augmentation_pct,
    t.aui_global,
    t.aui_lmic_relative
FROM stg_ilostat_lfs l
INNER JOIN mart_lfs_country_eligibility el
    ON l.iso3 = el.iso3 AND el.is_eligible = 1
INNER JOIN stg_countries_lmics c
    ON l.iso3 = c.iso3
INNER JOIN stg_ilo_exposure_isco1 e
    ON l.isco1 = e.isco1
LEFT JOIN stg_wdi_macro w
    ON l.iso3 = w.iso3 AND l.year = w.year
LEFT JOIN stg_tech_readiness t
    ON l.iso3 = t.iso3;

CREATE OR REPLACE TABLE mart_panel_causal_modeled AS
SELECT
    m.iso3,
    c.country_name,
    c.income_level,
    c.region,
    m.isco1,
    e.occupation_group,
    m.sex,
    m.year,
    m.source,
    m.emp_thousands,
    LN(m.emp_thousands) AS ln_emp,
    m.emp_thousands / NULLIF(SUM(m.emp_thousands) OVER (PARTITION BY m.iso3, m.sex, m.year), 0) AS emp_share,
    e.exposure_2025,
    e.exposure_2025_median,
    e.exposure_2025_p25,
    e.exposure_2025_p75,
    e.task_score_2025_mean,
    e.exposure_2025_gpt4o,
    e.exposure_2025_gemini,
    e.exposure_2023,
    e.share_exposed_grad34,
    e.share_exposed_grad12,
    e.share_automation_23,
    e.share_augmentation_23,
    CASE WHEN m.year >= 2023 THEN 1 ELSE 0 END AS post_2022,
    m.year - 2022 AS event_time,
    CASE WHEN m.year IN (2020, 2021) THEN 1 ELSE 0 END AS is_covid_2020_2021,
    CASE WHEN m.sex = 'SEX_F' THEN 1 ELSE 0 END AS is_female,
    COALESCE(el.is_eligible, 0) AS in_lfs_sample,
    m.iso3 || '_' || m.isco1 AS unit_fe,
    m.iso3 || '_' || CAST(m.year AS VARCHAR) AS country_year_fe,
    m.iso3 || '_' || m.isco1 || '_' || m.sex AS country_occ_sex_fe,
    m.iso3 || '_' || CAST(m.year AS VARCHAR) || '_' || m.sex AS country_year_sex_fe,
    w.lag_gdp_growth,
    w.lag_inflation_cpi,
    w.lag_trade_pct_gdp,
    w.lag_log_gdp_pc,
    t.devs_per_10k_wap,
    t.high_tech_readiness
FROM stg_ilostat_modeled m
INNER JOIN stg_countries_lmics c
    ON m.iso3 = c.iso3
INNER JOIN stg_ilo_exposure_isco1 e
    ON m.isco1 = e.isco1
LEFT JOIN mart_lfs_country_eligibility el
    ON m.iso3 = el.iso3
LEFT JOIN stg_wdi_macro w
    ON m.iso3 = w.iso3 AND m.year = w.year
LEFT JOIN stg_tech_readiness t
    ON m.iso3 = t.iso3;

-- Panel ngắn hạn theo Độ tuổi (Youth 15-24 vs Prime-Age 25-54)
CREATE OR REPLACE TABLE mart_panel_youth_ocu AS
WITH age_elig AS (
    SELECT
        iso3,
        COUNT(DISTINCT CASE WHEN year BETWEEN 2015 AND 2022 THEN year END) AS pre_yrs,
        COUNT(DISTINCT CASE WHEN year BETWEEN 2023 AND 2025 THEN year END) AS post_yrs,
        COUNT(DISTINCT isco1) AS n_occ
    FROM stg_ilostat_age_ocu
    WHERE age_group = 'AGE_AGGREGATE_Y15-24'
    GROUP BY iso3
)
SELECT
    a.iso3,
    c.country_name,
    c.income_level,
    c.region,
    a.age_group,
    CASE WHEN a.age_group = 'AGE_AGGREGATE_Y15-24' THEN 1 ELSE 0 END AS is_youth,
    a.isco1,
    e.occupation_group,
    a.sex,
    a.year,
    a.source,
    a.emp_thousands,
    LN(a.emp_thousands) AS ln_emp,
    e.exposure_2025,
    e.exposure_2023,
    e.share_exposed_grad34,
    CASE WHEN a.year >= 2023 THEN 1 ELSE 0 END AS post_2022,
    a.year - 2022 AS event_time,
    a.iso3 || '_' || a.isco1 AS unit_fe,
    a.iso3 || '_' || CAST(a.year AS VARCHAR) AS country_year_fe,
    a.iso3 || '_' || a.isco1 || '_' || a.age_group AS unit_age_fe,
    a.iso3 || '_' || CAST(a.year AS VARCHAR) || '_' || a.age_group AS country_age_year_fe
FROM stg_ilostat_age_ocu a
INNER JOIN mart_lfs_country_eligibility el
    ON a.iso3 = el.iso3 AND el.is_eligible = 1
INNER JOIN age_elig ae
    ON a.iso3 = ae.iso3 AND ae.pre_yrs >= 3 AND ae.post_yrs >= 1 AND ae.n_occ >= 7
INNER JOIN stg_countries_lmics c
    ON a.iso3 = c.iso3
INNER JOIN stg_ilo_exposure_isco1 e
    ON a.isco1 = e.isco1;

-- Panel ngắn hạn Giờ làm việc trung bình / tuần (Intensive Margin)
CREATE OR REPLACE TABLE mart_panel_hours_ocu AS
WITH hrs_elig AS (
    SELECT
        iso3,
        COUNT(DISTINCT CASE WHEN year BETWEEN 2015 AND 2022 THEN year END) AS pre_yrs,
        COUNT(DISTINCT CASE WHEN year BETWEEN 2023 AND 2025 THEN year END) AS post_yrs,
        COUNT(DISTINCT isco1) AS n_occ
    FROM stg_ilostat_hours_ocu
    GROUP BY iso3
)
SELECT
    h.iso3,
    c.country_name,
    c.income_level,
    c.region,
    h.isco1,
    e.occupation_group,
    h.sex,
    h.year,
    h.source,
    h.weekly_hours,
    LN(h.weekly_hours) AS ln_hours,
    e.exposure_2025,
    e.exposure_2023,
    e.share_exposed_grad34,
    CASE WHEN h.year >= 2023 THEN 1 ELSE 0 END AS post_2022,
    h.year - 2022 AS event_time,
    h.iso3 || '_' || h.isco1 AS unit_fe,
    h.iso3 || '_' || CAST(h.year AS VARCHAR) AS country_year_fe
FROM stg_ilostat_hours_ocu h
INNER JOIN mart_lfs_country_eligibility el
    ON h.iso3 = el.iso3 AND el.is_eligible = 1
INNER JOIN hrs_elig he
    ON h.iso3 = he.iso3 AND he.pre_yrs >= 3 AND he.post_yrs >= 1 AND he.n_occ >= 7
INNER JOIN stg_countries_lmics c
    ON h.iso3 = c.iso3
INNER JOIN stg_ilo_exposure_isco1 e
    ON h.isco1 = e.isco1;

-- Panel ngắn hạn Tiền lương danh nghĩa trung bình tháng (country_year_fe khử 100% lạm phát CPI & tỷ giá LCU)
CREATE OR REPLACE TABLE mart_panel_earnings_ocu AS
WITH ear_elig AS (
    SELECT
        iso3,
        COUNT(DISTINCT CASE WHEN year BETWEEN 2015 AND 2022 THEN year END) AS pre_yrs,
        COUNT(DISTINCT CASE WHEN year BETWEEN 2023 AND 2025 THEN year END) AS post_yrs,
        COUNT(DISTINCT isco1) AS n_occ
    FROM stg_ilostat_earnings_ocu
    GROUP BY iso3
)
SELECT
    w.iso3,
    c.country_name,
    c.income_level,
    c.region,
    w.isco1,
    e.occupation_group,
    w.sex,
    w.year,
    w.source,
    w.monthly_earnings_lcu,
    LN(w.monthly_earnings_lcu) AS ln_earnings,
    e.exposure_2025,
    e.exposure_2023,
    e.share_exposed_grad34,
    CASE WHEN w.year >= 2023 THEN 1 ELSE 0 END AS post_2022,
    w.year - 2022 AS event_time,
    w.iso3 || '_' || w.isco1 AS unit_fe,
    w.iso3 || '_' || CAST(w.year AS VARCHAR) AS country_year_fe
FROM stg_ilostat_earnings_ocu w
INNER JOIN mart_lfs_country_eligibility el
    ON w.iso3 = el.iso3 AND el.is_eligible = 1
INNER JOIN ear_elig ee
    ON w.iso3 = ee.iso3 AND ee.pre_yrs >= 3 AND ee.post_yrs >= 1 AND ee.n_occ >= 7
INNER JOIN stg_countries_lmics c
    ON w.iso3 = c.iso3
INNER JOIN stg_ilo_exposure_isco1 e
    ON w.isco1 = e.isco1;
