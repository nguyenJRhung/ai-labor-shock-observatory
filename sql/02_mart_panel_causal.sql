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
    t.aui_proxy
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
