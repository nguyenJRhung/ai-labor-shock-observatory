-- =============================================================================
-- 03_mart_vietnam_ba.sql
-- Lớp Mart Nghiệp vụ BA: Phân tích chuyên sâu Việt Nam, Khoảng cách Giới tính
-- (Gender Exposure Gap) và Đối chiếu Lý thuyết ILO vs Thực tế GitHub/AEI.
-- =============================================================================

CREATE OR REPLACE TABLE mart_vietnam_workforce_isco1 AS
WITH vnm_base AS (
    SELECT
        l.iso3,
        l.isco1,
        e.occupation_group,
        l.sex,
        l.year,
        l.emp_thousands,
        l.emp_thousands / NULLIF(SUM(l.emp_thousands) OVER (PARTITION BY l.sex, l.year), 0) AS emp_share,
        e.exposure_2025,
        e.share_exposed_grad34,
        e.share_automation_23,
        e.share_augmentation_23,
        CASE
            WHEN l.isco1 IN ('1', '2', '3', '4') THEN 'High Exposure (ISCO 1-4)'
            WHEN l.isco1 = '5' THEN 'Medium Exposure (ISCO 5)'
            ELSE 'Low Exposure (ISCO 6-9)'
        END AS exposure_tier
    FROM stg_ilostat_lfs l
    INNER JOIN stg_ilo_exposure_isco1 e ON l.isco1 = e.isco1
    WHERE l.iso3 = 'VNM'
)
SELECT
    *,
    (emp_thousands / NULLIF(LAG(emp_thousands) OVER (PARTITION BY isco1, sex ORDER BY year), 0) - 1.0) * 100.0 AS emp_yoy_pct
FROM vnm_base
ORDER BY sex, year, isco1;

-- Khoảng cách phơi nhiễm GenAI gia quyền theo Giới tính trên toàn khối LMICs
CREATE OR REPLACE TABLE mart_gender_exposure_gap AS
WITH both_sex_years AS (
    SELECT
        iso3,
        year,
        COUNT(DISTINCT sex) AS sex_cnt,
        COUNT(DISTINCT isco1) AS occ_cnt
    FROM stg_ilostat_lfs
    WHERE sex IN ('SEX_M', 'SEX_F') AND year >= 2018
    GROUP BY iso3, year
    HAVING COUNT(DISTINCT sex) = 2 AND COUNT(DISTINCT isco1) >= 7
),
latest_year AS (
    SELECT
        iso3,
        MAX(year) AS ref_year
    FROM both_sex_years
    GROUP BY iso3
),
weighted AS (
    SELECT
        l.iso3,
        ly.ref_year,
        l.sex,
        SUM(l.emp_thousands * e.exposure_2025) / NULLIF(SUM(l.emp_thousands), 0) AS weighted_exposure_2025,
        SUM(l.emp_thousands * e.share_exposed_grad34) / NULLIF(SUM(l.emp_thousands), 0) AS weighted_grad34_share,
        SUM(CASE WHEN l.isco1 = '4' THEN l.emp_thousands ELSE 0 END) / NULLIF(SUM(l.emp_thousands), 0) AS clerical_share,
        SUM(l.emp_thousands) AS total_emp_thousands
    FROM stg_ilostat_lfs l
    INNER JOIN latest_year ly ON l.iso3 = ly.iso3 AND l.year = ly.ref_year
    INNER JOIN stg_ilo_exposure_isco1 e ON l.isco1 = e.isco1
    WHERE l.sex IN ('SEX_M', 'SEX_F')
    GROUP BY l.iso3, ly.ref_year, l.sex
)
SELECT
    c.iso3,
    c.country_name,
    c.income_level,
    c.region,
    f.ref_year,
    f.weighted_exposure_2025 AS female_exposure_2025,
    m.weighted_exposure_2025 AS male_exposure_2025,
    f.weighted_exposure_2025 - m.weighted_exposure_2025 AS gender_exposure_gap,
    f.weighted_grad34_share AS female_grad34_share,
    m.weighted_grad34_share AS male_grad34_share,
    f.clerical_share AS female_clerical_share,
    m.clerical_share AS male_clerical_share,
    f.total_emp_thousands AS female_emp_thousands,
    m.total_emp_thousands AS male_emp_thousands
FROM weighted f
INNER JOIN weighted m ON f.iso3 = m.iso3 AND f.sex = 'SEX_F' AND m.sex = 'SEX_M'
INNER JOIN stg_countries_lmics c ON f.iso3 = c.iso3
ORDER BY gender_exposure_gap DESC;

-- Đối chiếu cấp quốc gia LMICs: Phơi nhiễm Lý thuyết (ILO) vs Ứng dụng Thực tế (GitHub & AEI)
CREATE OR REPLACE TABLE mart_theory_vs_adoption AS
WITH latest_total_year AS (
    SELECT
        iso3,
        MAX(year) AS emp_ref_year
    FROM stg_ilostat_lfs
    WHERE sex = 'SEX_T'
    GROUP BY iso3
    HAVING COUNT(DISTINCT isco1) >= 7
),
country_theo AS (
    SELECT
        l.iso3,
        lt.emp_ref_year,
        SUM(l.emp_thousands * e.exposure_2025) / NULLIF(SUM(l.emp_thousands), 0) AS theoretical_exposure_2025,
        SUM(l.emp_thousands * e.share_exposed_grad34) / NULLIF(SUM(l.emp_thousands), 0) AS theoretical_grad34_share,
        SUM(CASE WHEN l.isco1 IN ('1','2','3','4') THEN l.emp_thousands ELSE 0 END) / NULLIF(SUM(l.emp_thousands), 0) AS white_collar_share,
        SUM(l.emp_thousands) AS total_emp_thousands
    FROM stg_ilostat_lfs l
    INNER JOIN latest_total_year lt ON l.iso3 = lt.iso3 AND l.year = lt.emp_ref_year
    INNER JOIN stg_ilo_exposure_isco1 e ON l.isco1 = e.isco1
    WHERE l.sex = 'SEX_T'
    GROUP BY l.iso3, lt.emp_ref_year
)
SELECT
    ct.iso3,
    t.country_name,
    t.income_level,
    ct.emp_ref_year,
    ct.theoretical_exposure_2025,
    ct.theoretical_grad34_share,
    ct.white_collar_share,
    ct.total_emp_thousands,
    t.gh_developers,
    t.gh_git_pushes,
    t.gh_dev_growth_since_2022,
    t.devs_per_10k_wap,
    t.high_tech_readiness,
    t.aei_usage_count,
    t.aei_usage_pct,
    t.aei_automation_pct,
    t.aei_augmentation_pct,
    t.aui_proxy,
    COALESCE(el.is_eligible, 0) AS in_causal_panel
FROM country_theo ct
INNER JOIN stg_tech_readiness t ON ct.iso3 = t.iso3
LEFT JOIN mart_lfs_country_eligibility el ON ct.iso3 = el.iso3
ORDER BY ct.theoretical_exposure_2025 DESC;
