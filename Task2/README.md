# Task 2 — AISHE Enrolment Statistical Analysis

AI-assisted statistical analysis of **State/UT-wise estimated enrolment** at Ph.D., M.Phil., and Post Graduate levels in India (AISHE Report 2023–24, Table 6), with focus on **gender-wise participation** and **regional variation**.

## Project checklist

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Filter States/UTs (Ph.D. ≥ 100, PG ≥ 1,000) | `src/analysis.py` → `load_and_filter()` |
| 2 | Organize by state, level, gender, total | `organized_enrolment_long.csv` |
| 3 | Graphical presentations | `outputs/figures/` |
| 4 | Mean, median, SD, CV | `descriptive_statistics.csv` |
| 5 | Correlation Ph.D. ↔ PG | `correlation_phd_pg.json` |
| 6 | Gender gap analysis | `gender_gap_*.csv` |
| 7 | Regression Ph.D. ~ PG | `regression_*.json/txt` |
| 8 | ANOVA across zones | `anova_by_zone.json` |
| 9 | Report & interpretation | `outputs/reports/aishe_analysis_report.md` |

## Setup

```bash
cd Task2
pip install -r requirements.txt
```

## Run analysis

```bash
python src/analysis.py
```

Outputs are written under `outputs/figures/`, `outputs/tables/`, and `outputs/reports/`.

## Data

- **File:** `data/aishe_table6_enrolment.csv`
- **Format:** AISHE Table 6 style columns — State/UT, Zone, Ph.D./M.Phil./PG by Male, Female, Total.
- Replace the CSV with official AISHE export figures if you have the published table; column names must match.

## Zones (for ANOVA)

States/UTs are tagged **North / South / East / West / Central / Northeast** for regional comparison.
