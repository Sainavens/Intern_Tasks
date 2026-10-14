# Presentation Outline — AISHE Enrolment Analysis

**Topic:** State/UT-wise Ph.D., M.Phil., and PG enrolment in India (AISHE 2023–24), with gender and regional focus.

## Slide 1 — Title
- AI-assisted statistical analysis of enrolment at Ph.D., M.Phil., and PG levels
- Special reference: gender-wise participation & regional variation
- Data: AISHE Report 2023–24, Table 6

## Slide 2 — Objectives
1. Filter meaningful Ph.D./PG cohorts
2. Describe enrolment distribution
3. Measure Ph.D.–PG association
4. Quantify gender gaps
5. Predict Ph.D. from PG (regression)
6. Test zone differences (ANOVA)

## Slide 3 — Data & filters
- 36 States/UTs in source → 32 retained
- Criteria: Ph.D. ≥ 100 and PG ≥ 1,000
- Organized by State × Level × Gender

## Slide 4 — Descriptive picture
- Show mean / median / SD / CV table
- Highlight skew toward large States

## Slide 5 — Visual: Top States
- Figures `01_phd_enrolment_top15.png`, `02_pg_enrolment_top15.png`

## Slide 6 — Correlation & regression
- Pearson *r* ≈ 0.97; R² ≈ 0.93
- Figure `03_phd_vs_pg_scatter.png`

## Slide 7 — Gender gap
- Female share often higher at PG than Ph.D.
- Figure `04_gender_gap_female_share.png`

## Slide 8 — Regional ANOVA
- Significant zone differences for Ph.D. and PG (α = 0.05)
- Figure `05_anova_zone_boxplots.png`

## Slide 9 — Conclusions & policy notes
- Scale of PG system strongly predicts Ph.D. capacity
- Gender pipeline: strengthen doctoral pathways for women
- Zone-specific planning where means diverge

## Slide 10 — Q&A / appendix
- Full report: `aishe_analysis_report.md`
- Reproduce: `python src/analysis.py`
