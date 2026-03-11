# A/B Testing & Uplift Modeling — E-Mail Marketing Campaign

## Description

This project demonstrates a full end-to-end A/B testing and uplift modeling pipeline applied to a real-world e-mail marketing dataset. Starting from raw campaign data, the analysis covers exploratory data analysis, metric definition, statistical assumption checks (including Sample Ratio Mismatch detection), hypothesis testing, and uplift modeling to identify which customer segments are most persuadable by a marketing treatment. The goal is to go beyond simple A/B test significance and quantify *individual-level* treatment effects, enabling smarter targeting decisions.

## Tech Stack

- **Python** — core language
- **pandas / numpy** — data manipulation
- **scipy / statsmodels / pingouin** — statistical testing
- **scikit-learn** — machine learning utilities
- **causalml** — uplift modeling (S-Learner, T-Learner, X-Learner, etc.)
- **matplotlib / seaborn / plotly** — visualization
- **Streamlit** — interactive results dashboard
- **Jupyter Notebook** — analysis and reporting

## Dataset

The dataset is from **Kevin Hillstrom's MineThatData E-Mail Analytics and Data Mining Challenge (2008)**. It contains 64,000 customers who were randomly assigned to one of three groups: no e-mail, men's merchandise e-mail, or women's merchandise e-mail. Outcome variables include website visits, conversion, and revenue over the following two weeks.

Source: [MineThatData Blog](http://www.minethatdata.com)

## Project Structure

```
ab-testing-uplift-modeling/
├── data/               # Raw data (not tracked in git)
├── notebooks/          # Jupyter analysis notebooks
├── app/                # Streamlit dashboard
├── src/                # Utility modules
├── requirements.txt
└── README.md
```

## Phases

- **Phase 1** — Data loading, EDA, metric definitions, statistical assumption checks
- **Phase 2** — Hypothesis testing, confidence intervals, power analysis *(coming soon)*
- **Phase 3** — Uplift modeling and CATE estimation *(coming soon)*
- **Phase 4** — Streamlit dashboard *(coming soon)*
