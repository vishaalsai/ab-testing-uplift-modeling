# A/B Testing & Uplift Modeling — E-Mail Marketing Campaign

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ab-testing-uplift-modeling.streamlit.app)

---

## Project Overview

This project delivers a full end-to-end A/B testing and uplift modeling pipeline applied to a real-world e-mail marketing dataset. The analysis begins with exploratory data analysis and rigorous statistical assumption checks (including Sample Ratio Mismatch detection), proceeds through hypothesis testing with confidence intervals and power analysis, and culminates in a T-Learner uplift model that estimates individual-level treatment effects for every customer. The entire pipeline is surfaced through an interactive Streamlit dashboard with five pages: experiment overview, statistical results, uplift explorer, cohort analysis, and a final business decision recommendation.

The project goes beyond asking "did the campaign work overall?" and instead answers the more valuable question: "which customers are actually persuaded by the e-mail, and how do we target only them?" This distinction — between average treatment effects and heterogeneous treatment effects — is the foundation of modern experimentation practice at companies like Netflix, Airbnb, and Uber, and is directly applicable to any conversion-rate optimization or marketing spend problem.

---

## Key Findings

| Metric | Result |
|--------|--------|
| Control conversion rate | 0.573% |
| Treatment conversion rate | 1.068% |
| **Absolute lift** | **+0.495 pp** |
| **Relative lift** | **+86.5%** |
| P-value | < 0.000001 |
| Statistical significance | ✅ Yes (z = 6.24) |
| Visit rate lift (guardrail) | +6.09 pp — healthy ✅ |
| Study power achieved | **100%** (required n = 5,083; actual n = 21,306) |
| Top uplift segment | **$750–$1,000 spenders** — 3× the average lift (+1.458 pp) |
| **Final recommendation** | **TARGETED ROLLOUT** |

> The e-mail campaign produces a statistically significant and practically meaningful lift. However, T-Learner uplift modeling reveals strong treatment-effect heterogeneity — a significant fraction of customers shows near-zero or negative predicted uplift. Sending only to the top uplift quartile improves cost-efficiency while capturing the majority of incremental conversions.

---

## Project Structure

```
ab-testing-uplift-modeling/
├── data/
│   └── hillstrom.csv          # Raw dataset (64,000 customers)
├── notebooks/
│   └── analysis.ipynb         # Full Phase 1 + Phase 2 analysis
├── app/
│   └── streamlit_app.py       # Interactive dashboard (5 pages)
├── src/
│   └── uplift_utils.py        # Utility module (Phase 3+)
├── requirements.txt
└── README.md
```

**Notebook phases:**
- **Phase 1** — Data loading, EDA, metric definitions, SRM & assumption checks
- **Phase 2** — Hypothesis testing, confidence intervals, power analysis, T-Learner uplift modeling, cohort segmentation, decision recommendation

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/vishaalsai/ab-testing-uplift-modeling.git
cd ab-testing-uplift-modeling
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

### 4. Run the analysis notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

The dataset (`data/hillstrom.csv`) is included in the repository — no separate download needed.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data manipulation | `pandas`, `numpy` |
| Statistical testing | `scipy`, `statsmodels`, `pingouin` |
| Machine learning | `scikit-learn` (Random Forest T-Learner) |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Notebook | `jupyter` |

---

## Dataset

The dataset is the **Hillstrom MineThatData E-Mail Analytics and Data Mining Challenge (2008)**, published by Kevin Hillstrom. It contains 64,000 customers randomly assigned to one of three groups:

- **No E-Mail** — control group (n ≈ 21,306)
- **Mens E-Mail** — treatment: men's merchandise campaign (n ≈ 21,307)
- **Womens E-Mail** — treatment: women's merchandise campaign (n ≈ 21,387)

Outcome variables recorded over the following two weeks include website visits, purchases (conversion), and revenue. The dataset is a benchmark for causal inference and uplift modeling research.

> Source: [MineThatData Blog — Kevin Hillstrom](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv)

---

## References

- **Netflix Technology Blog** — [Experimentation Platform at Netflix](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15b)
- **Uber Engineering** — [CUPED: Improving Experiment Sensitivity](https://www.uber.com/blog/causal-inference-at-uber/)
- **Booking.com** — [150 Successful Machine Learning Models: 6 Lessons Learned](https://dl.acm.org/doi/10.1145/3292500.3330744)
- **Athey & Imbens (2016)** — [Recursive Partitioning for Heterogeneous Causal Effects](https://www.pnas.org/doi/10.1073/pnas.1510489113)
- **Künzel et al. (2019)** — [Metalearners for Estimating Heterogeneous Treatment Effects](https://www.pnas.org/doi/10.1073/pnas.1804597116) *(T-Learner / X-Learner framework used in this project)*

---

## License

MIT — free to use, adapt, and build on.
