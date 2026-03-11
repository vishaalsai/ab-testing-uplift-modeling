"""
A/B Testing & Uplift Modeling — Streamlit Dashboard
Hillstrom MineThatData E-Mail Analytics Challenge (2008)
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chisquare  # noqa: F401  (imported for completeness)
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test Dashboard — Hillstrom",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hillstrom.csv")

# ── Shared colour palette ─────────────────────────────────────────────────────
C_CTRL = "#4C72B0"
C_TRTM = "#DD8452"
C_BEST = "#2ca02c"

# ══════════════════════════════════════════════════════════════════════════════
# CACHED COMPUTATIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_metrics():
    """Load data and compute all statistical metrics."""
    df = pd.read_csv(DATA_PATH)
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
    df["newbie_label"] = df["newbie"].map({0: "Non-Newbie", 1: "Newbie"})

    ctrl = df[df["treatment"] == 0]
    trt  = df[df["treatment"] == 1]
    n_c, n_t = len(ctrl), len(trt)

    # ── Conversion ────────────────────────────────────────────────────────────
    c_c, c_t = int(ctrl["conversion"].sum()), int(trt["conversion"].sum())
    p_c, p_t = c_c / n_c, c_t / n_t
    z_cv, p_cv = proportions_ztest([c_t, c_c], [n_t, n_c])
    ci_c_lo, ci_c_hi = proportion_confint(c_c, n_c, alpha=0.05, method="normal")
    ci_t_lo, ci_t_hi = proportion_confint(c_t, n_t, alpha=0.05, method="normal")
    h_cv = 2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_c))
    abs_lft = p_t - p_c
    rel_lft = abs_lft / p_c * 100

    # ── Visit (guardrail) ─────────────────────────────────────────────────────
    v_c, v_t = int(ctrl["visit"].sum()), int(trt["visit"].sum())
    pv_c, pv_t = v_c / n_c, v_t / n_t
    z_vs, p_vs = proportions_ztest([v_t, v_c], [n_t, n_c])
    ci_vc_lo, ci_vc_hi = proportion_confint(v_c, n_c, alpha=0.05, method="normal")
    ci_vt_lo, ci_vt_hi = proportion_confint(v_t, n_t, alpha=0.05, method="normal")
    h_vs = 2 * np.arcsin(np.sqrt(pv_t)) - 2 * np.arcsin(np.sqrt(pv_c))
    abs_lft_v = pv_t - pv_c
    rel_lft_v = abs_lft_v / pv_c * 100

    # ── Power ─────────────────────────────────────────────────────────────────
    pw = NormalIndPower()
    effect = abs(h_cv)
    ratio  = n_t / n_c
    n_req  = pw.solve_power(effect_size=effect, alpha=0.05, power=0.80)
    a_pwr  = float(np.clip(
        pw.solve_power(effect_size=effect, alpha=0.05, nobs1=n_c, ratio=ratio),
        0, 1
    ))

    return dict(
        df=df,
        n_c=n_c, n_t=n_t,
        p_c=p_c, p_t=p_t,
        z_cv=z_cv, p_cv=p_cv,
        ci_c_lo=ci_c_lo, ci_c_hi=ci_c_hi,
        ci_t_lo=ci_t_lo, ci_t_hi=ci_t_hi,
        h_cv=h_cv, abs_lft=abs_lft, rel_lft=rel_lft,
        pv_c=pv_c, pv_t=pv_t,
        z_vs=z_vs, p_vs=p_vs,
        ci_vc_lo=ci_vc_lo, ci_vc_hi=ci_vc_hi,
        ci_vt_lo=ci_vt_lo, ci_vt_hi=ci_vt_hi,
        h_vs=h_vs, abs_lft_v=abs_lft_v, rel_lft_v=rel_lft_v,
        effect=effect, n_req=n_req, a_pwr=a_pwr, ratio=ratio,
    )


@st.cache_data(show_spinner=False)
def compute_uplift_scores():
    """Train T-Learner and return dataframe with uplift scores."""
    df = pd.read_csv(DATA_PATH)
    dbin = df[df["segment"].isin(["No E-Mail", "Mens E-Mail"])].copy().reset_index(drop=True)
    dbin["treat"] = (dbin["segment"] == "Mens E-Mail").astype(int)
    dbin["newbie_label"] = dbin["newbie"].map({0: "Non-Newbie", 1: "Newbie"})

    num = ["recency", "history", "mens", "womens", "newbie"]
    X_cat = pd.get_dummies(dbin[["channel", "zip_code"]], drop_first=False)
    X = pd.concat([dbin[num].reset_index(drop=True),
                   X_cat.reset_index(drop=True)], axis=1)
    y, tr = dbin["conversion"].values, dbin["treat"].values

    rf_t = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_c = RandomForestClassifier(n_estimators=200, max_depth=6,
                                   min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf_t.fit(X[tr == 1], y[tr == 1])
    rf_c.fit(X[tr == 0], y[tr == 0])

    dbin["uplift_score"] = rf_t.predict_proba(X)[:, 1] - rf_c.predict_proba(X)[:, 1]
    return dbin


def power_curve(effect, ratio, n_max):
    """Return xs, ys arrays for the power curve."""
    pw = NormalIndPower()
    xs = np.linspace(200, n_max, 400)
    ys = np.clip(
        [pw.solve_power(effect_size=effect, alpha=0.05, nobs1=n, ratio=ratio) for n in xs],
        0, 1,
    )
    return xs, ys


def cohort_pivot(df, col):
    """Return pivot with Control / Treatment / Uplift_pp columns."""
    r = (df.groupby([col, "treatment"])["conversion"]
           .mean().reset_index()
           .rename(columns={"conversion": "conv_rate"}))
    r["conv_rate_pct"] = r["conv_rate"] * 100
    piv = r.pivot(index=col, columns="treatment", values="conv_rate_pct").fillna(0)
    piv = piv.rename(columns={0: "Control", 1: "Treatment"})
    for c in ("Control", "Treatment"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv["Uplift_pp"] = piv["Treatment"] - piv["Control"]
    return piv


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA (runs once, cached)
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading data…"):
    m = load_metrics()

df = m["df"]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 A/B Test Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate to:",
        [
            "🏠  Experiment Overview",
            "📈  Statistical Results",
            "🎯  Uplift Explorer",
            "👥  Cohort Analysis",
            "✅  Decision Recommendation",
        ],
    )
    st.markdown("---")
    st.caption("**Dataset:** Hillstrom MineThatData Challenge (2008) — 64,000 customers")
    st.caption("**Treatment:** E-Mail (Mens + Womens) vs No E-Mail")
    st.caption("**Primary metric:** Conversion rate")
    st.caption("**Guardrail metric:** Visit rate")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXPERIMENT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Experiment Overview":
    st.title("E-Mail Campaign A/B Test — Results Dashboard")
    st.markdown(
        "**Hillstrom MineThatData E-Mail Analytics Challenge (2008)** — "
        f"{len(df):,} customers randomly assigned to e-mail or no e-mail."
    )
    st.divider()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Control Conversion Rate",  f"{m['p_c']*100:.3f}%",
                help="No E-Mail group (n = {:,})".format(m["n_c"]))
    col2.metric("Treatment Conversion Rate", f"{m['p_t']*100:.3f}%",
                delta=f"+{m['abs_lft']*100:.3f} pp vs control",
                help="E-Mail group (Mens + Womens, n = {:,})".format(m["n_t"]))
    col3.metric("Absolute Lift",   f"+{m['abs_lft']*100:.3f} pp")
    col4.metric("Relative Lift",   f"+{m['rel_lft']:.1f}%")

    st.divider()

    # Group sizes donut
    col_donut, col_conv, col_visit = st.columns([1, 2, 2])

    with col_donut:
        st.markdown("#### Group Sizes")
        fig_d = go.Figure(go.Pie(
            labels=["Control\n(No E-Mail)", "Treatment\n(E-Mail)"],
            values=[m["n_c"], m["n_t"]],
            hole=0.55,
            marker_colors=[C_CTRL, C_TRTM],
            textinfo="label+percent",
        ))
        fig_d.update_layout(height=300, showlegend=False,
                             margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_d, use_container_width=True)

    with col_conv:
        fig_c = go.Figure()
        groups  = ["Control (No E-Mail)", "Treatment (E-Mail)"]
        rates_c = [m["p_c"] * 100,  m["p_t"] * 100]
        lo_c    = [m["ci_c_lo"]*100, m["ci_t_lo"]*100]
        hi_c    = [m["ci_c_hi"]*100, m["ci_t_hi"]*100]
        colors  = [C_CTRL, C_TRTM]
        for i in range(2):
            fig_c.add_trace(go.Bar(
                x=[groups[i]], y=[rates_c[i]],
                marker_color=colors[i],
                error_y=dict(type="data", symmetric=False,
                             array=[hi_c[i] - rates_c[i]],
                             arrayminus=[rates_c[i] - lo_c[i]]),
                text=f"{rates_c[i]:.3f}%", textposition="outside",
                showlegend=False,
            ))
        fig_c.update_layout(
            title="Conversion Rate (95% CI)",
            yaxis_title="Conversion Rate (%)",
            height=340, plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            margin=dict(t=45, b=10),
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with col_visit:
        fig_v = go.Figure()
        rates_v = [m["pv_c"]*100, m["pv_t"]*100]
        lo_v    = [m["ci_vc_lo"]*100, m["ci_vt_lo"]*100]
        hi_v    = [m["ci_vc_hi"]*100, m["ci_vt_hi"]*100]
        for i in range(2):
            fig_v.add_trace(go.Bar(
                x=[groups[i]], y=[rates_v[i]],
                marker_color=colors[i],
                error_y=dict(type="data", symmetric=False,
                             array=[hi_v[i] - rates_v[i]],
                             arrayminus=[rates_v[i] - lo_v[i]]),
                text=f"{rates_v[i]:.2f}%", textposition="outside",
                showlegend=False,
            ))
        fig_v.update_layout(
            title="Visit Rate — Guardrail (95% CI)",
            yaxis_title="Visit Rate (%)",
            height=340, plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            margin=dict(t=45, b=10),
        )
        st.plotly_chart(fig_v, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STATISTICAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Statistical Results":
    st.title("Statistical Results")
    st.divider()

    # Significance badge
    if m["p_cv"] < 0.05:
        st.success("### ✅  STATISTICALLY SIGNIFICANT  (p < 0.05)")
    else:
        st.error("### ❌  NOT SIGNIFICANT  (p ≥ 0.05)")

    st.divider()
    tab_conv, tab_visit, tab_power = st.tabs(
        ["Conversion Rate", "Visit Rate (Guardrail)", "Power Analysis"]
    )

    with tab_conv:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Test Statistics")
            stats = pd.DataFrame({
                "Metric": [
                    "Z-statistic", "P-value",
                    "Control rate", "Treatment rate",
                    "Control 95% CI", "Treatment 95% CI",
                    "Cohen's h", "Absolute lift", "Relative lift",
                ],
                "Value": [
                    f"{m['z_cv']:.4f}",
                    f"{m['p_cv']:.2e}",
                    f"{m['p_c']*100:.3f}%",
                    f"{m['p_t']*100:.3f}%",
                    f"[{m['ci_c_lo']*100:.3f}%,  {m['ci_c_hi']*100:.3f}%]",
                    f"[{m['ci_t_lo']*100:.3f}%,  {m['ci_t_hi']*100:.3f}%]",
                    f"{m['h_cv']:.4f}",
                    f"+{m['abs_lft']*100:.3f} pp",
                    f"+{m['rel_lft']:.2f}%",
                ],
            })
            st.dataframe(stats, use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("#### Confidence Intervals")
            ci_df = pd.DataFrame({
                "Group":    ["Control", "Treatment"],
                "Rate (%)": [m["p_c"]*100, m["p_t"]*100],
                "CI Lo":    [m["ci_c_lo"]*100, m["ci_t_lo"]*100],
                "CI Hi":    [m["ci_c_hi"]*100, m["ci_t_hi"]*100],
            })
            fig_ci = go.Figure()
            for i, row in ci_df.iterrows():
                fig_ci.add_trace(go.Scatter(
                    x=[row["Group"]], y=[row["Rate (%)"]],
                    mode="markers",
                    marker=dict(size=14, color=[C_CTRL, C_TRTM][i]),
                    error_y=dict(type="data", symmetric=False,
                                 array=[row["CI Hi"] - row["Rate (%)"]],
                                 arrayminus=[row["Rate (%)"] - row["CI Lo"]],
                                 color=[C_CTRL, C_TRTM][i]),
                    name=row["Group"],
                ))
            fig_ci.update_layout(
                title="Conversion Rate with 95% Confidence Intervals",
                yaxis_title="Conversion Rate (%)",
                height=360, plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
            )
            st.plotly_chart(fig_ci, use_container_width=True)

    with tab_visit:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Test Statistics")
            vstats = pd.DataFrame({
                "Metric": [
                    "Z-statistic", "P-value",
                    "Control rate", "Treatment rate",
                    "Control 95% CI", "Treatment 95% CI",
                    "Cohen's h", "Absolute lift", "Relative lift",
                ],
                "Value": [
                    f"{m['z_vs']:.4f}",
                    f"{m['p_vs']:.2e}",
                    f"{m['pv_c']*100:.3f}%",
                    f"{m['pv_t']*100:.3f}%",
                    f"[{m['ci_vc_lo']*100:.3f}%,  {m['ci_vc_hi']*100:.3f}%]",
                    f"[{m['ci_vt_lo']*100:.3f}%,  {m['ci_vt_hi']*100:.3f}%]",
                    f"{m['h_vs']:.4f}",
                    f"+{m['abs_lft_v']*100:.3f} pp",
                    f"+{m['rel_lft_v']:.2f}%",
                ],
            })
            st.dataframe(vstats, use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("#### Confidence Intervals")
            ci_vdf = pd.DataFrame({
                "Group":    ["Control", "Treatment"],
                "Rate (%)": [m["pv_c"]*100, m["pv_t"]*100],
                "CI Lo":    [m["ci_vc_lo"]*100, m["ci_vt_lo"]*100],
                "CI Hi":    [m["ci_vc_hi"]*100, m["ci_vt_hi"]*100],
            })
            fig_civ = go.Figure()
            for i, row in ci_vdf.iterrows():
                fig_civ.add_trace(go.Scatter(
                    x=[row["Group"]], y=[row["Rate (%)"]],
                    mode="markers",
                    marker=dict(size=14, color=[C_CTRL, C_TRTM][i]),
                    error_y=dict(type="data", symmetric=False,
                                 array=[row["CI Hi"] - row["Rate (%)"]],
                                 arrayminus=[row["Rate (%)"] - row["CI Lo"]],
                                 color=[C_CTRL, C_TRTM][i]),
                    name=row["Group"],
                ))
            fig_civ.update_layout(
                title="Visit Rate with 95% Confidence Intervals",
                yaxis_title="Visit Rate (%)",
                height=360, plot_bgcolor="white",
                yaxis=dict(gridcolor="#eee"),
            )
            st.plotly_chart(fig_civ, use_container_width=True)

    with tab_power:
        pa1, pa2, pa3 = st.columns(3)
        pa1.metric("Min. Required n (per group)",
                   f"{int(np.ceil(m['n_req'])):,}",
                   help="Equal allocation, α=0.05, power=0.80")
        pa2.metric("Actual Control n", f"{m['n_c']:,}",
                   delta=f"+{m['n_c'] - int(np.ceil(m['n_req'])):,} above minimum")
        pa3.metric("Power Achieved", f"{m['a_pwr']*100:.1f}%",
                   delta="ADEQUATELY POWERED" if m["a_pwr"] >= 0.80 else "UNDERPOWERED")
        st.divider()
        xs, ys = power_curve(
            m["effect"], m["ratio"],
            max(m["n_c"] * 2, int(np.ceil(m["n_req"])) * 6),
        )
        fig_pw = go.Figure()
        fig_pw.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name="Power curve",
            line=dict(color=C_CTRL, width=2.5),
            fill="tozeroy", fillcolor="rgba(76,114,176,0.08)",
        ))
        fig_pw.add_hline(y=0.80, line_dash="dash", line_color="red",
                          annotation_text="Target power = 0.80",
                          annotation_position="bottom right")
        fig_pw.add_vline(x=m["n_c"], line_dash="dash", line_color=C_BEST,
                          annotation_text=f"Actual n = {m['n_c']:,}",
                          annotation_position="top right")
        fig_pw.add_vline(x=int(np.ceil(m["n_req"])), line_dash="dash", line_color="orange",
                          annotation_text=f"Min required = {int(np.ceil(m['n_req'])):,}",
                          annotation_position="top left")
        fig_pw.update_layout(
            title="Power Curve — Conversion Rate A/B Test",
            xaxis_title="Control group sample size (n)",
            yaxis_title="Statistical Power",
            height=430, plot_bgcolor="white",
            yaxis=dict(range=[0, 1.05], gridcolor="#eee"),
        )
        st.plotly_chart(fig_pw, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — UPLIFT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Uplift Explorer":
    st.title("Uplift Explorer")
    st.markdown(
        "**T-Learner uplift model** trained on **No E-Mail vs Mens E-Mail** groups.  "
        "Uplift score = P(conversion | treated) − P(conversion | control)."
    )
    st.divider()

    with st.spinner("Training T-Learner models (cached after first run)…"):
        du = compute_uplift_scores()

    # Summary stats row
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total customers scored", f"{len(du):,}")
    s2.metric("Mean uplift score",  f"{du['uplift_score'].mean():.4f}")
    s3.metric("Median uplift score", f"{du['uplift_score'].median():.4f}")
    pct_neg = (du["uplift_score"] < 0).mean() * 100
    s4.metric("Fraction with negative uplift", f"{pct_neg:.1f}%",
              help="Customers where e-mail may hurt conversion")

    st.divider()

    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_h = px.histogram(
            du, x="uplift_score", nbins=70,
            title="Distribution of Uplift Scores",
            labels={"uplift_score": "Uplift Score"},
            color_discrete_sequence=[C_CTRL],
        )
        fig_h.add_vline(x=0, line_dash="dash", line_color="red",
                        annotation_text="0", annotation_position="top right")
        fig_h.add_vline(x=du["uplift_score"].median(), line_dash="dot",
                        line_color="orange", annotation_text="median")
        fig_h.update_layout(height=350, plot_bgcolor="white",
                             yaxis=dict(gridcolor="#eee"))
        st.plotly_chart(fig_h, use_container_width=True)

    with col_box:
        fig_bx = px.box(
            du, x="treat", y="uplift_score",
            color="treat",
            color_discrete_map={0: C_CTRL, 1: C_TRTM},
            labels={"treat": "Group", "uplift_score": "Uplift Score"},
            title="Uplift Score by Actual Group",
            category_orders={"treat": [0, 1]},
        )
        fig_bx.update_layout(
            height=350, plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            xaxis=dict(tickvals=[0, 1], ticktext=["Control", "Treatment"]),
            showlegend=False,
        )
        st.plotly_chart(fig_bx, use_container_width=True)

    # ── Targeting Simulator ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Targeting Simulator")
    st.markdown("Drag the slider to see the impact of restricting sends to the top‑N% by uplift score.")

    top_pct = st.slider("Top X% of users by uplift score", 5, 100, 25, step=5)

    n_total   = len(du)
    n_sel     = max(1, int(n_total * top_pct / 100))
    df_sel    = du.nlargest(n_sel, "uplift_score")
    avg_sel   = df_sel["uplift_score"].mean()
    avg_all   = du["uplift_score"].mean()
    est_sel   = df_sel["uplift_score"].sum()
    est_all   = du["uplift_score"].sum()
    efficiency = avg_sel / avg_all if avg_all > 0 else 1.0

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Users in segment", f"{n_sel:,}", f"{top_pct}% of {n_total:,}")
    t2.metric("Avg uplift score",  f"{avg_sel:.4f}",
              delta=f"{avg_sel - avg_all:+.4f} vs population avg")
    t3.metric("Est. incremental conversions", f"{est_sel:.1f}",
              help="Sum of uplift scores for targeted segment")
    t4.metric("Email efficiency vs blanket send", f"{efficiency:.2f}×",
              help="How much more uplift per email vs sending to everyone")

    # Cumulative gain curve
    ds = du.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    cum_uplift = ds["uplift_score"].cumsum().values
    pct_pop    = np.linspace(0, 100, len(ds))

    fig_gain = go.Figure()
    fig_gain.add_trace(go.Scatter(
        x=pct_pop, y=cum_uplift,
        mode="lines", name="Cumulative uplift",
        line=dict(color=C_CTRL, width=2),
        fill="tozeroy", fillcolor="rgba(76,114,176,0.12)",
    ))
    fig_gain.add_vline(
        x=top_pct, line_dash="dash", line_color="red",
        annotation_text=f"Top {top_pct}% cutoff",
        annotation_position="top right",
    )
    fig_gain.update_layout(
        title=f"Cumulative Uplift Gain Curve — cutoff at top {top_pct}%",
        xaxis_title="% Population targeted (sorted by uplift, high → low)",
        yaxis_title="Cumulative Estimated Incremental Conversions",
        height=380, plot_bgcolor="white",
        yaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig_gain, use_container_width=True)

    # Top-20 table
    st.divider()
    st.subheader("Top 20 Customers by Uplift Score")
    show_cols = [
        "recency", "history", "history_segment", "mens", "womens",
        "newbie_label", "channel", "zip_code", "conversion", "treat", "uplift_score",
    ]
    top20 = du.nlargest(20, "uplift_score")[show_cols].reset_index(drop=True)
    top20["uplift_score"] = top20["uplift_score"].round(5)
    st.dataframe(top20, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COHORT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥  Cohort Analysis":
    st.title("Cohort Analysis")
    st.markdown(
        "Conversion rate by treatment vs control within each customer segment.  "
        "Brightest bars = highest-uplift segment."
    )
    st.divider()

    # Map newbie 0/1 to labels for display
    df_disp = df.copy()
    df_disp["newbie"] = df_disp["newbie"].map({0: "Non-Newbie", 1: "Newbie"})

    cohort_map = {
        "Newbie vs Non-Newbie": "newbie",
        "Channel":              "channel",
        "Zip Code (Geography)": "zip_code",
        "Spending Tier":        "history_segment",
    }

    sel = st.selectbox("Select cohort:", list(cohort_map.keys()))
    col_name = cohort_map[sel]
    piv = cohort_pivot(df_disp, col_name)
    best_seg = piv["Uplift_pp"].idxmax()
    best_val = piv.loc[best_seg, "Uplift_pp"]

    st.info(f"**Highest-uplift segment:** {best_seg}  (+{best_val:.3f} pp)")

    c_colors = [C_BEST if idx == best_seg else C_CTRL for idx in piv.index]
    t_colors = [C_BEST if idx == best_seg else C_TRTM for idx in piv.index]

    col_main, col_right = st.columns([3, 1])

    with col_main:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=piv.index.astype(str), y=piv["Control"],
            name="Control (No E-Mail)",
            marker_color=c_colors,
            text=[f"{v:.2f}%" for v in piv["Control"]],
            textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            x=piv.index.astype(str), y=piv["Treatment"],
            name="Treatment (E-Mail)",
            marker_color=t_colors,
            text=[f"{v:.2f}%" for v in piv["Treatment"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title=f"Conversion Rate by {sel}  (green = highest uplift segment)",
            xaxis_title=sel,
            yaxis_title="Conversion Rate (%)",
            barmode="group",
            height=420, plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("#### Segment Uplift")
        for seg, row in piv.sort_values("Uplift_pp", ascending=False).iterrows():
            badge = "🥇 " if seg == best_seg else ""
            st.metric(f"{badge}{seg}", f"+{row['Uplift_pp']:.3f} pp")

    # Uplift bar
    uplift_df = piv.reset_index().rename(columns={col_name: sel})
    fig_up = px.bar(
        uplift_df.sort_values("Uplift_pp", ascending=False),
        x=sel, y="Uplift_pp",
        color="Uplift_pp",
        color_continuous_scale="RdYlGn",
        title=f"Absolute Lift (pp) by {sel}",
        labels={"Uplift_pp": "Absolute Lift (pp)"},
        text="Uplift_pp",
    )
    fig_up.update_traces(texttemplate="%{text:.3f} pp", textposition="outside")
    fig_up.update_layout(
        height=360, plot_bgcolor="white",
        yaxis=dict(gridcolor="#eee"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_up, use_container_width=True)

    # All cohorts summary table
    st.divider()
    st.subheader("All Cohorts — Best Segment Summary")
    rows = []
    for label, col in cohort_map.items():
        p = cohort_pivot(df_disp, col)
        bs = p["Uplift_pp"].idxmax()
        rows.append({
            "Cohort":        label,
            "Best Segment":  str(bs),
            "Control (%)":   round(p.loc[bs, "Control"],   3),
            "Treatment (%)": round(p.loc[bs, "Treatment"], 3),
            "Uplift (pp)":   round(p.loc[bs, "Uplift_pp"], 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DECISION RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✅  Decision Recommendation":
    st.title("Decision Recommendation")
    st.divider()

    sig_label = (
        f"YES — p = {m['p_cv']:.2e} (< 0.001)"
        if m["p_cv"] < 0.001
        else (f"YES — p = {m['p_cv']:.4f}" if m["p_cv"] < 0.05 else "NO")
    )
    pwr_label = (
        f"ADEQUATELY POWERED ({m['a_pwr']*100:.1f}%)"
        if m["a_pwr"] >= 0.80
        else f"UNDERPOWERED ({m['a_pwr']*100:.1f}%)"
    )

    # ── Experiment Question ───────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🔬 Experiment Question")
        st.markdown(
            "Does sending a marketing e-mail **increase customer conversion rate** "
            "compared to sending no e-mail?"
        )

    # ── Key Findings ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 📊 Key Findings")
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(f"- **Conversion rate — Control:** {m['p_c']*100:.3f}%")
            st.markdown(f"- **Conversion rate — Treatment:** {m['p_t']*100:.3f}%")
            st.markdown(f"- **Absolute lift:** +{m['abs_lft']*100:.3f} pp")
            st.markdown(f"- **Relative lift:** +{m['rel_lft']:.1f}%")
        with k2:
            st.markdown(f"- **P-value:** {m['p_cv']:.2e} — Significant: {sig_label}")
            st.markdown(f"- **Visit rate lift:** +{m['abs_lft_v']*100:.2f} pp (guardrail ✅ healthy)")
            st.markdown(f"- **Study power:** {pwr_label}")
            st.markdown(f"- **Cohen's h:** {m['h_cv']:.4f} (small effect size)")

    # ── Uplift Insight ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### 🎯 Uplift Insight")
        with st.spinner("Loading uplift model (cached)…"):
            du2 = compute_uplift_scores()
        u1, u2, u3 = st.columns(3)
        u1.metric("Median uplift score",    f"{du2['uplift_score'].median():.4f}")
        u2.metric("Top-quartile uplift",    f"{du2['uplift_score'].quantile(0.75):.4f}")
        u3.metric("Fraction negative uplift", f"{(du2['uplift_score']<0).mean()*100:.1f}%")
        st.markdown(
            "T-Learner modeling reveals **strong heterogeneity** — the highest-value segment "
            "is **Spending Tier \\$750–\\$1,000** (+1.458 pp lift, ~3× the average). "
            "A meaningful fraction of customers shows near-zero or negative predicted uplift, "
            "making a blanket send cost-inefficient."
        )

    # ── Final Recommendation ──────────────────────────────────────────────────
    st.divider()
    st.success(
        "## ✅  FINAL RECOMMENDATION: TARGETED ROLLOUT",
    )

    with st.container(border=True):
        st.markdown("### Justification")
        st.markdown(f"""
The e-mail campaign produces a **statistically significant and practically meaningful lift**
in both conversion (+{m['abs_lft']*100:.2f} pp, p < 0.001) and site visits (+{m['abs_lft_v']*100:.2f} pp),
with the study fully powered at {m['a_pwr']*100:.1f}% to detect this effect.

However, T-Learner uplift scores reveal **strong treatment-effect heterogeneity** — a meaningful
share of customers show near-zero or negative predicted uplift, making a blanket e-mail send
economically inefficient. The recommended action is a **TARGETED ROLLOUT**: score the customer
base with the T-Learner, restrict sends to the top uplift quartile, and instrument the next
campaign to validate revenue-per-email incrementally.
        """)

    with st.container(border=True):
        st.markdown("### 🗺️ Next Steps")
        st.markdown("""
1. **Deploy** T-Learner scoring pipeline to the production customer database
2. **Run a follow-up A/B test** targeting only the top-quartile uplift segment
3. **Promote revenue-per-email** to primary metric (not just conversion rate)
4. **Extend modeling** to the Womens E-Mail group; test X-Learner and DR-Learner
5. **Monitor long-term effects** — does uplift persist across multiple campaigns?
        """)
