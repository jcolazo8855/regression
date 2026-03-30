# ═══════════════════════════════════════════════════════════════════════════════
#  📐 Regression Lab — OLS · Ridge · LASSO  (interactive Streamlit demo)
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# ── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regression Lab",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@600;800&display=swap');

html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
.stApp { background: #0a0c10; color: #e8eaf0; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: #0d0f14 !important;
    border-right: 1px solid #232835 !important;
}
section[data-testid="stSidebar"] * { color: #8892a4 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label { color: #8892a4 !important; font-size: 12px !important; }

/* metric cards */
.metric-card {
    background: #111318; border: 1px solid #232835;
    padding: 14px 16px; border-radius: 0; margin-bottom: 0;
}
.metric-label { font-size: 10px; letter-spacing: 2px; color: #5a6070; text-transform: uppercase; }
.metric-value { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 22px; margin-top: 4px; }
.metric-row { display: flex; gap: 8px; margin-bottom: 16px; }
.metric-row .metric-card { flex: 1; }

/* section header */
.section-hdr {
    font-size: 10px; letter-spacing: 2px; color: #3a4050;
    text-transform: uppercase; border-bottom: 1px solid #1a1f2a;
    padding-bottom: 6px; margin-bottom: 14px;
}
/* model badge */
.model-badge {
    display: inline-block; padding: 3px 10px;
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    margin-bottom: 8px;
}
/* coef bar */
.coef-item { margin-bottom: 10px; }
.coef-header { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 11px; }
.coef-name { color: #8892a4; }
.coef-val  { font-weight: 700; }
.coef-track { height: 4px; background: #1a1f2a; border-radius: 0; }
.coef-fill-pos { height: 4px; background: #00e5ff; border-radius: 0; }
.coef-fill-neg { height: 4px; background: #ff6b6b; border-radius: 0; }
.coef-zero     { color: #5a6070 !important; font-style: italic; font-size: 10px; }

/* info box */
.info-box {
    background: #111318; border: 1px solid #232835; border-left: 3px solid;
    padding: 14px 16px; font-size: 12px; line-height: 1.8; margin-bottom: 16px;
}

/* page title */
.page-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 28px;
    letter-spacing: -1px; margin-bottom: 2px;
}
.page-sub { color: #3a4050; font-size: 12px; letter-spacing: 1px; margin-bottom: 24px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
DATASETS = {
    "🏠 House Prices": {
        "features": ["Size (sqft)", "Bedrooms", "Age (yrs)", "Location Score"],
        "true_coefs": [150.0, 8000.0, -500.0, 12000.0],
        "intercept": 50000.0,
        "noise_scale": 15000.0,
        "desc": "Predict house price from structural and location features.",
    },
    "📚 Exam Scores": {
        "features": ["Study Hours", "Sleep (hrs)", "Attendance %", "Prior Score"],
        "true_coefs": [4.5, 2.1, 0.3, 0.4],
        "intercept": 20.0,
        "noise_scale": 8.0,
        "desc": "Predict final exam score from student habits and history.",
    },
    "💉 Blood Pressure": {
        "features": ["Age", "Weight (kg)", "Sodium (mg)", "Exercise (hr/wk)"],
        "true_coefs": [0.5, 0.3, 0.008, -1.2],
        "intercept": 80.0,
        "noise_scale": 6.0,
        "desc": "Predict systolic blood pressure from patient lifestyle metrics.",
    },
    "📈 Stock Returns": {
        "features": ["P/E Ratio", "Volume (M)", "Volatility", "Momentum"],
        "true_coefs": [-0.15, 0.002, -0.8, 1.2],
        "intercept": 5.0,
        "noise_scale": 3.0,
        "desc": "Predict monthly stock returns from fundamental and technical factors.",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_data(dataset_name: str, n: int, noise: float, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    ds  = DATASETS[dataset_name]
    p   = len(ds["features"])

    # Generate correlated features (more realistic)
    base = rng.randn(n, p)
    corr = 0.25
    X_raw = base + corr * rng.randn(n, p)

    # Scale each feature to a reasonable range
    scales = [500, 3, 30, 10]  # rough feature-specific scales
    while len(scales) < p:
        scales.append(10)

    X = np.zeros((n, p))
    for j in range(p):
        X[:, j] = X_raw[:, j] * scales[j % len(scales)]

    coefs = np.array(ds["true_coefs"])
    y = X @ coefs + ds["intercept"] + noise * ds["noise_scale"] * rng.randn(n)

    return X, y

# ═══════════════════════════════════════════════════════════════════════════════
#  FITTING
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_COLORS = {
    "OLS":   "#a8ff78",
    "Ridge": "#00e5ff",
    "LASSO": "#ff6b6b",
}

def fit_model(X, y, model_name: str, alpha: float,
              normalize: bool, fit_intercept: bool):
    X_fit = X.copy()
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_fit  = scaler.fit_transform(X_fit)

    if model_name == "OLS":
        model = LinearRegression(fit_intercept=fit_intercept)
    elif model_name == "Ridge":
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
    else:  # LASSO
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept,
                      max_iter=50000, tol=1e-4)

    model.fit(X_fit, y)
    y_hat = model.predict(X_fit)

    # Recover original-scale coefficients when normalised
    if normalize and scaler is not None:
        coefs_orig = model.coef_ / scaler.scale_
    else:
        coefs_orig = model.coef_

    return model, coefs_orig, y_hat

def metrics(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mse    = ss_res / len(y)
    coef_norm = float(np.sum(coefs ** 2)) if (coefs := y_hat) is not None else 0.0
    return r2, mse

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    plot_bgcolor="#111318",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono", color="#5a6070", size=10),
    margin=dict(t=10, b=40, l=50, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    xaxis=dict(gridcolor="#1a1f2a", linecolor="#232835", zerolinecolor="#232835"),
    yaxis=dict(gridcolor="#1a1f2a", linecolor="#232835", zerolinecolor="#232835"),
)


def scatter_chart(X, y, y_hat, feature_name: str, model_name: str):
    x_vals = X[:, 0]
    color  = MODEL_COLORS[model_name]

    order  = np.argsort(x_vals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y, mode="markers", name="Actual",
        marker=dict(color="rgba(0,229,255,0.35)", size=5,
                    line=dict(color="rgba(0,229,255,0.7)", width=0.7)),
    ))
    fig.add_trace(go.Scatter(
        x=x_vals[order], y=y_hat[order], mode="lines", name=f"{model_name} fit",
        line=dict(color=color, width=2),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=300,
        xaxis_title=feature_name,
        yaxis_title="Target",
    )
    return fig


def coef_chart(coefs, feature_names: list, model_name: str):
    color  = MODEL_COLORS[model_name]
    colors = [color if c >= 0 else "#ff6b6b" for c in coefs]
    fig = go.Figure(go.Bar(
        x=feature_names, y=coefs,
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.add_hline(y=0, line_color="#3a4050", line_width=1)
    fig.update_layout(
        **PLOT_LAYOUT,
        height=220,
        yaxis_title="Coefficient",
        showlegend=False,
    )
    return fig


def reg_path_chart(X, y, feature_names: list, model_name: str,
                   normalize: bool, fit_intercept: bool,
                   current_alpha: float):
    """Sweep alpha and plot how coefficients change."""
    if model_name == "OLS":
        return None

    # Log-spaced alphas for a better view
    alphas = np.logspace(-3, 3, 80)
    paths  = {f: [] for f in feature_names}

    X_fit = X.copy()
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_fit  = scaler.fit_transform(X_fit)

    for a in alphas:
        if model_name == "Ridge":
            m = Ridge(alpha=a, fit_intercept=fit_intercept, max_iter=10000)
        else:
            m = Lasso(alpha=a, fit_intercept=fit_intercept,
                      max_iter=50000, tol=1e-4)
        m.fit(X_fit, y)
        c = m.coef_
        if normalize and scaler is not None:
            c = c / scaler.scale_
        for j, f in enumerate(feature_names):
            paths[f].append(float(c[j]))

    palette = ["#00e5ff", "#ff6b6b", "#a8ff78", "#ffbe5c",
               "#c084fc", "#f472b6", "#34d399", "#fb923c"]
    fig = go.Figure()
    for j, fname in enumerate(feature_names):
        fig.add_trace(go.Scatter(
            x=np.log10(alphas), y=paths[fname],
            mode="lines", name=fname,
            line=dict(color=palette[j % len(palette)], width=1.5),
            opacity=0.85,
        ))

    # Vertical line at current alpha
    if current_alpha > 0:
        log_cur = np.log10(current_alpha)
        fig.add_vline(x=log_cur, line_color="rgba(255,255,255,0.35)",
                      line_width=1, line_dash="dash",
                      annotation_text=f"  α={current_alpha:.3g}",
                      annotation_font=dict(color="rgba(255,255,255,0.5)", size=10))

    fig.add_hline(y=0, line_color="#3a4050", line_width=1, line_dash="dot")
    fig.update_layout(
        **PLOT_LAYOUT,
        height=220,
        xaxis_title="log₁₀(α)",
        yaxis_title="Coefficient value",
    )
    return fig


def residual_chart(y, y_hat, model_name: str):
    residuals = y - y_hat
    color = MODEL_COLORS[model_name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_hat, y=residuals, mode="markers",
        name="Residual",
        marker=dict(color=color, size=4, opacity=0.6),
    ))
    fig.add_hline(y=0, line_color="#3a4050", line_width=1, line_dash="dot")
    fig.update_layout(
        **PLOT_LAYOUT,
        height=220,
        xaxis_title="Fitted value",
        yaxis_title="Residual",
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📐 Regression Lab")
    st.markdown("---")

    st.markdown('<div class="section-hdr">DATASET</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox("", list(DATASETS.keys()), label_visibility="collapsed")

    st.markdown('<div class="section-hdr" style="margin-top:20px;">MODEL</div>', unsafe_allow_html=True)
    model_name = st.selectbox("", ["OLS", "Ridge", "LASSO"], label_visibility="collapsed")

    alpha = 1.0
    if model_name != "OLS":
        st.markdown(f'<div class="section-hdr" style="margin-top:20px;">REGULARIZATION (α)</div>',
                    unsafe_allow_html=True)
        alpha = st.slider("Alpha", min_value=0.001, max_value=10000.0,
                          value=1.0, step=0.001,
                          format="%.3f", label_visibility="collapsed")
        st.caption(f"α = {alpha:.3g}")

        # Per-dataset guidance for LASSO sparsity
        if model_name == "LASSO":
            sparsity_tips = {
                "🏠 House Prices":    "Try α > 2000 for sparsity (large y-scale)",
                "📚 Exam Scores":     "Try α ≈ 1–5 to see zeros appear",
                "💉 Blood Pressure":  "Try α ≈ 0.5–3 for sparsity",
                "📈 Stock Returns":   "Try α ≈ 0.1–1 for sparsity",
            }
            tip = sparsity_tips.get(dataset_name, "Increase α to drive coefs to zero")
            st.caption(f"💡 {tip}")

    st.markdown('<div class="section-hdr" style="margin-top:20px;">DATA</div>', unsafe_allow_html=True)
    n_samples = st.slider("Sample size",      50, 500, 150, step=10)
    noise_lvl = st.slider("Noise level",       0.0, 3.0, 1.0, step=0.1)

    st.markdown('<div class="section-hdr" style="margin-top:20px;">OPTIONS</div>', unsafe_allow_html=True)
    normalize     = st.checkbox("Standardize features", value=True)
    fit_intercept = st.checkbox("Fit intercept",         value=True)

    st.markdown("---")
    st.markdown("""
<div style="font-size:10px; color:#3a4050; line-height:1.7;">
<b style="color:#5a6070;">OLS</b> — minimises RSS, no penalty<br>
<b style="color:#5a6070;">Ridge</b> — L2 penalty, shrinks all coefs<br>
<b style="color:#5a6070;">LASSO</b> — L1 penalty, sparse solution<br><br>
Larger α → stronger regularization
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — data + fit
# ═══════════════════════════════════════════════════════════════════════════════
ds     = DATASETS[dataset_name]
color  = MODEL_COLORS[model_name]

X, y = generate_data(dataset_name, n_samples, noise_lvl)
model, coefs, y_hat = fit_model(X, y, model_name, alpha, normalize, fit_intercept)
r2, mse = metrics(y, y_hat)
coef_norm = float(np.sum(coefs ** 2))
intercept_val = float(model.intercept_) if fit_intercept else None
n_zero = int(np.sum(np.abs(coefs) < 1e-6))

# ═══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="page-title">Regression <span style="color:{color}">Lab</span></div>'
    f'<div class="page-sub">{dataset_name} &nbsp;·&nbsp; {model_name} &nbsp;·&nbsp; '
    f'n={n_samples} &nbsp;·&nbsp; noise={noise_lvl:.1f}</div>',
    unsafe_allow_html=True,
)

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

def metric_html(label, value, color="#a8ff78"):
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
    </div>"""

with c1:
    st.markdown(metric_html("R²", f"{r2:.4f}", color), unsafe_allow_html=True)
with c2:
    st.markdown(metric_html("MSE", f"{mse:.2f}" if mse < 1e6 else f"{mse:.2e}", color), unsafe_allow_html=True)
with c3:
    st.markdown(metric_html("‖β‖²", f"{coef_norm:.4f}", color), unsafe_allow_html=True)
with c4:
    if model_name == "LASSO":
        st.markdown(metric_html("Zero coefs", f"{n_zero} / {len(coefs)}", "#ffbe5c"),
                    unsafe_allow_html=True)
    else:
        iv = f"{intercept_val:.2f}" if intercept_val is not None else "—"
        st.markdown(metric_html("Intercept", iv, color), unsafe_allow_html=True)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# ── Model info box ────────────────────────────────────────────────────────────
MODEL_INFO = {
    "OLS": (
        "#a8ff78",
        "Ordinary Least Squares minimises the residual sum of squares (RSS) with no "
        "constraint on coefficient magnitudes. It is the BLUE estimator (Gauss-Markov) "
        "when assumptions hold, but can overfit with correlated or many features.",
    ),
    "Ridge": (
        "#00e5ff",
        "Ridge adds an L2 penalty λ‖β‖² to the RSS objective. All coefficients are "
        "shrunk toward zero proportionally — none reach exactly zero. It handles "
        "multicollinearity well and has a closed-form solution.",
    ),
    "LASSO": (
        "#ff6b6b",
        "LASSO adds an L1 penalty λ‖β‖₁. The L1 geometry drives some coefficients "
        "to exactly zero, performing automatic feature selection. Large α produces "
        "sparse models. No closed-form; solved via coordinate descent.",
    ),
}
info_color, info_text = MODEL_INFO[model_name]
alpha_note = f" &nbsp;·&nbsp; α = {alpha:.3g}" if model_name != "OLS" else ""
st.markdown(
    f'<div class="info-box" style="border-color:{info_color};">'
    f'<b style="color:{info_color};">{model_name}{alpha_note}</b><br>'
    f'{info_text}</div>',
    unsafe_allow_html=True,
)

# ── Main charts row ───────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-hdr">SCATTER — feature₁ vs target</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        scatter_chart(X, y, y_hat, ds["features"][0], model_name),
        use_container_width=True, config={"displayModeBar": False},
    )

with col_right:
    st.markdown('<div class="section-hdr">COEFFICIENTS</div>', unsafe_allow_html=True)

    max_abs = max(np.abs(coefs).max(), 1e-9)
    for j, fname in enumerate(ds["features"]):
        val  = float(coefs[j])
        pct  = abs(val) / max_abs * 100
        is_zero = abs(val) < 1e-6
        val_str = (
            f'<span class="coef-zero">≈ 0 (removed)</span>'
            if is_zero else
            f'<span style="color:{color}; font-weight:700;">{val:.4f}</span>'
        )
        fill_cls = "coef-fill-neg" if val < 0 else "coef-fill-pos"
        st.markdown(f"""
        <div class="coef-item">
            <div class="coef-header">
                <span class="coef-name">{fname}</span>{val_str}
            </div>
            <div class="coef-track"><div class="{fill_cls}" style="width:{pct:.1f}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr" style="margin-top:16px;">COEFFICIENT CHART</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        coef_chart(coefs, ds["features"], model_name),
        use_container_width=True, config={"displayModeBar": False},
    )

# ── Second row: regularization path + residuals ───────────────────────────────
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
col_path, col_resid = st.columns(2)

with col_path:
    if model_name != "OLS":
        st.markdown('<div class="section-hdr">REGULARIZATION PATH</div>',
                    unsafe_allow_html=True)
        path_fig = reg_path_chart(
            X, y, ds["features"], model_name,
            normalize, fit_intercept, alpha,
        )
        if path_fig:
            st.plotly_chart(path_fig, use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.markdown('<div class="section-hdr">ABOUT OLS</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="background:#111318; border:1px solid #232835; padding:16px; font-size:12px; line-height:1.8; color:#5a6070;">
OLS has no hyperparameter — there is no regularization path to plot.<br><br>
Switch to <b style="color:#00e5ff;">Ridge</b> or <b style="color:#ff6b6b;">LASSO</b>
to see how coefficient values change across the full range of α values.<br><br>
The regularization path shows whether features are important at all
values of α (stable path) or only at weak regularization (shrinks quickly).
</div>
""", unsafe_allow_html=True)

with col_resid:
    st.markdown('<div class="section-hdr">RESIDUALS vs FITTED</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        residual_chart(y, y_hat, model_name),
        use_container_width=True, config={"displayModeBar": False},
    )

# ── Model comparison table ────────────────────────────────────────────────────
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-hdr">SIDE-BY-SIDE MODEL COMPARISON</div>',
            unsafe_allow_html=True)

rows = []
for mn in ["OLS", "Ridge", "LASSO"]:
    _, c_comp, yh_comp = fit_model(X, y, mn, alpha, normalize, fit_intercept)
    r2_c, mse_c = metrics(y, yh_comp)
    nz = int(np.sum(np.abs(c_comp) < 1e-6))
    rows.append({
        "Model":       mn,
        "R²":          round(float(r2_c), 4),
        "MSE":         round(float(mse_c), 2),
        "‖β‖²":        round(float(np.sum(c_comp**2)), 4),
        "Zero coefs":  f"{nz}/{len(c_comp)}",
        **{f: round(float(c_comp[j]), 4) for j, f in enumerate(ds["features"])},
    })

df_comp = pd.DataFrame(rows).set_index("Model")

def highlight_model(row):
    styles = []
    for col in row.index:
        if row.name == model_name:
            styles.append(f"background-color: {color}18; color: {color}; font-weight: bold;")
        else:
            styles.append("color: #5a6070;")
    return styles

st.dataframe(
    df_comp.style.apply(highlight_model, axis=1),
    use_container_width=True,
)

st.markdown(
    f'<div style="font-size:10px; color:#3a4050; text-align:right; margin-top:4px;">'
    f'Comparison uses same α={alpha:.3g} for Ridge & LASSO &nbsp;·&nbsp; '
    f'n={n_samples} &nbsp;·&nbsp; noise={noise_lvl:.1f}</div>',
    unsafe_allow_html=True,
)
