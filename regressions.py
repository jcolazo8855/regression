# ═══════════════════════════════════════════════════════════════════════════════
#  📐 Regression Lab — OLS · Ridge · LASSO
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regression Lab",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — white theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App background */
.stApp { background: #ffffff; color: #1a1a2e; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f8f9fc !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] * { color: #374151 !important; }
section[data-testid="stSidebar"] label {
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

/* Metric cards */
.metric-card {
    background: #f8f9fc;
    border: 1px solid #e2e8f0;
    border-top: 3px solid;
    padding: 14px 16px;
    border-radius: 6px;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: #94a3b8;
    text-transform: uppercase;
}
.metric-value {
    font-weight: 700;
    font-size: 24px;
    margin-top: 6px;
    letter-spacing: -0.5px;
}

/* Section headers */
.section-hdr {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #94a3b8;
    text-transform: uppercase;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 4px;
}

/* Info box */
.info-box {
    background: #f8f9fc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.75;
    margin-bottom: 18px;
    border-radius: 0 6px 6px 0;
    color: #374151;
}

/* Page title */
.page-title {
    font-weight: 800;
    font-size: 30px;
    letter-spacing: -1px;
    margin-bottom: 2px;
    color: #0f172a;
}
.page-sub {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 400;
    margin-bottom: 24px;
}

/* Coef bars */
.coef-item { margin-bottom: 12px; }
.coef-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
    font-size: 12px;
    font-weight: 500;
}
.coef-name { color: #64748b; }
.coef-val  { font-weight: 700; font-size: 12px; }
.coef-track {
    height: 5px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
}
.coef-fill { height: 5px; border-radius: 3px; }
.coef-zero { color: #94a3b8 !important; font-style: italic; font-size: 11px; }

/* OLS info panel */
.ols-info {
    background: #f8f9fc;
    border: 1px solid #e2e8f0;
    padding: 18px;
    font-size: 13px;
    line-height: 1.8;
    color: #64748b;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PALETTE  (works on white)
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_COLORS = {
    "OLS":   "#16a34a",   # green-600
    "Ridge": "#2563eb",   # blue-600
    "LASSO": "#dc2626",   # red-600
}
MODEL_BG = {
    "OLS":   "#f0fdf4",
    "Ridge": "#eff6ff",
    "LASSO": "#fef2f2",
}
MODEL_BORDER = {
    "OLS":   "#86efac",
    "Ridge": "#93c5fd",
    "LASSO": "#fca5a5",
}

# Reg-path palette — distinct, readable on white
PATH_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706",
                "#7c3aed", "#db2777", "#0891b2", "#ea580c"]

# ── Shared Plotly layout (white theme) ────────────────────────────────────────
FONT_AXIS = dict(family="Inter, sans-serif", size=12, color="#374151")
FONT_TICK = dict(family="Inter, sans-serif", size=11, color="#6b7280")

def base_layout(height: int = 300, **extra) -> dict:
    return dict(
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        margin=dict(t=16, b=52, l=60, r=24),
        height=height,
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        ),
        xaxis=dict(
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            zerolinecolor="#e2e8f0",
            tickfont=FONT_TICK,
            title_font=FONT_AXIS,
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            linecolor="#e2e8f0",
            zerolinecolor="#e2e8f0",
            tickfont=FONT_TICK,
            title_font=FONT_AXIS,
        ),
        **extra,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
DATASETS = {
    "🏠 House Prices": {
        "features":   ["Size (sqft)", "Bedrooms", "Age (yrs)", "Location Score"],
        "true_coefs": [150.0, 8000.0, -500.0, 12000.0],
        "intercept":  50000.0,
        "noise_scale": 15000.0,
        "desc": "Predict house price from structural and location features.",
    },
    "📚 Exam Scores": {
        "features":   ["Study Hours", "Sleep (hrs)", "Attendance %", "Prior Score"],
        "true_coefs": [4.5, 2.1, 0.3, 0.4],
        "intercept":  20.0,
        "noise_scale": 8.0,
        "desc": "Predict final exam score from student habits and history.",
    },
    "💉 Blood Pressure": {
        "features":   ["Age", "Weight (kg)", "Sodium (mg)", "Exercise (hr/wk)"],
        "true_coefs": [0.5, 0.3, 0.008, -1.2],
        "intercept":  80.0,
        "noise_scale": 6.0,
        "desc": "Predict systolic blood pressure from patient lifestyle metrics.",
    },
    "📈 Stock Returns": {
        "features":   ["P/E Ratio", "Volume (M)", "Volatility", "Momentum"],
        "true_coefs": [-0.15, 0.002, -0.8, 1.2],
        "intercept":  5.0,
        "noise_scale": 3.0,
        "desc": "Predict monthly stock returns from fundamental and technical factors.",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_data(dataset_name: str, n: int, noise: float, seed: int = 42):
    rng    = np.random.RandomState(seed)
    ds     = DATASETS[dataset_name]
    p      = len(ds["features"])
    base   = rng.randn(n, p)
    X_raw  = base + 0.25 * rng.randn(n, p)
    scales = [500, 3, 30, 10]
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
def fit_model(X, y, model_name: str, alpha: float,
              normalize: bool, fit_intercept: bool):
    X_fit  = X.copy()
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_fit  = scaler.fit_transform(X_fit)

    if model_name == "OLS":
        model = LinearRegression(fit_intercept=fit_intercept)
    elif model_name == "Ridge":
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
    else:
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept,
                      max_iter=50000, tol=1e-4)

    model.fit(X_fit, y)
    y_hat = model.predict(X_fit)

    coefs_orig = model.coef_ / scaler.scale_ if (normalize and scaler) else model.coef_
    return model, scaler, coefs_orig, y_hat


def calc_metrics(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2  = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mse = float(ss_res / len(y))
    return r2, mse

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def scatter_chart(X, y, model, scaler, feature_name: str,
                  model_name: str, normalize: bool) -> go.Figure:
    """
    Plot observed points + a smooth fitted line.

    The line is built by sweeping feature₁ across its observed range in 200
    steps while holding all other features fixed at their column means.
    This gives a clean straight line for linear models.
    """
    color  = MODEL_COLORS[model_name]
    x_obs  = X[:, 0]

    # Build smooth grid over feature 1, others held at mean
    x_grid = np.linspace(x_obs.min(), x_obs.max(), 200)
    X_grid = np.tile(X.mean(axis=0), (200, 1))
    X_grid[:, 0] = x_grid

    X_grid_fit = scaler.transform(X_grid) if (normalize and scaler) else X_grid
    y_grid = model.predict(X_grid_fit)

    fig = go.Figure()

    # Observed scatter
    fig.add_trace(go.Scatter(
        x=x_obs, y=y,
        mode="markers",
        name="Observed",
        marker=dict(
            color="rgba(30,30,30,0.12)",
            size=6,
            line=dict(color="rgba(30,30,30,0.35)", width=1),
        ),
    ))

    # Smooth fitted line
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_grid,
        mode="lines",
        name=f"{model_name} fit",
        line=dict(color=color, width=2.5),
    ))

    fig.update_layout(
        **base_layout(height=320,
                      xaxis_title=feature_name,
                      yaxis_title="Target"),
    )
    return fig


def coef_chart(coefs, feature_names: list, model_name: str) -> go.Figure:
    color  = MODEL_COLORS[model_name]
    colors = [color if c >= 0 else "#dc2626" for c in coefs]

    fig = go.Figure(go.Bar(
        x=feature_names,
        y=coefs,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.3f}" for v in coefs],
        textposition="outside",
        textfont=dict(family="Inter, sans-serif", size=11, color="#374151"),
    ))
    fig.add_hline(y=0, line_color="#cbd5e1", line_width=1.5)
    fig.update_layout(
        **base_layout(height=240,
                      yaxis_title="Coefficient value",
                      showlegend=False),
    )
    fig.update_yaxes(range=[
        min(min(coefs) * 1.3, -0.1),
        max(max(coefs) * 1.3,  0.1),
    ])
    return fig


def reg_path_chart(X, y, feature_names: list, model_name: str,
                   normalize: bool, fit_intercept: bool,
                   current_alpha: float) -> go.Figure | None:
    if model_name == "OLS":
        return None

    alphas = np.logspace(-3, 4, 100)
    paths  = {f: [] for f in feature_names}

    X_fit  = X.copy()
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_fit  = scaler.fit_transform(X_fit)

    for a in alphas:
        m = (Ridge(alpha=a, fit_intercept=fit_intercept, max_iter=10000)
             if model_name == "Ridge"
             else Lasso(alpha=a, fit_intercept=fit_intercept, max_iter=50000, tol=1e-4))
        m.fit(X_fit, y)
        c = m.coef_ / scaler.scale_ if (normalize and scaler) else m.coef_
        for j, f in enumerate(feature_names):
            paths[f].append(float(c[j]))

    fig = go.Figure()
    for j, fname in enumerate(feature_names):
        fig.add_trace(go.Scatter(
            x=np.log10(alphas),
            y=paths[fname],
            mode="lines",
            name=fname,
            line=dict(color=PATH_PALETTE[j % len(PATH_PALETTE)], width=2),
        ))

    # Current alpha marker
    if current_alpha > 0:
        log_cur = np.log10(current_alpha)
        fig.add_vline(
            x=log_cur,
            line_color="#64748b", line_width=1.5, line_dash="dash",
            annotation_text=f"α = {current_alpha:.3g}",
            annotation_font=dict(family="Inter, sans-serif",
                                 color="#374151", size=12),
            annotation_position="top right",
        )

    fig.add_hline(y=0, line_color="#cbd5e1", line_width=1, line_dash="dot")
    fig.update_layout(
        **base_layout(height=260,
                      xaxis_title="log₁₀(α)",
                      yaxis_title="Coefficient value"),
    )
    return fig


def residual_chart(y, y_hat, model_name: str) -> go.Figure:
    color     = MODEL_COLORS[model_name]
    residuals = y - y_hat

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_hat, y=residuals,
        mode="markers",
        name="Residual",
        marker=dict(color=color, size=5, opacity=0.55,
                    line=dict(color=color, width=0.5)),
    ))
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1.5, line_dash="dot")
    fig.update_layout(
        **base_layout(height=260,
                      xaxis_title="Fitted value",
                      yaxis_title="Residual",
                      showlegend=False),
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📐 Regression Lab")
    st.markdown("---")

    st.markdown("**Dataset**")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()),
                                label_visibility="collapsed")

    st.markdown("**Model**")
    model_name = st.selectbox("Model", ["OLS", "Ridge", "LASSO"],
                              label_visibility="collapsed")

    alpha = 1.0
    if model_name != "OLS":
        st.markdown(f"**Regularization strength (α)**")
        alpha = st.slider("alpha", 0.001, 10000.0, 1.0, 0.001,
                          format="%.3f", label_visibility="collapsed")
        st.caption(f"α = {alpha:.3g}")

        if model_name == "LASSO":
            tips = {
                "🏠 House Prices":   "Try α > 2 000 to see zeros",
                "📚 Exam Scores":    "Try α ≈ 1–5 to see zeros",
                "💉 Blood Pressure": "Try α ≈ 0.5–3 to see zeros",
                "📈 Stock Returns":  "Try α ≈ 0.1–1 to see zeros",
            }
            st.caption(f"💡 {tips.get(dataset_name, 'Increase α to zero coefs')}")

    st.markdown("**Data settings**")
    n_samples = st.slider("Sample size",  50, 500, 150, 10)
    noise_lvl = st.slider("Noise level", 0.0, 3.0, 1.0, 0.1)

    st.markdown("**Options**")
    normalize     = st.checkbox("Standardize features", value=True)
    fit_intercept = st.checkbox("Fit intercept",         value=True)

    st.markdown("---")
    st.markdown("""
**OLS** — no penalty, minimises RSS  
**Ridge** — L2 penalty, shrinks all β  
**LASSO** — L1 penalty, sparsifies β  

Larger α → stronger regularization
""")

# ═══════════════════════════════════════════════════════════════════════════════
#  FIT
# ═══════════════════════════════════════════════════════════════════════════════
ds    = DATASETS[dataset_name]
color = MODEL_COLORS[model_name]
bg    = MODEL_BG[model_name]
bdr   = MODEL_BORDER[model_name]

X, y = generate_data(dataset_name, n_samples, noise_lvl)
model, scaler, coefs, y_hat = fit_model(X, y, model_name, alpha,
                                        normalize, fit_intercept)
r2, mse     = calc_metrics(y, y_hat)
coef_norm   = float(np.sum(coefs ** 2))
intercept_v = float(model.intercept_) if fit_intercept else None
n_zero      = int(np.sum(np.abs(coefs) < 1e-6))

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="page-title">Regression '
    f'<span style="color:{color};">Lab</span></div>'
    f'<div class="page-sub">'
    f'{dataset_name} &nbsp;·&nbsp; '
    f'<b style="color:{color};">{model_name}</b>'
    f'{f" &nbsp;·&nbsp; α = {alpha:.3g}" if model_name != "OLS" else ""}'
    f' &nbsp;·&nbsp; n = {n_samples} &nbsp;·&nbsp; noise = {noise_lvl:.1f}'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

def metric_card(label, value, accent_color, border_color):
    return (
        f'<div class="metric-card" '
        f'style="border-top-color:{accent_color}; background:{bg};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{accent_color};">{value}</div>'
        f'</div>'
    )

with c1:
    st.markdown(metric_card("R²", f"{r2:.4f}", color, bdr), unsafe_allow_html=True)
with c2:
    mse_str = f"{mse:.2f}" if mse < 1e6 else f"{mse:.2e}"
    st.markdown(metric_card("MSE", mse_str, color, bdr), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("‖β‖²", f"{coef_norm:.4f}", color, bdr), unsafe_allow_html=True)
with c4:
    if model_name == "LASSO":
        st.markdown(metric_card("Zero coefs", f"{n_zero} / {len(coefs)}",
                                "#d97706", "#fde68a"), unsafe_allow_html=True)
    else:
        iv = f"{intercept_v:.2f}" if intercept_v is not None else "—"
        st.markdown(metric_card("Intercept", iv, color, bdr), unsafe_allow_html=True)

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

# ── Model info ────────────────────────────────────────────────────────────────
MODEL_INFO = {
    "OLS": (
        "Ordinary Least Squares minimises the residual sum of squares (RSS) with no "
        "constraint on coefficient magnitudes. It is the BLUE estimator (Gauss-Markov) "
        "when assumptions hold, but can overfit with many correlated features."
    ),
    "Ridge": (
        "Ridge adds an L2 penalty λ‖β‖² to the RSS objective. All coefficients are "
        "shrunk toward zero proportionally — none reach exactly zero. It handles "
        "multicollinearity well and has a closed-form analytical solution."
    ),
    "LASSO": (
        "LASSO adds an L1 penalty λ‖β‖₁. The L1 geometry drives some coefficients "
        "to exactly zero, performing automatic feature selection. Large α produces "
        "sparse models. No closed-form — solved via coordinate descent."
    ),
}
st.markdown(
    f'<div class="info-box" style="border-left-color:{color}; background:{bg};">'
    f'<strong style="color:{color};">{model_name}</strong>'
    f'{"  ·  α = " + f"{alpha:.3g}" if model_name != "OLS" else ""}'
    f'<br>{MODEL_INFO[model_name]}</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CHARTS ROW
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-hdr">Scatter — observed vs fitted line</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        scatter_chart(X, y, model, scaler, ds["features"][0],
                      model_name, normalize),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with col_right:
    st.markdown('<div class="section-hdr">Coefficients</div>', unsafe_allow_html=True)

    max_abs = max(float(np.abs(coefs).max()), 1e-9)
    for j, fname in enumerate(ds["features"]):
        val     = float(coefs[j])
        pct     = abs(val) / max_abs * 100
        is_zero = abs(val) < 1e-6

        val_str = (
            '<span class="coef-zero">≈ 0 (eliminated by LASSO)</span>'
            if is_zero else
            f'<span class="coef-val" style="color:{color};">{val:+.4f}</span>'
        )
        bar_color = color if val >= 0 else "#dc2626"
        st.markdown(f"""
        <div class="coef-item">
            <div class="coef-header">
                <span class="coef-name">{fname}</span>{val_str}
            </div>
            <div class="coef-track">
                <div class="coef-fill"
                     style="width:{pct:.1f}%; background:{bar_color};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr" style="margin-top:20px;">Coefficient chart</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        coef_chart(coefs, ds["features"], model_name),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  SECOND ROW — reg path + residuals
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
col_path, col_resid = st.columns(2)

with col_path:
    st.markdown('<div class="section-hdr">Regularization path</div>',
                unsafe_allow_html=True)
    if model_name != "OLS":
        path_fig = reg_path_chart(X, y, ds["features"], model_name,
                                  normalize, fit_intercept, alpha)
        if path_fig:
            st.plotly_chart(path_fig, use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.markdown("""
<div class="ols-info">
OLS has no regularization hyperparameter — there is no path to plot.<br><br>
Switch to <strong>Ridge</strong> or <strong>LASSO</strong> to see how coefficients
change across the full α range.<br><br>
A stable path means a feature is important regardless of regularization strength.
A path that collapses quickly means the feature is weak or correlated with others.
</div>""", unsafe_allow_html=True)

with col_resid:
    st.markdown('<div class="section-hdr">Residuals vs fitted</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        residual_chart(y, y_hat, model_name),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-hdr">Side-by-side model comparison</div>',
            unsafe_allow_html=True)

rows = []
for mn in ["OLS", "Ridge", "LASSO"]:
    _, _, c_comp, yh_comp = fit_model(X, y, mn, alpha, normalize, fit_intercept)
    r2_c, mse_c = calc_metrics(y, yh_comp)
    nz = int(np.sum(np.abs(c_comp) < 1e-6))
    rows.append({
        "Model":      mn,
        "R²":         round(float(r2_c), 4),
        "MSE":        round(float(mse_c), 2),
        "‖β‖²":       round(float(np.sum(c_comp ** 2)), 4),
        "Zero coefs": f"{nz}/{len(c_comp)}",
        **{f: round(float(c_comp[j]), 4)
           for j, f in enumerate(ds["features"])},
    })

df_comp = pd.DataFrame(rows).set_index("Model")

def highlight_model(row):
    mc = MODEL_COLORS[row.name]
    mbg = MODEL_BG[row.name]
    if row.name == model_name:
        return [f"background:{mbg}; color:{mc}; font-weight:600;"] * len(row)
    return ["color:#94a3b8;"] * len(row)

st.dataframe(
    df_comp.style.apply(highlight_model, axis=1),
    use_container_width=True,
)

st.caption(
    f"Ridge and LASSO both use α = {alpha:.3g}  ·  "
    f"n = {n_samples}  ·  noise = {noise_lvl:.1f}  ·  "
    f"{'features standardized' if normalize else 'features not standardized'}"
)
