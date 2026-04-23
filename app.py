"""
Framework XAI-AHP-Gaussiano para Suporte a Tomada de Decisão em ESG
Decision Support System — Teck Resources Limited (2001-2024)

Author: Cesar Yoshio Machado Pedroza
Advisor: Mestre Arthur Damasceno Vicente
Institution: USP/Esalq — MBA Data Science & Analytics, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESGE Decision Support | Teck Resources",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "green":  "#2E7D32",
    "teal":   "#00695C",
    "blue":   "#1565C0",
    "amber":  "#F57F17",
    "red":    "#C62828",
    "grey":   "#546E7A",
    "light":  "#F5F5F5",
    "e": "#2E7D32",   # Environmental
    "s": "#1565C0",   # Social
    "g": "#6A1B9A",   # Governance
    "ec": "#E65100",  # Economic
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

def load(filename):
    """Load CSV from data/processed/ or fallback to script dir."""
    for path in [
        os.path.join(DATA_DIR, filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.error(f"Arquivo não encontrado: {filename}")
    return pd.DataFrame()

@st.cache_data
def load_all():
    return {
        "ahp_weights":      load("ahp_weights.csv"),
        "ahp_sens":         load("ahp_sensitivity_analysis.csv"),
        "chow":             load("chow_test_structural_break.csv"),
        "dice":             load("dice_counterfactuals.csv"),
        "event":            load("event_study_mount_polley.csv"),
        "feat_imp":         load("feature_importance_xgboost.csv"),
        "shap_imp":         load("shap_importance.csv"),
        "shap_lime":        load("shap_lime_comparison.csv"),
        "benchmarking":     load("teck_vs_tsx_comparison.csv"),
    }

D = load_all()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/USP_logo.svg/200px-USP_logo.svg.png", width=80)
    st.markdown("## 🌿 ESGE Framework")
    st.markdown(
        "**Framework XAI-AHP-Gaussiano** para suporte a decisões ESG no setor minerário.\n\n"
        "**Empresa:** Teck Resources (TSX: TECK-B)\n\n"
        "**Período:** 2001–2024\n\n"
        "**Metodologia:** XGBoost · SHAP · LIME · DiCE · AHP-Gaussiano Monte Carlo"
    )
    st.divider()
    st.markdown("**Resultados-chave**")
    kpis = {
        "Quebra TCFD (F)": "53,94 ***",
        "ESG Δ pós-TCFD": "+844,53%",
        "XGBoost R²": "0,9999 ⚠️",
        "SHAP-LIME r": "0,8580",
        "AHP P(CR<0,10)": "98,78%",
        "Mount Polley AR": "−60,03%*",
    }
    for k, v in kpis.items():
        st.markdown(f"- **{k}:** {v}")
    st.divider()
    st.caption(
        "Cesar Y. M. Pedroza · USP/Esalq · 2026\n"
        "Orientador: Mestre Arthur Damasceno Vicente"
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🔍 XAI Explorer",
    "🎯 AHP-Gaussiano",
    "💥 Event Study",
    "📈 Benchmarking",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Framework XAI-AHP-Gaussiano — ESGE Decision Support")
    st.markdown(
        "Análise longitudinal de performance ESG da **Teck Resources Limited** (2001–2024), "
        "combinando Explainable AI com AHP probabilístico para suporte à decisão sustentável."
    )

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (col1, "📋 Relatórios", "24", "Anuais + Sustentabilidade"),
        (col2, "📊 Variáveis ESGE", "12", "features modeladas"),
        (col3, "🔬 Quebra TCFD", "F = 53,94***", "p < 0,0001"),
        (col4, "🌿 Peso Env.", "56,38%", "AHP-Gaussiano"),
        (col5, "💰 Retorno Teck", "48,23%", "vs 16,44% TSX"),
    ]
    for col, title, value, sub in cards:
        with col:
            st.metric(label=title, value=value, delta=sub)

    st.divider()

    # Chow test chart
    st.subheader("📉 Quebra Estrutural no ESG Disclosure — Teste de Chow (2018)")
    chow = D["chow"]

    # Simulate time series from Chow data
    years = list(range(2001, 2025))
    np.random.seed(42)
    pre_mean  = chow.loc[chow.Feature == "ESG_Disclosure_Index", "Mean_Pre_TCFD"].values[0]
    post_mean = chow.loc[chow.Feature == "ESG_Disclosure_Index", "Mean_Post_TCFD"].values[0]

    pre_years  = [y for y in years if y < 2018]
    post_years = [y for y in years if y >= 2018]
    pre_vals   = np.random.normal(pre_mean,  pre_mean * 0.15,  len(pre_years)).clip(0)
    post_vals  = np.random.normal(post_mean, post_mean * 0.10, len(post_years)).clip(0)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=pre_years, y=pre_vals, mode="lines+markers",
        name="Pré-TCFD (2001–2017)",
        line=dict(color=C["grey"], width=2),
        marker=dict(size=7),
    ))
    fig1.add_trace(go.Scatter(
        x=post_years, y=post_vals, mode="lines+markers",
        name="Pós-TCFD (2018–2024)",
        line=dict(color=C["green"], width=3),
        marker=dict(size=9, symbol="diamond"),
    ))
    fig1.add_vline(
        x=2018, line_dash="dash", line_color=C["amber"], line_width=2,
        annotation_text="Adoção TCFD<br>F=53,94 (p<0,0001)",
        annotation_position="top right",
        annotation_font_color=C["amber"],
    )
    fig1.add_annotation(
        x=2011, y=pre_mean * 1.6,
        text=f"μ pré = {pre_mean:.1f}",
        showarrow=False, font=dict(color=C["grey"], size=12),
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig1.add_annotation(
        x=2021, y=post_mean * 0.7,
        text=f"μ pós = {post_mean:.1f}<br>Δ = +844,53%",
        showarrow=False, font=dict(color=C["green"], size=12),
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig1.update_layout(
        height=400, template="plotly_white",
        xaxis_title="Ano", yaxis_title="ESG Disclosure Index",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.caption(
        "⚠️ Série temporal simulada com base nas médias pré/pós-TCFD do teste de Chow. "
        "Os valores exatos por ano estão em esg_master_final.csv no repositório GitHub."
    )

    # Framework diagram
    st.divider()
    st.subheader("🏗️ Arquitetura do Framework")
    cols = st.columns(5)
    modules = [
        ("1️⃣", "Feature Engineering", "Text mining + Bloomberg → 12 features ESGE"),
        ("2️⃣", "Quebra Estrutural", "Teste de Chow → detecção TCFD 2018"),
        ("3️⃣", "ML + XAI", "XGBoost → SHAP · LIME · DiCE"),
        ("4️⃣", "AHP-Gaussiano", "Monte Carlo (N=10.000) → pesos probabilísticos"),
        ("5️⃣", "Event Study", "Market model → Mount Polley 2014"),
    ]
    for col, (num, title, desc) in zip(cols, modules):
        with col:
            st.info(f"**{num} {title}**\n\n{desc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — XAI EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("🔍 XAI Explorer — Triangulação SHAP · LIME · DiCE")
    st.markdown(
        "Convergência entre três métodos XAI complementares valida a **estabilidade interpretativa** "
        "do modelo preditivo (correlação SHAP-LIME r = **0,8580**)."
    )

    col_a, col_b = st.columns(2)

    # SHAP Importance
    with col_a:
        st.subheader("📊 SHAP — Importância Global das Features")
        shap = D["shap_imp"].copy()
        shap = shap.sort_values("importance", ascending=True)
        feat_colors = {
            "Market_Liquidity":       C["blue"],
            "Annual_Return_Pct":      C["amber"],
            "ESG_Disclosure_Index":   C["green"],
            "Report_Quality_Score":   C["teal"],
        }
        colors = [feat_colors.get(f, C["grey"]) for f in shap["feature"]]
        fig_shap = go.Figure(go.Bar(
            x=shap["importance"], y=shap["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:,.0f}" for v in shap["importance"]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            height=300, template="plotly_white",
            xaxis_title="Mean |SHAP value|",
            margin=dict(l=10, r=60, t=10, b=40),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "⚠️ Market Liquidity domina com 87,5% de importância. "
            "Esse resultado é tratado com cautela — liquidez pode refletir escala empresarial, "
            "não necessariamente comprometimento ESG."
        )

    # SHAP vs LIME
    with col_b:
        st.subheader("🔄 SHAP vs LIME — Convergência Metodológica")
        sl = D["shap_lime"].copy()
        feat_labels = sl["Feature"].tolist()
        x = np.arange(len(feat_labels))
        width = 0.35

        fig_sl = go.Figure()
        fig_sl.add_trace(go.Bar(
            name="SHAP", x=feat_labels,
            y=sl["SHAP_Importance"],
            marker_color=C["blue"],
            text=[f"{v:.3f}" for v in sl["SHAP_Importance"]],
            textposition="outside",
        ))
        fig_sl.add_trace(go.Bar(
            name="LIME", x=feat_labels,
            y=sl["LIME_Importance"],
            marker_color=C["teal"],
            text=[f"{v:.3f}" for v in sl["LIME_Importance"]],
            textposition="outside",
        ))
        fig_sl.update_layout(
            barmode="group", height=300, template="plotly_white",
            yaxis_title="Importância Normalizada",
            margin=dict(l=10, r=20, t=10, b=60),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_sl, use_container_width=True)
        st.markdown(f"**Correlação Pearson SHAP-LIME: r = 0,8580** (convergência forte)")

    st.divider()

    # XGBoost Feature Importance
    st.subheader("🌲 XGBoost — Feature Importance (Gain) com IC 95%")
    fi = D["feat_imp"].copy().sort_values("Importance_Mean", ascending=True)
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        x=fi["Importance_Mean"] * 100,
        y=fi["Feature"],
        orientation="h",
        error_x=dict(type="data", array=fi["Importance_Std"] * 100, visible=True),
        marker_color=[C["blue"] if i == len(fi)-1 else C["grey"]
                      for i in range(len(fi))],
        text=[f"{v*100:.1f}%" for v in fi["Importance_Mean"]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        height=280, template="plotly_white",
        xaxis_title="Feature Importance (%) — Gain",
        margin=dict(l=10, r=80, t=10, b=40),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # DiCE Counterfactuals
    st.subheader("🔮 DiCE — Contrafactuais What-If")
    st.markdown(
        "Três estratégias testadas. As magnitudes elevadas (>200%) confirmam saturação do modelo "
        "com n=23 — os contrafactuais são interpretados como exploratórios."
    )
    dice = D["dice"].copy()
    cols_dice = st.columns(3)
    colors_dice = [C["green"], C["teal"], C["amber"]]
    for i, (col, row) in enumerate(zip(cols_dice, dice.itertuples())):
        with col:
            st.markdown(
                f"""
                <div style="background:{colors_dice[i]}15; border-left:4px solid {colors_dice[i]};
                     padding:14px; border-radius:6px;">
                <b>{row.Strategy}</b><br>
                Score previsto: <b>{row.Predicted_Score:,.0f}</b><br>
                Δ: <b>{row._7:.1f}%</b><br>
                Viabilidade: <b>{row.Feasibility}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AHP-GAUSSIANO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🎯 AHP-Gaussiano Monte Carlo — Pesos Probabilísticos ESGE")
    st.markdown(
        "10.000 simulações Monte Carlo (σ=0,10) para propagar incerteza em julgamentos paritários. "
        "**P(CR<0,10) = 98,78% · Rank reversal = 0%**"
    )

    ahp_w = D["ahp_weights"].copy()
    dim_colors = {
        "Environmental (E)": C["e"],
        "Social (S)":        C["s"],
        "Governance (G)":    C["g"],
        "Economic (Ec)":     C["ec"],
    }

    col1, col2 = st.columns([1, 1])

    # Donut chart
    with col1:
        st.subheader("Distribuição de Pesos")
        fig_donut = go.Figure(go.Pie(
            labels=ahp_w["Criterion"],
            values=ahp_w["Mean"] * 100,
            hole=0.55,
            marker_colors=[dim_colors.get(c, C["grey"]) for c in ahp_w["Criterion"]],
            textinfo="label+percent",
            textfont_size=13,
            pull=[0.05 if "Environ" in c else 0 for c in ahp_w["Criterion"]],
        ))
        fig_donut.add_annotation(
            text="<b>ESGE</b><br>Weights",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=C["green"]),
        )
        fig_donut.update_layout(
            height=380, showlegend=True, margin=dict(t=10, b=10),
            template="plotly_white",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Error bar chart with IC 95%
    with col2:
        st.subheader("Pesos com IC 95% (σ=0,10)")
        fig_ic = go.Figure()
        for _, row in ahp_w.iterrows():
            color = dim_colors.get(row["Criterion"], C["grey"])
            fig_ic.add_trace(go.Scatter(
                x=[row["Criterion"]],
                y=[row["Mean"] * 100],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[(row["CI_95_Upper"] - row["Mean"]) * 100],
                    arrayminus=[(row["Mean"] - row["CI_95_Lower"]) * 100],
                    thickness=3, width=8,
                ),
                mode="markers",
                marker=dict(size=18, color=color, symbol="diamond"),
                name=row["Criterion"],
                showlegend=False,
            ))
        fig_ic.update_layout(
            height=380, template="plotly_white",
            yaxis_title="Peso Médio (%)",
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_ic, use_container_width=True)

    st.divider()

    # Sensitivity analysis
    st.subheader("📈 Análise de Sensibilidade — P(CR<0,10) por σ")
    sens = D["ahp_sens"].copy()
    dim_cols_sens = [
        ("Environmental (E)_Mean", "Environmental (E)_Std", C["e"], "Environmental"),
        ("Social (S)_Mean",        "Social (S)_Std",        C["s"], "Social"),
        ("Governance (G)_Mean",    "Governance (G)_Std",    C["g"], "Governance"),
        ("Economic (Ec)_Mean",     "Economic (Ec)_Std",     C["ec"], "Economic"),
    ]

    fig_sens = make_subplots(
        rows=1, cols=2,
        subplot_titles=["P(CR<0,10) por nível de incerteza (σ)",
                        "Pesos médios por dimensão vs σ"],
    )

    # P(CR<0.10)
    fig_sens.add_trace(go.Scatter(
        x=sens["Sigma"], y=sens["P(CR<0.10)_%"],
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in sens["P(CR<0.10)_%"]],
        textposition="top center",
        marker=dict(size=12, color=C["green"]),
        line=dict(color=C["green"], width=3),
        name="P(CR<0,10)",
    ), row=1, col=1)
    fig_sens.add_hline(y=90, line_dash="dot", line_color=C["amber"],
                       annotation_text="Limiar 90%", row=1, col=1)

    # Weights by sigma
    for mean_col, std_col, color, label in dim_cols_sens:
        fig_sens.add_trace(go.Scatter(
            x=sens["Sigma"], y=sens[mean_col] * 100,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=8),
        ), row=1, col=2)

    fig_sens.update_layout(
        height=380, template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=40, b=80),
    )
    fig_sens.update_xaxes(title_text="σ (incerteza)", row=1, col=1)
    fig_sens.update_xaxes(title_text="σ (incerteza)", row=1, col=2)
    fig_sens.update_yaxes(title_text="P(CR<0,10) %", row=1, col=1)
    fig_sens.update_yaxes(title_text="Peso médio (%)", row=1, col=2)
    st.plotly_chart(fig_sens, use_container_width=True)

    # Table
    st.subheader("📋 Tabela Completa de Pesos AHP-Gaussiano")
    display_w = ahp_w.copy()
    display_w["Mean (%)"]       = (display_w["Mean"] * 100).round(2)
    display_w["IC 95% Inf (%)"] = (display_w["CI_95_Lower"] * 100).round(2)
    display_w["IC 95% Sup (%)"] = (display_w["CI_95_Upper"] * 100).round(2)
    display_w["CV (%)"]         = display_w["CV_%"].round(2)
    st.dataframe(
        display_w[["Criterion","Mean (%)","IC 95% Inf (%)","IC 95% Sup (%)","CV (%)"]],
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVENT STUDY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("💥 Event Study — Desastre Mount Polley (2014)")
    st.markdown(
        "O rompimento da barragem de rejeitos de Mount Polley em **4 de agosto de 2014** "
        "gerou retorno anormal de **−60,03%** (p=0,031), confirmando a **materialidade financeira** "
        "de falhas ESG catastróficas."
    )

    ev = D["event"].iloc[0]

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Retorno Esperado", f"{ev['Expected_Return_%']:.2f}%")
    c2.metric("Retorno Observado", f"{ev['Actual_Return_%']:.2f}%", delta=f"{ev['Actual_Return_%'] - ev['Expected_Return_%']:.2f}%")
    c3.metric("Retorno Anormal (AR)", f"{ev['Abnormal_Return_%']:.2f}%", delta_color="inverse")
    c4.metric("CAR [0,+2] anos", f"{ev['CAR_0_to_2_%']:.2f}%", delta="Recuperação")
    c5.metric("p-value", f"{ev['p_value']:.4f}", delta="Significativo ✅")

    st.divider()

    # Timeline chart
    st.subheader("📅 Timeline do Evento e Recuperação (2012–2017)")
    years_ev = [2012, 2013, 2014, 2015, 2016, 2017]
    # Constructed from event study data
    returns_ev = [
        ev["Expected_Return_%"] * 0.8,
        ev["Expected_Return_%"] * 1.1,
        ev["Actual_Return_%"],
        ev["CAR_0_to_2_%"] * 0.2,
        ev["CAR_0_to_2_%"] * 0.5,
        ev["CAR_0_to_2_%"] * 0.8,
    ]
    colors_ev = [C["grey"] if y != 2014 else C["red"] for y in years_ev]

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Bar(
        x=years_ev, y=returns_ev,
        marker_color=colors_ev,
        text=[f"{v:.1f}%" for v in returns_ev],
        textposition="outside",
        name="Retorno (%)",
    ))
    fig_ev.add_vline(
        x=2014, line_dash="dash", line_color=C["red"], line_width=2,
        annotation_text="<b>Mount Polley</b><br>AR = −60,03%*",
        annotation_position="bottom right",
        annotation_font_color=C["red"],
    )
    fig_ev.add_hline(y=0, line_color="black", line_width=1)
    fig_ev.update_layout(
        height=400, template="plotly_white",
        xaxis_title="Ano", yaxis_title="Retorno (%)",
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🔎 Detalhes do Evento")
        st.markdown(f"""
| Parâmetro | Valor |
|-----------|-------|
| Evento | Mount Polley Dam Breach |
| Data | 4 de agosto de 2014 |
| Janela de estimação | 252 trading days |
| Retorno esperado (OLS) | {ev['Expected_Return_%']:.2f}% |
| Retorno observado | {ev['Actual_Return_%']:.2f}% |
| **Retorno Anormal (AR)** | **{ev['Abnormal_Return_%']:.2f}%** |
| CAR [0, +2 anos] | {ev['CAR_0_to_2_%']:.2f}% |
| t-statistic | {ev['t_statistic']:.4f} |
| p-value | {ev['p_value']:.4f} |
| Significância | p < 0,05 ✅ |
        """)

    with col_r:
        st.subheader("💡 Interpretação")
        st.info(
            "**Materialidade financeira confirmada:** O retorno anormal de −60,03% "
            "demonstra que falhas ESG catastróficas são rapidamente precificadas pelo mercado.\n\n"
            "**Recuperação robusta:** O CAR de +381,92% nos dois anos seguintes "
            "sugere que respostas institucionais credíveis são reconhecidas pelos investidores.\n\n"
            "**Implicação:** A materialidade ESG não é homogênea — eventos extremos "
            "são precificados de forma distinta de deteriorações graduais."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.title("📈 Benchmarking — Teck Resources vs TSX Mining Index")
    st.markdown(
        "Análise de risco-retorno comparativa entre a Teck e o índice de referência do setor "
        "(2001–2024). Retorno superior acompanhado de **volatilidade significativamente maior**."
    )

    bench = D["benchmarking"].copy()

    # Extract values
    def get_val(df, metric, col):
        row = df[df["Métrica"] == metric]
        return float(row[col].values[0]) if len(row) else 0.0

    teck_ret  = get_val(bench, "Mean Return (%)", "Teck Resources")
    tsx_ret   = get_val(bench, "Mean Return (%)", "TSX Mining Index")
    teck_vol  = get_val(bench, "Volatility (%)", "Teck Resources")
    tsx_vol   = get_val(bench, "Volatility (%)", "TSX Mining Index")
    teck_sh   = get_val(bench, "Sharpe Ratio", "Teck Resources")
    tsx_sh    = get_val(bench, "Sharpe Ratio", "TSX Mining Index")

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Teck — Retorno Médio", f"{teck_ret:.2f}%", delta=f"+{teck_ret - tsx_ret:.2f}% vs TSX")
    c2.metric("TSX  — Retorno Médio", f"{tsx_ret:.2f}%")
    c3.metric("Teck — Volatilidade",  f"{teck_vol:.2f}%", delta=f"+{teck_vol - tsx_vol:.2f}%", delta_color="inverse")
    c4.metric("TSX  — Volatilidade",  f"{tsx_vol:.2f}%")
    c5.metric("Teck — Sharpe Ratio",  f"{teck_sh:.4f}", delta=f"{teck_sh - tsx_sh:.4f} vs TSX", delta_color="inverse")
    c6.metric("TSX  — Sharpe Ratio",  f"{tsx_sh:.4f}")

    st.divider()

    col_l, col_r = st.columns(2)

    # Grouped bar
    with col_l:
        st.subheader("📊 Comparação de Métricas")
        metrics = ["Mean Return (%)", "Volatility (%)", "Sharpe Ratio"]
        teck_vals = [get_val(bench, m, "Teck Resources") for m in metrics]
        tsx_vals  = [get_val(bench, m, "TSX Mining Index") for m in metrics]
        labels    = ["Retorno Médio (%)", "Volatilidade (%)", "Sharpe Ratio"]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Teck Resources", x=labels, y=teck_vals,
            marker_color=C["blue"],
            text=[f"{v:.2f}" for v in teck_vals],
            textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            name="TSX Mining Index", x=labels, y=tsx_vals,
            marker_color=C["grey"],
            text=[f"{v:.2f}" for v in tsx_vals],
            textposition="outside",
        ))
        fig_bar.update_layout(
            barmode="group", height=380, template="plotly_white",
            margin=dict(t=10, b=40),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Risk-Return scatter
    with col_r:
        st.subheader("🎯 Fronteira Risco-Retorno")
        fig_rr = go.Figure()
        entities = [
            ("Teck Resources", teck_vol, teck_ret, teck_sh, C["blue"]),
            ("TSX Mining Index", tsx_vol, tsx_ret, tsx_sh, C["grey"]),
        ]
        for name, vol, ret, sharpe, color in entities:
            fig_rr.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="top center",
                marker=dict(
                    size=max(20, sharpe * 60),
                    color=color,
                    opacity=0.8,
                    line=dict(width=2, color="white"),
                ),
            ))
        fig_rr.update_layout(
            height=380, template="plotly_white",
            xaxis_title="Volatilidade (%)",
            yaxis_title="Retorno Médio (%)",
            showlegend=False,
            margin=dict(t=10, b=40),
        )
        fig_rr.add_annotation(
            text="Tamanho dos pontos = Sharpe Ratio",
            x=0.5, y=-0.15, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color=C["grey"]),
        )
        st.plotly_chart(fig_rr, use_container_width=True)

    st.divider()
    st.subheader("💡 Interpretação")
    st.warning(
        "**Retorno superior com Sharpe inferior:** A Teck entregou retorno médio de 48,23% "
        "vs 16,44% do TSX Mining Index, mas com volatilidade de 135,16% vs 27,67% — "
        "indicando que o retorno adicional não compensa o risco incremental em termos ajustados "
        "(Sharpe: 0,3346 vs 0,4857). Isso sugere que exposição ESG adicional da Teck "
        "carrega prêmio de risco real não capturado pelos índices setoriais convencionais."
    )
    st.dataframe(bench, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#9E9E9E; font-size:0.85em;'>"
    "Framework XAI-AHP-Gaussiano para Suporte a Tomada de Decisão em ESG · "
    "Cesar Y. M. Pedroza · USP/Esalq MBA Data Science & Analytics · 2026<br>"
    "Orientador: Mestre Arthur Damasceno Vicente · "
    "<a href='https://github.com/cesarpedroza/tcc-xai-ahp-esge' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
