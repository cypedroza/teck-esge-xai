"""
ESGE Decision Support — Teck Resources (2001–2024)
===================================================

Bilingual (PT/EN) dashboard for the XAI-AHP-Gaussian framework.

Every number shown here is read from the result tables produced by the
pipeline scripts (01–05). The app looks for them in ``outputs/tables`` (local
run) and falls back to the snapshot in ``app/results`` (see its README), so it
also runs on Streamlit Community Cloud. Nothing is simulated or typed in by hand.

Run locally:  streamlit run app/streamlit_app.py
Author: Cesar Yoshio Machado Pedroza · Advisor: Arthur Damasceno Vicente
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import config  # noqa: E402

RESULTS_SNAPSHOT = Path(__file__).resolve().parent / "results"
ARTICLE_URL = "https://doi.org/10.22167/2675-6528-202602779"
REPO_URL = "https://github.com/cypedroza/teck-esge-xai"
LINKEDIN_URL = "https://www.linkedin.com/in/cesar-y-m-pedroza"

st.set_page_config(
    page_title="ESGE Decision Support | Teck Resources",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "green": "#2E7D32", "teal": "#00695C", "blue": "#1565C0", "amber": "#F57F17",
    "red": "#C62828", "grey": "#546E7A",
    "Environmental (E)": "#2E7D32", "Social (S)": "#1565C0",
    "Governance (G)": "#6A1B9A", "Economic (Ec)": "#E65100",
}

# ─────────────────────────────────────────────────────────────────────────────
# TEXT (PT / EN)
# ─────────────────────────────────────────────────────────────────────────────
T = {
    "pt": {
        "tabs": ["🏠 Visão geral", "🔍 IA explicável", "🎯 AHP-Gaussiano",
                 "💥 Mount Polley", "📈 Benchmarking"],
        "side_title": "Framework XAI-AHP-Gaussiano",
        "side_desc": "Apoio à decisão em ESG no setor de mineração, combinando "
                     "machine learning explicável e AHP probabilístico.",
        "side_company": "**Empresa:** Teck Resources (TSX: TECK-B)",
        "side_period": "**Período:** 2001–2024 (24 relatórios anuais)",
        "side_key": "Resultados-chave",
        "side_links": "Links",
        "article": "Artigo (Revista E&S, 2026)", "code": "Código no GitHub",
        "side_footer": "Cesar Y. M. Pedroza · MBA em Data Science e Analytics, USP/Esalq · "
                       "Orientador: Arthur Damasceno Vicente",
        # overview
        "ov_title": "Framework XAI-AHP-Gaussiano — Apoio à Decisão ESGE",
        "ov_intro": "Análise de **24 anos de divulgação ESG da Teck Resources** (2001–2024). "
                    "O framework identifica mudanças no padrão de divulgação, explica quais "
                    "fatores estão associados a ele e pondera as dimensões ESG levando em "
                    "conta a incerteza dos julgamentos.",
        "k_reports": "Relatórios analisados", "k_reports_sub": "anuais e de sustentabilidade",
        "k_break": "Quebra em 2018", "k_break_sub": "divulgação ~9× maior",
        "k_env": "Peso Ambiental", "k_env_sub": "AHP-Gaussiano",
        "k_r2": "R² leave-one-out", "k_r2_sub": "Random Forest",
        "k_car": "Mount Polley (CAR)", "k_car_sub": "sem efeito significativo",
        "ov_chart": "📉 Divulgação ESG ao longo do tempo — quebra estrutural em 2018",
        "ov_y": "ESG Disclosure Index (menções)", "ov_x": "Ano",
        "pre": "Antes do TCFD (2001–2017)", "post": "Depois do TCFD (2018–2024)",
        "tcfd": "Adoção do TCFD", "mean": "média",
        "ov_caption": "Valores reais por ano (contagem de termos ESG nos relatórios). "
                      "Teste de Chow: F = {F} (p < 0,0001). A média passou de {pre} para {post} "
                      "menções por relatório; o salto aparece a partir do relatório de 2019. O "
                      "índice mede o volume de divulgação, não o desempenho ESG.",
        "ov_arch": "🏗️ Como o framework funciona",
        "mods": [
            ("1️⃣ Dados", "Text mining de 24 relatórios + dados de mercado (Yahoo Finance)"),
            ("2️⃣ Quebra estrutural", "Teste de Chow em 2018 (adoção do TCFD)"),
            ("3️⃣ ML + XAI", "Random Forest e XGBoost explicados com SHAP, LIME e DiCE"),
            ("4️⃣ AHP-Gaussiano", "10.000 simulações Monte Carlo para os pesos ESGE"),
            ("5️⃣ Mercado", "Benchmark contra o S&P/TSX e estudo de evento"),
        ],
        "ov_takeaways": "💡 Principais conclusões",
        "takeaways": [
            "A divulgação ESG da Teck mudou de patamar em 2018, junto com a adoção do TCFD.",
            "A extensão do relatório e o preço da ação são os fatores mais associados ao "
            "nível de divulgação (leitura descritiva, n = 23).",
            "A dimensão Ambiental recebe o maior peso, de forma estável mesmo com incerteza.",
            "O desastre de Mount Polley (2014, Imperial Metals) não teve efeito mensurável "
            "sobre a ação da Teck.",
        ],
        # xai
        "xai_title": "🔍 IA explicável — o que explica o nível de divulgação ESG?",
        "xai_intro": "Modelos treinados para explicar log(1 + ESG Disclosure Index) a partir de "
                     "quatro variáveis. Com 23 anos de dados, as importâncias são associações "
                     "descritivas, não relações de causa e efeito.",
        "xai_perf": "📏 Desempenho fora da amostra (leave-one-out)",
        "xai_perf_y": "R² leave-one-out",
        "xai_perf_cap": "Cada ano é previsto por um modelo treinado sem ele. Valores abaixo de "
                        "zero indicam desempenho pior que prever a média. Na validação temporal, "
                        "todos os modelos falham após a quebra de 2018: o módulo explica, mas não "
                        "prevê.",
        "xai_shap_lime": "🔄 SHAP × LIME — participação de cada variável",
        "xai_share": "Participação na importância total (%)",
        "xai_conv": "Os dois métodos concordam: Pearson r = {r}, Spearman ρ = {rho} "
                    "(4 variáveis, indicador descritivo).",
        "xai_perm": "🌲 Importância por permutação (XGBoost)",
        "xai_perm_x": "Participação (%)",
        "xai_dice": "🔮 DiCE — cenários contrafactuais para 2024",
        "xai_dice_intro": "Partindo de um índice previsto de **{q}**, os contrafactuais que levam "
                          "a previsão para cerca de **{t}** pedem um relatório ~22% mais curto. "
                          "O resultado é contraintuitivo e mostra uma limitação do modelo com "
                          "poucos dados, não uma recomendação de gestão.",
        "cf": "Cenário", "changed": "Variáveis alteradas",
        # ahp
        "ahp_title": "🎯 AHP-Gaussiano — pesos das dimensões ESGE",
        "ahp_intro": "Os julgamentos par a par recebem ruído gaussiano e são simulados "
                     "**10.000 vezes** (σ = 0,10). Assim, os pesos vêm com intervalo de "
                     "confiança. Consistência: **P(CR < 0,10) = {p}%**.",
        "ahp_donut": "Distribuição dos pesos", "ahp_ci": "Pesos com intervalo de 95%",
        "ahp_w": "Peso médio (%)",
        "ahp_sens": "📈 Sensibilidade à incerteza (σ)",
        "ahp_sens_a": "Matrizes consistentes, P(CR < 0,10)", "ahp_sens_b": "Peso médio por dimensão",
        "sigma": "σ (incerteza)",
        "ahp_sens_cap": "A ordem das dimensões quase não muda: inversão de ranking de "
                        "{rr0}% em σ = 0,10 e {rr2}% em σ = 0,20.",
        "ahp_table": "📋 Tabela de pesos",
        "dims": {"Environmental (E)": "Ambiental (E)", "Social (S)": "Social (S)",
                 "Governance (G)": "Governança (G)", "Economic (Ec)": "Econômica (Ec)"},
        "col_dim": "Dimensão", "col_mean": "Média (%)", "col_lo": "IC 95% inf. (%)",
        "col_hi": "IC 95% sup. (%)", "col_cv": "CV (%)",
        # event
        "ev_title": "💥 Mount Polley (2014) — o mercado puniu a Teck?",
        "ev_intro": "Em **4 de agosto de 2014** rompeu a barragem de rejeitos de Mount Polley, "
                    "operação da **Imperial Metals** (não da Teck), na mesma província. O "
                    "estudo de evento verifica se a ação da Teck teve retorno anormal nos dias "
                    "seguintes.",
        "ev_ar0": "Retorno anormal (dia 0)", "ev_car": "CAR [0, +2]",
        "ev_p": "p-valor", "ev_beta": "Beta (estimação)",
        "ev_ns": "não significativo",
        "ev_chart": "📅 Retornos anormais diários e acumulados",
        "ev_ar": "Retorno anormal diário", "ev_carl": "Retorno anormal acumulado",
        "ev_day": "Pregão (dia 0 = 05/08/2014)", "ev_pct": "%",
        "ev_method": "🔎 Método",
        "ev_method_txt": "Modelo de mercado diário (OLS). Janela de estimação: 250 pregões "
                         "(do 260º ao 11º antes do evento). Janela do evento: dia 0 a dia +2. "
                         "A bolsa de Toronto não abriu em 04/08/2014 (feriado), então o dia 0 "
                         "é 05/08/2014.",
        "ev_robust": "🧪 Verificação com outro índice",
        "ev_market": "Mercado", "ev_conc": "💡 Conclusão",
        "ev_conc_txt": "Não há evidência de retorno anormal da Teck associado ao desastre "
                       "(CAR de {car}%, p = {p}). Com o mercado como referência, parte de um "
                       "efeito setorial já fica absorvida, e a janela de três pregões capta só a "
                       "reação imediata.",
        # bench
        "bm_title": "📈 Benchmarking — Teck Resources × S&P/TSX Composite",
        "bm_intro": "Retornos anuais de **2001 a 2024** com séries reais do Yahoo Finance. O "
                    "S&P/TSX Composite não inclui dividendos, então a comparação principal usa "
                    "retornos de preço para os dois. A verificação com retornos totais usa o ETF "
                    "XIC.TO (2002–2024).",
        "bm_ret": "Retorno médio anual", "bm_geo": "Retorno composto (geom.)",
        "bm_vol": "Volatilidade anual", "bm_sharpe": "Sharpe",
        "bm_growth": "📈 Crescimento de 1 dólar canadense (escala log)",
        "bm_growth_y": "Valor acumulado (fim de 2000 = 1)",
        "bm_rr": "🎯 Risco × retorno",
        "bm_rr_x": "Volatilidade anual (%)", "bm_rr_y": "Retorno médio anual (%)",
        "bm_table": "📋 Métricas (retornos de preço)",
        "bm_check": "🧪 Verificação com retornos totais (com dividendos, 2002–2024)",
        "bm_conc": "💡 Interpretação",
        "bm_conc_txt": "A Teck teve retorno médio bem maior que o índice, com volatilidade quase "
                       "nove vezes maior, e a diferença de médias não é estatisticamente "
                       "significativa (teste t pareado, p = {p}). Em retornos de preço o Sharpe "
                       "da Teck é maior ({s1} × {s2}); com dividendos a relação se inverte "
                       "({s3} × {s4}). Por isso não dá para afirmar que o retorno extra "
                       "compensou o risco. Beta: {b1} com retornos anuais e {b2} com diários.",
        "metric": "Métrica",
        "mnames": {"mean_return": "Retorno médio (%)", "geometric_mean_return": "Retorno composto (%)",
                   "volatility": "Volatilidade (%)", "sharpe": "Sharpe", "sortino": "Sortino",
                   "VaR_95": "VaR 95% (%)", "CVaR_95": "CVaR 95% (%)",
                   "max_drawdown": "Queda máxima (%)", "beta": "Beta (anual)",
                   "beta_daily": "Beta (diário)"},
        "feat": {"Report_Quality_Score": "Extensão do relatório", "close_price": "Preço da ação",
                 "Market_Liquidity": "Liquidez de mercado", "Annual_Return_Pct": "Retorno anual"},
        "models": {"Baseline (mean)": "Média (referência)", "Linear regression": "Regressão linear",
                   "Random Forest": "Random Forest", "XGBoost": "XGBoost"},
        "footer": "Todos os números vêm das tabelas geradas pelo código do repositório. "
                  "Dados de mercado: Yahoo Finance.",
    },
    "en": {
        "tabs": ["🏠 Overview", "🔍 Explainable AI", "🎯 Gaussian AHP",
                 "💥 Mount Polley", "📈 Benchmarking"],
        "side_title": "XAI-AHP-Gaussian Framework",
        "side_desc": "ESG decision support for mining, combining explainable machine learning "
                     "and probabilistic AHP.",
        "side_company": "**Company:** Teck Resources (TSX: TECK-B)",
        "side_period": "**Period:** 2001–2024 (24 annual reports)",
        "side_key": "Key results",
        "side_links": "Links",
        "article": "Article (Revista E&S, 2026)", "code": "Code on GitHub",
        "side_footer": "Cesar Y. M. Pedroza · MBA in Data Science and Analytics, USP/Esalq · "
                       "Advisor: Arthur Damasceno Vicente",
        "ov_title": "XAI-AHP-Gaussian Framework — ESGE Decision Support",
        "ov_intro": "Analysis of **24 years of ESG disclosure at Teck Resources** (2001–2024). "
                    "The framework detects shifts in disclosure, explains which factors are "
                    "associated with it, and weights the ESG dimensions while accounting for "
                    "uncertainty in the judgments.",
        "k_reports": "Reports analysed", "k_reports_sub": "annual and sustainability",
        "k_break": "2018 break", "k_break_sub": "disclosure ~9× higher",
        "k_env": "Environmental weight", "k_env_sub": "Gaussian AHP",
        "k_r2": "Leave-one-out R²", "k_r2_sub": "Random Forest",
        "k_car": "Mount Polley (CAR)", "k_car_sub": "no significant effect",
        "ov_chart": "📉 ESG disclosure over time — structural break in 2018",
        "ov_y": "ESG Disclosure Index (mentions)", "ov_x": "Year",
        "pre": "Before TCFD (2001–2017)", "post": "After TCFD (2018–2024)",
        "tcfd": "TCFD adoption", "mean": "mean",
        "ov_caption": "Actual yearly values (count of ESG terms in the reports). Chow test: "
                      "F = {F} (p < 0.0001). The mean rose from {pre} to {post} mentions per "
                      "report; the jump shows from the 2019 report onward. The index measures the "
                      "volume of disclosure, not ESG performance.",
        "ov_arch": "🏗️ How the framework works",
        "mods": [
            ("1️⃣ Data", "Text mining of 24 reports + market data (Yahoo Finance)"),
            ("2️⃣ Structural break", "Chow test at 2018 (TCFD adoption)"),
            ("3️⃣ ML + XAI", "Random Forest and XGBoost explained with SHAP, LIME and DiCE"),
            ("4️⃣ Gaussian AHP", "10,000 Monte Carlo runs for the ESGE weights"),
            ("5️⃣ Market", "Benchmark against the S&P/TSX and an event study"),
        ],
        "ov_takeaways": "💡 Key takeaways",
        "takeaways": [
            "Teck's ESG disclosure shifted to a new level in 2018, when it adopted TCFD.",
            "Report length and share price are the factors most associated with the level of "
            "disclosure (descriptive reading, n = 23).",
            "The Environmental dimension gets the largest weight, and it stays stable under "
            "uncertainty.",
            "The Mount Polley disaster (2014, Imperial Metals) had no measurable effect on "
            "Teck's stock.",
        ],
        "xai_title": "🔍 Explainable AI — what explains the level of ESG disclosure?",
        "xai_intro": "Models trained to explain log(1 + ESG Disclosure Index) from four "
                     "variables. With 23 years of data, importances are descriptive "
                     "associations, not cause and effect.",
        "xai_perf": "📏 Out-of-sample performance (leave-one-out)",
        "xai_perf_y": "Leave-one-out R²",
        "xai_perf_cap": "Each year is predicted by a model trained without it. Values below zero "
                        "mean worse than predicting the mean. In time-ordered validation every "
                        "model fails after the 2018 break: the module explains, it does not "
                        "forecast.",
        "xai_shap_lime": "🔄 SHAP vs LIME — share of each variable",
        "xai_share": "Share of total importance (%)",
        "xai_conv": "Both methods agree: Pearson r = {r}, Spearman ρ = {rho} "
                    "(4 variables, descriptive indicator).",
        "xai_perm": "🌲 Permutation importance (XGBoost)",
        "xai_perm_x": "Share (%)",
        "xai_dice": "🔮 DiCE — counterfactual scenarios for 2024",
        "xai_dice_intro": "Starting from a predicted index of **{q}**, the counterfactuals that "
                          "move the prediction to about **{t}** call for a ~22% shorter report. "
                          "The result is counterintuitive and shows a limitation of a model "
                          "trained on little data, not a management recommendation.",
        "cf": "Scenario", "changed": "Variables changed",
        "ahp_title": "🎯 Gaussian AHP — weights of the ESGE dimensions",
        "ahp_intro": "Pairwise judgments get Gaussian noise and are simulated **10,000 times** "
                     "(σ = 0.10), so the weights come with confidence intervals. Consistency: "
                     "**P(CR < 0.10) = {p}%**.",
        "ahp_donut": "Weight distribution", "ahp_ci": "Weights with 95% interval",
        "ahp_w": "Mean weight (%)",
        "ahp_sens": "📈 Sensitivity to uncertainty (σ)",
        "ahp_sens_a": "Consistent matrices, P(CR < 0.10)", "ahp_sens_b": "Mean weight by dimension",
        "sigma": "σ (uncertainty)",
        "ahp_sens_cap": "The ranking of the dimensions barely changes: rank reversal of {rr0}% "
                        "at σ = 0.10 and {rr2}% at σ = 0.20.",
        "ahp_table": "📋 Weights table",
        "dims": {"Environmental (E)": "Environmental (E)", "Social (S)": "Social (S)",
                 "Governance (G)": "Governance (G)", "Economic (Ec)": "Economic (Ec)"},
        "col_dim": "Dimension", "col_mean": "Mean (%)", "col_lo": "95% CI low (%)",
        "col_hi": "95% CI high (%)", "col_cv": "CV (%)",
        "ev_title": "💥 Mount Polley (2014) — did the market punish Teck?",
        "ev_intro": "On **4 August 2014** the Mount Polley tailings dam failed. It was an "
                    "**Imperial Metals** operation (not Teck's), in the same province. The event "
                    "study checks whether Teck's stock had an abnormal return in the days that "
                    "followed.",
        "ev_ar0": "Abnormal return (day 0)", "ev_car": "CAR [0, +2]",
        "ev_p": "p-value", "ev_beta": "Beta (estimation)",
        "ev_ns": "not significant",
        "ev_chart": "📅 Daily and cumulative abnormal returns",
        "ev_ar": "Daily abnormal return", "ev_carl": "Cumulative abnormal return",
        "ev_day": "Trading day (day 0 = 5 Aug 2014)", "ev_pct": "%",
        "ev_method": "🔎 Method",
        "ev_method_txt": "Daily market model (OLS). Estimation window: 250 trading days (the "
                         "260th to the 11th before the event). Event window: day 0 to day +2. "
                         "The Toronto exchange was closed on 4 Aug 2014 (holiday), so day 0 is "
                         "5 Aug 2014.",
        "ev_robust": "🧪 Check with another index",
        "ev_market": "Market", "ev_conc": "💡 Conclusion",
        "ev_conc_txt": "There is no evidence of an abnormal return for Teck around the disaster "
                       "(CAR of {car}%, p = {p}). Using the market as the reference absorbs part "
                       "of any sector-wide effect, and a three-day window captures only the "
                       "immediate reaction.",
        "bm_title": "📈 Benchmarking — Teck Resources vs S&P/TSX Composite",
        "bm_intro": "Annual returns from **2001 to 2024** using real Yahoo Finance series. The "
                    "S&P/TSX Composite excludes dividends, so the main comparison uses price "
                    "returns for both. The total-return check uses the XIC.TO ETF (2002–2024).",
        "bm_ret": "Mean annual return", "bm_geo": "Compound return (geom.)",
        "bm_vol": "Annual volatility", "bm_sharpe": "Sharpe",
        "bm_growth": "📈 Growth of 1 Canadian dollar (log scale)",
        "bm_growth_y": "Cumulative value (end of 2000 = 1)",
        "bm_rr": "🎯 Risk vs return",
        "bm_rr_x": "Annual volatility (%)", "bm_rr_y": "Mean annual return (%)",
        "bm_table": "📋 Metrics (price returns)",
        "bm_check": "🧪 Total-return check (with dividends, 2002–2024)",
        "bm_conc": "💡 Interpretation",
        "bm_conc_txt": "Teck's mean return was much higher than the index's, with almost nine "
                       "times the volatility, and the difference in means is not statistically "
                       "significant (paired t-test, p = {p}). On price returns Teck's Sharpe is "
                       "higher ({s1} vs {s2}); with dividends the order reverses ({s3} vs {s4}). "
                       "So it cannot be said that the extra return paid for the extra risk. "
                       "Beta: {b1} on annual returns and {b2} on daily returns.",
        "metric": "Metric",
        "mnames": {"mean_return": "Mean return (%)", "geometric_mean_return": "Compound return (%)",
                   "volatility": "Volatility (%)", "sharpe": "Sharpe", "sortino": "Sortino",
                   "VaR_95": "VaR 95% (%)", "CVaR_95": "CVaR 95% (%)",
                   "max_drawdown": "Max drawdown (%)", "beta": "Beta (annual)",
                   "beta_daily": "Beta (daily)"},
        "feat": {"Report_Quality_Score": "Report length", "close_price": "Share price",
                 "Market_Liquidity": "Market liquidity", "Annual_Return_Pct": "Annual return"},
        "models": {"Baseline (mean)": "Mean (baseline)", "Linear regression": "Linear regression",
                   "Random Forest": "Random Forest", "XGBoost": "XGBoost"},
        "footer": "Every number comes from the tables produced by the code in the repository. "
                  "Market data: Yahoo Finance.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def result_path(name: str) -> Path:
    local = config.OUTPUTS_TABLES / name
    return local if local.exists() else RESULTS_SNAPSHOT / name


@st.cache_data
def load_all() -> dict:
    files = {
        "chow": "chow_test_structural_break.csv",
        "perf": "model_performance.csv",
        "shap_lime": "shap_lime_comparison.csv",
        "conv": "shap_lime_convergence.csv",
        "perm": "permutation_importance_xgboost.csv",
        "dice": "dice_counterfactuals.csv",
        "ahp": "ahp_weights.csv",
        "ahp_sens": "ahp_sensitivity_analysis.csv",
        "event": "event_study_mount_polley.csv",
        "event_daily": "event_study_mount_polley_daily_ar.csv",
        "fin": "financial_metrics.csv",
        "fin_tr": "financial_metrics_total_return.csv",
        "tests": "teck_vs_index_mean_tests.csv",
        "tests_tr": "mean_tests_total_return.csv",
        "returns": "annual_returns_teck_vs_index.csv",
    }
    d = {k: pd.read_csv(result_path(v)) for k, v in files.items()}
    d["master"] = pd.read_csv(config.DATA_PROCESSED / config.MASTER_FILE)
    return d


try:
    D = load_all()
except FileNotFoundError as e:
    st.error(f"Result table not found: {e.filename}. Run the pipeline (run_pipeline.py) "
             "or restore app/results.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE + FORMATTING
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    lang_label = st.radio("🌐 Idioma / Language", ["Português", "English"],
                          horizontal=True, key="lang")
L = "pt" if lang_label == "Português" else "en"
t = T[L]


def num(x: float, d: int = 2) -> str:
    """Number with the decimal separator of the selected language."""
    s = f"{x:,.{d}f}"
    if L == "pt":
        s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return s


def pct(x: float, d: int = 2) -> str:
    return num(100 * x, d)


def fname(f: str) -> str:
    return t["feat"].get(f, f)


def card(col, label: str, value: str, note: str = "") -> None:
    """KPI card: value plus a neutral note (no up/down arrow)."""
    with col:
        st.metric(label, value)
        if note:
            st.caption(note)


chow = D["chow"].set_index("Feature")
esg_row = chow.loc["ESG_Disclosure_Index"]
perf = D["perf"].set_index("Model")
conv = D["conv"].iloc[0]
ahp = D["ahp"].set_index("Criterion")
sens = D["ahp_sens"]
ev = D["event"]
ev_main = ev.iloc[0]
ev_xic = ev.iloc[1]
fin = D["fin"].set_index("metric")
fin_tr = D["fin_tr"].set_index("metric")
tests = D["tests"].iloc[0]
teck_col, idx_col = fin.columns[0], fin.columns[1]
tr_teck_col, tr_idx_col = fin_tr.columns[0], fin_tr.columns[1]
p_cr_010 = float(sens.loc[sens["sigma"].round(2) == 0.10, "P(CR<0.1)_%"].iloc[0])

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🌿 {t['side_title']}")
    st.markdown(t["side_desc"])
    st.markdown(t["side_company"] + "\n\n" + t["side_period"])
    st.divider()
    st.markdown(f"**{t['side_key']}**")
    st.markdown(
        f"- Chow F (2018): **{num(esg_row['Chow_F'])}** (p < 0{',' if L == 'pt' else '.'}0001)\n"
        f"- R² LOO (Random Forest): **{num(perf.loc['Random Forest', 'R2_LOO'])}**\n"
        f"- SHAP × LIME: r = **{num(conv['pearson_r'])}**\n"
        f"- {t['dims']['Environmental (E)']}: **{pct(ahp.loc['Environmental (E)', 'Mean'])}%**\n"
        f"- Mount Polley CAR: **{pct(ev_main['car'])}%** (p = {num(ev_main['p_value_car'])})"
    )
    st.divider()
    st.markdown(f"**{t['side_links']}**")
    st.markdown(f"- [{t['article']}]({ARTICLE_URL})\n- [{t['code']}]({REPO_URL})\n"
                f"- [LinkedIn]({LINKEDIN_URL})")
    st.divider()
    st.caption(t["side_footer"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(t["tabs"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title(t["ov_title"])
    st.markdown(t["ov_intro"])

    c = st.columns(5)
    card(c[0], t["k_reports"], "24", t["k_reports_sub"])
    card(c[1], t["k_break"], f"F = {num(esg_row['Chow_F'])}", t["k_break_sub"])
    card(c[2], t["k_env"], f"{pct(ahp.loc['Environmental (E)', 'Mean'], 1)}%", t["k_env_sub"])
    card(c[3], t["k_r2"], num(perf.loc["Random Forest", "R2_LOO"]), t["k_r2_sub"])
    card(c[4], t["k_car"], f"{pct(ev_main['car'])}%", t["k_car_sub"])

    st.divider()
    st.subheader(t["ov_chart"])
    m = D["master"]
    pre, post = m[m["year"] < 2018], m[m["year"] >= 2018]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pre["year"], y=pre["esg_disclosure_score"], mode="lines+markers",
                             name=t["pre"], line=dict(color=C["grey"], width=2),
                             marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=post["year"], y=post["esg_disclosure_score"], mode="lines+markers",
                             name=t["post"], line=dict(color=C["green"], width=3),
                             marker=dict(size=9, symbol="diamond")))
    for seg, mean, color in [(pre, esg_row["Mean_Pre_TCFD"], C["grey"]),
                             (post, esg_row["Mean_Post_TCFD"], C["green"])]:
        fig.add_trace(go.Scatter(x=[seg["year"].min(), seg["year"].max()], y=[mean, mean],
                                 mode="lines", line=dict(color=color, dash="dot", width=1.5),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=2017.5, line_dash="dash", line_color=C["amber"], line_width=2,
                  annotation_text=t["tcfd"], annotation_position="top left",
                  annotation_font_color=C["amber"])
    fig.add_annotation(x=2009, y=esg_row["Mean_Pre_TCFD"] + 25,
                       text=f"{t['mean']} = {num(esg_row['Mean_Pre_TCFD'], 1)}",
                       showarrow=False, font=dict(color=C["grey"]))
    fig.add_annotation(x=2021, y=esg_row["Mean_Post_TCFD"] + 40,
                       text=f"{t['mean']} = {num(esg_row['Mean_Post_TCFD'], 1)}",
                       showarrow=False, font=dict(color=C["green"]))
    fig.update_layout(height=420, template="plotly_white", xaxis_title=t["ov_x"],
                      yaxis_title=t["ov_y"], legend=dict(orientation="h", y=-0.18),
                      margin=dict(t=30, b=60))
    st.plotly_chart(fig, width="stretch")
    st.caption(t["ov_caption"].format(F=num(esg_row["Chow_F"]),
                                      pre=num(esg_row["Mean_Pre_TCFD"], 1),
                                      post=num(esg_row["Mean_Post_TCFD"], 1)))

    st.divider()
    st.subheader(t["ov_arch"])
    for col, (title, desc) in zip(st.columns(5), t["mods"]):
        col.info(f"**{title}**\n\n{desc}")

    st.subheader(t["ov_takeaways"])
    st.markdown("\n".join(f"- {x}" for x in t["takeaways"]))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — XAI
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title(t["xai_title"])
    st.markdown(t["xai_intro"])

    st.subheader(t["xai_perf"])
    order = ["Baseline (mean)", "Linear regression", "XGBoost", "Random Forest"]
    pv = perf.loc[order, "R2_LOO"]
    fig = go.Figure(go.Bar(
        x=[t["models"][m] for m in order], y=pv,
        marker_color=[C["grey"], C["teal"], C["blue"], C["green"]],
        text=[num(v) for v in pv], textposition="outside"))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=340, template="plotly_white", yaxis_title=t["xai_perf_y"],
                      margin=dict(t=20, b=40), yaxis=dict(range=[min(pv.min() * 1.4, -0.2), 0.65]))
    st.plotly_chart(fig, width="stretch")
    st.caption(t["xai_perf_cap"])

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(t["xai_shap_lime"])
        sl = D["shap_lime"].sort_values("SHAP_share", ascending=False)
        labels = [fname(f) for f in sl["feature"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="SHAP", x=labels, y=100 * sl["SHAP_share"], marker_color=C["blue"],
                             text=[num(100 * v, 1) for v in sl["SHAP_share"]], textposition="outside"))
        fig.add_trace(go.Bar(name="LIME", x=labels, y=100 * sl["LIME_share"], marker_color=C["teal"],
                             text=[num(100 * v, 1) for v in sl["LIME_share"]], textposition="outside"))
        fig.update_layout(barmode="group", height=360, template="plotly_white",
                          yaxis_title=t["xai_share"], legend=dict(orientation="h", y=-0.25),
                          margin=dict(t=20, b=60))
        st.plotly_chart(fig, width="stretch")
        st.markdown(t["xai_conv"].format(r=num(conv["pearson_r"]), rho=num(conv["spearman_rho"])))
    with col_b:
        st.subheader(t["xai_perm"])
        pm = D["perm"].sort_values("share_%")
        fig = go.Figure(go.Bar(
            x=pm["share_%"], y=[fname(f) for f in pm["feature"]], orientation="h",
            marker_color=[C["green"] if i == len(pm) - 1 else C["grey"] for i in range(len(pm))],
            text=[num(v, 1) + "%" for v in pm["share_%"]], textposition="outside"))
        fig.update_layout(height=360, template="plotly_white", xaxis_title=t["xai_perm_x"],
                          margin=dict(l=10, r=60, t=20, b=40))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader(t["xai_dice"])
    dice = D["dice"]
    st.markdown(t["xai_dice_intro"].format(q=num(dice["query_predicted_index"].iloc[0], 1),
                                           t=num(dice["predicted_disclosure_index"].iloc[0], 0)))
    change_cols = [c for c in dice.columns if c.endswith("_change_%")]
    rows = []
    for i, r in dice.iterrows():
        changed = [f"{fname(c.replace('_change_%', ''))}: {num(r[c], 1)}%"
                   for c in change_cols if abs(r[c]) > 1e-9]
        rows.append({t["cf"]: i + 1, t["changed"]: " · ".join(changed)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — AHP
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title(t["ahp_title"])
    st.markdown(t["ahp_intro"].format(p=num(p_cr_010)))

    dims = list(ahp.index)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t["ahp_donut"])
        fig = go.Figure(go.Pie(
            labels=[t["dims"][d] for d in dims], values=100 * ahp["Mean"], hole=0.55,
            marker_colors=[C[d] for d in dims], sort=False, textinfo="text",
            text=[f"{t['dims'][d]}<br>{pct(ahp.loc[d, 'Mean'], 1)}%" for d in dims],
            hovertemplate="%{label}: %{text}<extra></extra>",
            pull=[0.05 if d.startswith("Environ") else 0 for d in dims]))
        fig.add_annotation(text="<b>ESGE</b>", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=C["green"]))
        fig.update_layout(height=380, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader(t["ahp_ci"])
        fig = go.Figure()
        for d in dims:
            r = ahp.loc[d]
            fig.add_trace(go.Scatter(
                x=[t["dims"][d]], y=[100 * r["Mean"]], mode="markers",
                marker=dict(size=16, color=C[d], symbol="diamond"), showlegend=False,
                error_y=dict(type="data", symmetric=False, thickness=3, width=8,
                             array=[100 * (r["CI_95_Upper"] - r["Mean"])],
                             arrayminus=[100 * (r["Mean"] - r["CI_95_Lower"])])))
        fig.update_layout(height=380, template="plotly_white", yaxis_title=t["ahp_w"],
                          margin=dict(t=10, b=40))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader(t["ahp_sens"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=[t["ahp_sens_a"], t["ahp_sens_b"]])
    fig.add_trace(go.Scatter(x=sens["sigma"], y=sens["P(CR<0.1)_%"], mode="lines+markers+text",
                             text=[num(v, 1) + "%" for v in sens["P(CR<0.1)_%"]],
                             textposition="top center", marker=dict(size=11, color=C["green"]),
                             line=dict(color=C["green"], width=3), showlegend=False), row=1, col=1)
    for d in dims:
        fig.add_trace(go.Scatter(x=sens["sigma"], y=100 * sens[f"{d}_mean"], mode="lines+markers",
                                 name=t["dims"][d], line=dict(color=C[d], width=2)), row=1, col=2)
    fig.update_xaxes(title_text=t["sigma"])
    fig.update_yaxes(title_text="%", row=1, col=1, range=[85, 102])
    fig.update_yaxes(title_text=t["ahp_w"], row=1, col=2)
    fig.update_layout(height=380, template="plotly_white", legend=dict(orientation="h", y=-0.25),
                      margin=dict(t=40, b=80))
    st.plotly_chart(fig, width="stretch")
    rr = sens.set_index(sens["sigma"].round(2))["rank_reversal_%"]
    st.caption(t["ahp_sens_cap"].format(rr0=num(rr.loc[0.10]), rr2=num(rr.loc[0.20])))

    st.subheader(t["ahp_table"])
    tab = pd.DataFrame({
        t["col_dim"]: [t["dims"][d] for d in dims],
        t["col_mean"]: [num(100 * ahp.loc[d, "Mean"]) for d in dims],
        t["col_lo"]: [num(100 * ahp.loc[d, "CI_95_Lower"]) for d in dims],
        t["col_hi"]: [num(100 * ahp.loc[d, "CI_95_Upper"]) for d in dims],
        t["col_cv"]: [num(ahp.loc[d, "CV_%"]) for d in dims],
    })
    st.dataframe(tab, width="stretch", hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVENT STUDY
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title(t["ev_title"])
    st.markdown(t["ev_intro"])

    c = st.columns(4)
    card(c[0], t["ev_ar0"], f"{pct(ev_main['abnormal_return_t0'])}%")
    card(c[1], t["ev_car"], f"{pct(ev_main['car'])}%", t["ev_ns"])
    card(c[2], t["ev_p"], num(ev_main["p_value_car"]), f"t = {num(ev_main['t_stat_car'])}")
    card(c[3], t["ev_beta"], num(ev_main["beta"]))

    st.divider()
    st.subheader(t["ev_chart"])
    dd = D["event_daily"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dd["relative_day"], y=100 * dd["abnormal_return"], name=t["ev_ar"],
                         marker_color=[C["green"] if v >= 0 else C["red"] for v in dd["abnormal_return"]],
                         text=[pct(v) + "%" for v in dd["abnormal_return"]], textposition="outside"))
    fig.add_trace(go.Scatter(x=dd["relative_day"], y=100 * dd["CAR"], name=t["ev_carl"],
                             mode="lines+markers", line=dict(color=C["blue"], width=3),
                             marker=dict(size=10)))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=380, template="plotly_white", xaxis_title=t["ev_day"],
                      yaxis_title=t["ev_pct"], xaxis=dict(tickmode="array", tickvals=[0, 1, 2]),
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=20, b=60))
    st.plotly_chart(fig, width="stretch")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(t["ev_method"])
        st.markdown(t["ev_method_txt"])
        st.subheader(t["ev_robust"])
        rob = pd.DataFrame({
            t["ev_market"]: ["S&P/TSX Composite (^GSPTSE)", "iShares S&P/TSX Capped (XIC.TO)"],
            "Beta": [num(ev_main["beta"]), num(ev_xic["beta"])],
            "CAR [0, +2] (%)": [pct(ev_main["car"]), pct(ev_xic["car"])],
            t["ev_p"]: [num(ev_main["p_value_car"]), num(ev_xic["p_value_car"])],
        })
        st.dataframe(rob, width="stretch", hide_index=True)
    with col_r:
        st.subheader(t["ev_conc"])
        st.info(t["ev_conc_txt"].format(car=pct(ev_main["car"]), p=num(ev_main["p_value_car"])))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — BENCHMARKING
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.title(t["bm_title"])
    st.markdown(t["bm_intro"])

    def f(m, col, frame=fin):
        return float(frame.loc[m, col])

    c = st.columns(4)
    card(c[0], f"{t['bm_ret']} · Teck", f"{pct(f('mean_return', teck_col))}%",
         f"S&P/TSX: {pct(f('mean_return', idx_col))}%")
    card(c[1], f"{t['bm_geo']} · Teck", f"{pct(f('geometric_mean_return', teck_col))}%",
         f"S&P/TSX: {pct(f('geometric_mean_return', idx_col))}%")
    card(c[2], f"{t['bm_vol']} · Teck", f"{pct(f('volatility', teck_col))}%",
         f"S&P/TSX: {pct(f('volatility', idx_col))}%")
    card(c[3], f"{t['bm_sharpe']} · Teck", num(f("sharpe", teck_col)),
         f"S&P/TSX: {num(f('sharpe', idx_col))}")

    st.divider()
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader(t["bm_growth"])
        r = D["returns"].set_index("year")
        growth = (1 + r).cumprod()
        growth = pd.concat([pd.DataFrame([[1.0, 1.0]], columns=r.columns,
                                         index=[r.index.min() - 1]), growth])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=growth.index, y=growth.iloc[:, 0], name="Teck Resources",
                                 mode="lines+markers", line=dict(color=C["blue"], width=3)))
        fig.add_trace(go.Scatter(x=growth.index, y=growth.iloc[:, 1], name="S&P/TSX Composite",
                                 mode="lines+markers", line=dict(color=C["grey"], width=2, dash="dash")))
        fig.add_vline(x=2014.6, line_dash="dot", line_color=C["red"],
                      annotation_text="Mount Polley", annotation_font_color=C["red"])
        fig.update_layout(height=400, template="plotly_white", yaxis_type="log",
                          yaxis_title=t["bm_growth_y"], legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=20, b=60))
        st.plotly_chart(fig, width="stretch")
    with col_r:
        st.subheader(t["bm_rr"])
        fig = go.Figure()
        for name, col, color in [("Teck", teck_col, C["blue"]), ("S&P/TSX", idx_col, C["grey"])]:
            fig.add_trace(go.Scatter(
                x=[100 * f("volatility", col)], y=[100 * f("mean_return", col)],
                mode="markers+text", text=[name], textposition="top center", showlegend=False,
                marker=dict(size=26, color=color, opacity=0.85, line=dict(width=2, color="white"))))
        vols = [100 * f("volatility", c_) for c_ in (teck_col, idx_col)]
        rets = [100 * f("mean_return", c_) for c_ in (teck_col, idx_col)]
        fig.update_layout(height=400, template="plotly_white", xaxis_title=t["bm_rr_x"],
                          yaxis_title=t["bm_rr_y"], margin=dict(t=20, b=40),
                          xaxis=dict(range=[0, max(vols) * 1.15]),
                          yaxis=dict(range=[min(0, min(rets)) - 5, max(rets) * 1.25]))
        st.plotly_chart(fig, width="stretch")

    def metric_table(frame, cols):
        keys = [k for k in t["mnames"] if k in frame.index]
        is_pct = {"mean_return", "geometric_mean_return", "volatility", "VaR_95", "CVaR_95",
                  "max_drawdown"}
        out = {t["metric"]: [t["mnames"][k] for k in keys]}
        for col in cols:
            out[col] = [pct(frame.loc[k, col]) if k in is_pct else num(frame.loc[k, col])
                        for k in keys]
        return pd.DataFrame(out)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(t["bm_table"])
        st.dataframe(metric_table(fin, [teck_col, idx_col]), width="stretch", hide_index=True)
    with col_b:
        st.subheader(t["bm_check"])
        st.dataframe(metric_table(fin_tr, [tr_teck_col, tr_idx_col]), width="stretch",
                     hide_index=True)

    st.subheader(t["bm_conc"])
    st.info(t["bm_conc_txt"].format(
        p=num(tests["paired_p"], 3), s1=num(f("sharpe", teck_col)), s2=num(f("sharpe", idx_col)),
        s3=num(f("sharpe", tr_teck_col, fin_tr)), s4=num(f("sharpe", tr_idx_col, fin_tr)),
        b1=num(f("beta", teck_col)), b2=num(f("beta_daily", teck_col))))

# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#9E9E9E;font-size:0.85em;'>"
    f"{t['footer']}<br>Cesar Yoshio Machado Pedroza · USP/Esalq · 2026 · "
    f"<a href='{REPO_URL}' target='_blank'>GitHub</a> · "
    f"<a href='{ARTICLE_URL}' target='_blank'>DOI 10.22167/2675-6528-202602779</a>"
    "</div>",
    unsafe_allow_html=True,
)
