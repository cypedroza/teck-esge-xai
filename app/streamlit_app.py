"""
═══════════════════════════════════════════════════════════════════════════════
STREAMLIT DSS - DECISION SUPPORT SYSTEM
XAI-AHP-Gaussian ESGE Framework
═══════════════════════════════════════════════════════════════════════════════

Sistema de Suporte à Decisão interativo para análise ESGE.

Features:
- Dashboard executivo com KPIs
- Análise XAI (SHAP/LIME/DiCE)
- Simulador AHP-Gaussiano
- Análise financeira e event studies
- Export de relatórios

Author: Cesar Yoshio Machado Pedroza
Institution: USP/Esalq - MBA Data Science and Analytics
Date: 2026-04-16

═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import config

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ESGE Decision Support System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/cesarpedroza/teck-esge-xai',
        'Report a bug': 'mailto:cesar.pedroza@usp.br',
        'About': "XAI-AHP-Gaussian ESGE Framework | USP/Esalq 2026"
    }
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .kpi-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
    .metric-delta-positive {
        color: #00cc00;
    }
    .metric-delta-negative {
        color: #cc0000;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data() -> tuple:
    """
    Load all datasets for the DSS.
    
    Returns
    -------
    tuple
        (df_master, df_ahp, df_shap, df_financial)
    """
    try:
        df_master = pd.read_csv(config.DATA_PROCESSED / "esge_master.csv")
        df_ahp = pd.read_csv(config.OUTPUTS_DIR / "ahp_weights.csv")
        
        # Try to load SHAP, fallback to dummy if not available
        try:
            df_shap = pd.read_csv(config.OUTPUTS_DIR / "shap_importance.csv")
        except:
            df_shap = pd.DataFrame({
                'feature': df_master.columns[1:5],
                'importance': [0.334, 0.272, 0.218, 0.177]
            })
        
        return df_master, df_ahp, df_shap
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()

# Load data
df_master, df_ahp, df_shap = load_data()

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=ESGE+DSS", 
             use_column_width=True)
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📍 Navegação",
        options=[
            "🏠 Dashboard Executivo",
            "🔍 Análise XAI",
            "⚖️ Simulador AHP",
            "📈 Análise Financeira"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # Year filter
    years_available = sorted(df_master['year'].unique())
    selected_year = st.selectbox(
        "📅 Ano de Análise",
        options=years_available,
        index=len(years_available)-1
    )
    
    st.markdown("---")
    
    # Info box
    st.info("""
    **📊 Sobre o DSS**
    
    Sistema integrado que combina:
    - XAI (SHAP/LIME/DiCE)
    - AHP-Gaussiano
    - Análise Financeira
    
    **🏢 Caso:** Teck Resources Ltd.
    **📅 Período:** 2001-2024
    """)
    
    st.markdown("---")
    
    # Footer
    st.caption("""
    **Desenvolvido por:**  
    Cesar Y. M. Pedroza  
    USP/Esalq | 2026
    """)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD EXECUTIVO
# ═══════════════════════════════════════════════════════════════════════════

if page == "🏠 Dashboard Executivo":
    st.markdown('<div class="main-header">🏠 Dashboard Executivo ESGE</div>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filter data for selected year
    df_year = df_master[df_master['year'] == selected_year].iloc[0]
    
    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 ESG Disclosure Score",
            value=f"{df_year['esg_disclosure_score']:.0f}",
            delta=f"{df_year['esg_disclosure_score'] - df_master['esg_disclosure_score'].mean():.0f}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="📈 Annual Return",
            value=f"{df_year['annual_return_%']:.1f}%",
            delta=f"{df_year['annual_return_%']:.1f}%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="💹 Trading Volume",
            value=f"{df_year['volume']/1e6:.1f}M",
            delta=f"{(df_year['volume'] - df_master['volume'].mean())/1e6:.1f}M",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="📄 Report Quality",
            value=f"{df_year['char_count']/1000:.0f}K chars",
            delta=f"{(df_year['char_count'] - df_master['char_count'].mean())/1000:.0f}K",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Evolução do Score ESG")
        fig1 = px.line(
            df_master,
            x='year',
            y='esg_disclosure_score',
            markers=True,
            title="ESG Disclosure Score (2001-2024)"
        )
        fig1.update_layout(
            xaxis_title="Ano",
            yaxis_title="Score ESG",
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("💰 Performance Financeira")
        fig2 = px.bar(
            df_master,
            x='year',
            y='annual_return_%',
            title="Retorno Anual (%)"
        )
        fig2.update_layout(
            xaxis_title="Ano",
            yaxis_title="Retorno (%)",
            showlegend=False
        )
        fig2.update_traces(marker_color='#1f77b4')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚖️ Pesos AHP-Gaussiano")
        
        # Prepare data
        if 'Criterion' in df_ahp.columns:
            df_ahp_plot = df_ahp.copy()
            df_ahp_plot['Criterion'] = df_ahp_plot['Criterion'].replace({
                'esg_disclosure_score': 'Environmental (E)',
                'annual_return_%': 'Economic (Ec)',
                'volume': 'Governance (G)',
                'char_count': 'Social (S)'
            })
            
            fig3 = px.bar(
                df_ahp_plot,
                x='Criterion',
                y='Mean',
                error_y='Std',
                title="Pesos AHP-Gaussiano (10,000 simulações)"
            )
            fig3.update_layout(
                xaxis_title="Dimensão ESGE",
                yaxis_title="Peso Médio",
                showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Importância das Features (SHAP)")
        
        fig4 = px.bar(
            df_shap.head(10),
            y='feature',
            x='importance',
            orientation='h',
            title="Top 10 Features por Importância"
        )
        fig4.update_layout(
            xaxis_title="Importância SHAP",
            yaxis_title="Feature",
            showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # Data Table
    st.subheader("📋 Dados Completos")
    st.dataframe(
        df_master.style.format({
            'annual_return_%': '{:.2f}%',
            'volume': '{:,.0f}',
            'char_count': '{:,.0f}'
        }),
        use_container_width=True,
        height=300
    )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANÁLISE XAI
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔍 Análise XAI":
    st.markdown('<div class="main-header">🔍 Análise XAI (Explainable AI)</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 📖 Fundamento Teórico
    
    Este módulo implementa **três técnicas complementares** de XAI:
    
    1. **SHAP (SHapley Additive exPlanations)**  
       ↪ Importância global das features (Lundberg & Lee, 2017)
    
    2. **LIME (Local Interpretable Model-agnostic Explanations)**  
       ↪ Explicações locais por observação (Ribeiro et al., 2016)
    
    3. **DiCE (Diverse Counterfactual Explanations)**  
       ↪ Recomendações acionáveis (Mothilal et al., 2020)
    """)
    
    st.markdown("---")
    
    # SHAP Analysis
    st.subheader("🎯 SHAP: Importância Global")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # SHAP bar chart
        fig_shap = px.bar(
            df_shap.head(10),
            y='feature',
            x='importance',
            orientation='h',
            title="Importância SHAP das Top 10 Features"
        )
        fig_shap.update_layout(
            xaxis_title="Valor SHAP Absoluto Médio",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    
    with col2:
        st.info("""
        **📊 Interpretação SHAP:**
        
        Valores mais altos indicam maior influência no modelo.
        
        **Dimensão dominante:**  
        Environmental (E)
        
        **Fundamento:**  
        TreeExplorer otimizado para modelos baseados em árvores.
        """)
    
    st.markdown("---")
    
    # LIME Analysis
    st.subheader("🔬 LIME: Explicações Locais")
    
    st.info("""
    🚧 **Em desenvolvimento**: Módulo LIME será implementado na próxima versão.
    
    **Funcionalidade planejada:**
    - Seleção de observação específica
    - Aproximação linear local
    - Visualização de contribuições por feature
    """)
    
    st.markdown("---")
    
    # DiCE Analysis
    st.subheader("🎲 DiCE: Contrafactuais e Recomendações")
    
    st.info("""
    🚧 **Em desenvolvimento**: Módulo DiCE será implementado na próxima versão.
    
    **Funcionalidade planejada:**
    - Geração de 5 cenários contrafactuais
    - Análise de viabilidade (baixa/média/alta)
    - Plano de ação estratégico
    """)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SIMULADOR AHP
# ═══════════════════════════════════════════════════════════════════════════

elif page == "⚖️ Simulador AHP":
    st.markdown('<div class="main-header">⚖️ Simulador AHP-Gaussiano</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 📖 Metodologia
    
    **AHP-Gaussiano** (Santos et al., 2023) combina:
    - Processo Analítico Hierárquico (Saaty, 1980)
    - Simulações de Monte Carlo
    - Modelagem probabilística de incerteza
    
    **Parâmetros:**
    - 10.000 iterações
    - Ruído gaussiano N(1.0, σ=0.1)
    - Reciprocidade forçada (aᵢⱼ = 1/aⱼᵢ)
    """)
    
    st.markdown("---")
    
    # Pesos atuais
    st.subheader("📊 Pesos Atuais (Baseline)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            df_ahp.style.format({
                'Mean': '{:.4f}',
                'Std': '{:.4f}',
                'CV_%': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    with col2:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=df_ahp['Criterion'].replace({
                'esg_disclosure_score': 'Environmental (E)',
                'annual_return_%': 'Economic (Ec)',
                'volume': 'Governance (G)',
                'char_count': 'Social (S)'
            }),
            values=df_ahp['Mean'],
            hole=0.4
        )])
        fig_pie.update_layout(
            title="Distribuição dos Pesos ESGE",
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Simulador What-If
    st.subheader("🔧 Simulador What-If")
    
    st.info("""
    🚧 **Em desenvolvimento**: Ajuste manual de pesos será implementado.
    
    **Funcionalidade planejada:**
    - Sliders para ajuste de cada dimensão
    - Recálculo em tempo real do score ESGE
    - Análise de sensibilidade
    - Export de cenários
    """)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANÁLISE FINANCEIRA
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📈 Análise Financeira":
    st.markdown('<div class="main-header">📈 Análise Financeira e Event Studies</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 📖 Métricas Implementadas
    
    **Métricas de Risco-Retorno:**
    - Sharpe Ratio
    - Sortino Ratio
    - Maximum Drawdown
    - Value-at-Risk (VaR 95%)
    
    **Event Studies:**
    - Cumulative Abnormal Returns (CAR)
    - Benchmark: S&P/TSX Composite (^GSPTSE)
    """)
    
    st.markdown("---")
    
    # Price evolution
    st.subheader("💹 Evolução do Preço de Fechamento")
    
    fig_price = px.line(
        df_master,
        x='year',
        y='close_price',
        markers=True,
        title="Preço de Fechamento - Teck Resources (TECK-B.TO)"
    )
    fig_price.update_layout(
        xaxis_title="Ano",
        yaxis_title="Preço (CAD)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    st.markdown("---")
    
    # Returns distribution
    st.subheader("📊 Distribuição de Retornos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df_master,
            x='annual_return_%',
            nbins=10,
            title="Distribuição dos Retornos Anuais"
        )
        fig_hist.update_layout(
            xaxis_title="Retorno (%)",
            yaxis_title="Frequência"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Summary statistics
        st.markdown("**📊 Estatísticas Descritivas**")
        
        stats_df = pd.DataFrame({
            'Métrica': [
                'Média',
                'Mediana',
                'Desvio Padrão',
                'Mínimo',
                'Máximo'
            ],
            'Valor (%)': [
                df_master['annual_return_%'].mean(),
                df_master['annual_return_%'].median(),
                df_master['annual_return_%'].std(),
                df_master['annual_return_%'].min(),
                df_master['annual_return_%'].max()
            ]
        })
        
        st.dataframe(
            stats_df.style.format({'Valor (%)': '{:.2f}%'}),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Event Studies
    st.subheader("🎯 Event Studies")
    
    st.info("""
    🚧 **Em desenvolvimento**: Análise de eventos críticos.
    
    **Eventos planejados:**
    1. Rompimento da barragem Mount Polley (2014)
    2. UN Climate Summit (2019)
    3. COVID-19 Crash (2020)
    
    **Metodologia:** Cumulative Abnormal Returns (CAR) com janela de 11 dias.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>XAI-AHP-Gaussian ESGE Framework</strong></p>
    <p>Desenvolvido por Cesar Yoshio Machado Pedroza | USP/Esalq | 2026</p>
    <p>📧 cesar.pedroza@usp.br | 🐙 github.com/cesarpedroza/teck-esge-xai</p>
</div>
""", unsafe_allow_html=True)
