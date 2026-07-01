# XAI-AHP-Gaussian ESGE Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Framework integrado de Inteligência Artificial Explicável (XAI), AHP-Gaussiano e Sistema de Suporte à Decisão (DSS) para decisões estratégicas em ESGE (Environmental, Social, Governance, and Economic).**

## 📚 Sobre o Projeto

Este repositório contém a implementação completa do framework desenvolvido como parte do **Trabalho de Conclusão de Curso (TCC)** do MBA em Data Science and Analytics da **USP/Esalq**, sob orientação do Prof. Dr. Arthur Damasceno Vicente.

### 🎯 Objetivo

Desenvolver e validar um framework que combine:
- **XAI** (SHAP, LIME, DiCE) para explicabilidade
- **AHP-Gaussiano** para ponderação multicritério com incerteza
- **DSS** interativo para decisões estratégicas ESGE

### 🏆 Contribuições Principais

1. **Expansão ESG → ESGE**: Incorporação da dimensão Econômica como critério intrínseco
2. **Triangulação XAI**: SHAP (global) + LIME (local) + DiCE (contrafactuais)
3. **Robustez Probabilística**: AHP-Gaussiano com 10.000 simulações Monte Carlo
4. **Auditabilidade**: Pipeline completo rastreável e reprodutível

### 📊 Caso de Aplicação

**Teck Resources Limited** (TECK-B.TO)
- Setor: Mineração
- Período: 2001-2024 (24 anos)
- Dados: Relatórios de Sustentabilidade + Yahoo Finance

---

## 🗂️ Estrutura do Repositório

```
teck-esge-xai/
├── README.md                          # Este arquivo
├── LICENSE                            # Licença MIT
├── requirements.txt                   # Dependências Python
├── .gitignore                         # Arquivos ignorados pelo Git
│
├── notebooks/                         # Jupyter Notebooks (Pipeline Completo)
│   ├── 01_setup.ipynb                # Configuração do ambiente
│   ├── 02_data_extraction.ipynb      # Extração PDF + API Financeira
│   ├── 03_ml_xai_pipeline.ipynb      # ML (RF/XGBoost) + XAI (SHAP/LIME/DiCE)
│   ├── 04_financial_analysis.ipynb   # Event Studies + Métricas Financeiras
│   ├── 05_ahp_gaussian.ipynb         # AHP-Gaussiano Monte Carlo
│   ├── 06_powerbi_export.ipynb       # Exportação Star Schema
│   └── 07_dss_integration.ipynb      # Teste do DSS Streamlit
│
├── src/                               # Código Python Modular
│   ├── __init__.py
│   ├── config.py                      # Configuração centralizada
│   ├── data_extraction.py             # Extrator de PDFs e APIs
│   ├── ml_models.py                   # Random Forest e XGBoost
│   ├── xai_explainers.py              # SHAP, LIME, DiCE
│   ├── ahp_gaussian.py                # AHP-Gaussiano com Monte Carlo
│   ├── financial_metrics.py           # Sharpe, Sortino, VaR, Event Studies
│   └── utils.py                       # Funções utilitárias
│
├── app/                               # DSS Streamlit
│   ├── streamlit_app.py               # Aplicação principal
│   ├── pages/                         # Páginas multi-página
│   │   ├── 01_📊_Dashboard.py
│   │   ├── 02_🔍_XAI_Explorer.py
│   │   ├── 03_⚖️_AHP_Simulator.py
│   │   └── 04_📈_Financial_Analysis.py
│   └── components/                    # Componentes reutilizáveis
│       ├── charts.py
│       └── widgets.py
│
├── data/                              # Dados (não versionados)
│   ├── raw/                           # PDFs originais
│   ├── processed/                     # CSVs processados
│   └── powerbi/                       # Star Schema para BI
│
├── outputs/                           # Resultados (não versionados)
│   ├── figures/                       # Gráficos exportados
│   ├── tables/                        # Tabelas CSV
│   ├── models/                        # Modelos treinados (.pkl)
│   └── logs/                          # Logs de execução
│
├── docs/                              # Documentação
│   ├── methodology.md                 # Descrição metodológica completa
│   ├── api_reference.md               # Referência das classes/funções
│   └── deployment.md                  # Guia de deploy do DSS
│
└── tests/                             # Testes unitários
    ├── test_data_extraction.py
    ├── test_ml_models.py
    ├── test_ahp_gaussian.py
    └── test_utils.py
```

---

## 🚀 Instalação e Uso

### Pré-requisitos

- Python 3.10 ou superior
- Git
- Jupyter Notebook / JupyterLab

### 1. Clonar o Repositório

```bash
git clone https://github.com/cesarpedroza/teck-esge-xai.git
cd teck-esge-xai
```

### 2. Criar Ambiente Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar Paths

Edite `src/config.py` e ajuste os caminhos conforme seu sistema:

```python
BASE_DIR = Path("C:/Users/seu_usuario/teck-esge-xai")  # Windows
# ou
BASE_DIR = Path("/home/seu_usuario/teck-esge-xai")     # Linux/Mac
```

### 5. Executar Pipeline Completo

#### Opção A: Via Notebooks (Recomendado para TCC)

Execute os notebooks na ordem:

```bash
jupyter notebook
```

1. `01_setup.ipynb` - Configuração inicial
2. `02_data_extraction.ipynb` - Extração de dados
3. `03_ml_xai_pipeline.ipynb` - ML + XAI
4. `04_financial_analysis.ipynb` - Análise financeira
5. `05_ahp_gaussian.ipynb` - AHP-Gaussiano
6. `06_powerbi_export.ipynb` - Exportação BI

#### Opção B: Via Script Python (Produção)

```python
from src.config import ESGEConfig
from src.data_extraction import MasterExtractor
from src.ml_models import ESGEPredictor
from src.ahp_gaussian import AHPGaussianDSS

# Configuração
config = ESGEConfig()

# 1. Extração de Dados
extractor = MasterExtractor(ticker="TECK-B.TO", start_year=2001, end_year=2024)
df_master = extractor.run()

# 2. Modelagem e XAI
predictor = ESGEPredictor(config)
results = predictor.run_full_pipeline(df_master)

# 3. AHP-Gaussiano
ahp = AHPGaussianDSS(config)
weights_df = ahp.run_monte_carlo(n_simulations=10000)

print("✅ Pipeline executado com sucesso!")
```

### 6. Executar DSS Streamlit

```bash
cd app
streamlit run streamlit_app.py
```

Acesse: http://localhost:8501

---

## 📊 Resultados Principais

### Modelagem Preditiva

| Modelo | R² | RMSE | MAE |
|--------|------|------|-----|
| **Random Forest** | 0.7307 | 12.45 | 8.32 |
| **XGBoost** | 0.7501 | 11.89 | 7.98 |

### Top 5 Features (SHAP)

1. **esg_disclosure_score** - Importância: 0.334
2. **annual_return_%** - Importância: 0.272
3. **volume** - Importância: 0.218
4. **char_count** - Importância: 0.177

### Pesos AHP-Gaussiano

| Dimensão | Peso Médio | DP | CV% | IC 95% |
|----------|------------|-------|------|--------|
| **Environmental (E)** | 0.334 | 0.013 | 3.87% | [0.308, 0.360] |
| **Economic (Ec)** | 0.272 | 0.012 | 4.27% | [0.249, 0.295] |
| **Governance (G)** | 0.218 | 0.010 | 4.67% | [0.198, 0.238] |
| **Social (S)** | 0.177 | 0.009 | 4.84% | [0.160, 0.194] |

**Consistency Ratio médio**: 0.0038 ✅ (< 0.10)

---

## 🔬 Metodologia

### 1. Extração de Dados

- **PDFs**: pdfplumber (extração de texto)
- **API Financeira**: Yahoo Finance (TECK-B.TO)
- **Período**: 2001-2024 (24 observações anuais)

### 2. Machine Learning

- **Modelos**: Random Forest, XGBoost
- **Target**: Close Price (variável contínua)
- **Validação**: Train/Test Split 80/20 + LOOCV

### 3. Explicabilidade (XAI)

- **SHAP**: TreeExplainer para importância global
- **LIME**: Explicações locais por observação
- **DiCE**: Contrafactuais acionáveis

### 4. Ponderação Multicritério

- **AHP-Gaussiano**: 10.000 simulações Monte Carlo
- **Ruído**: N(1.0, σ=0.1) aplicado à matriz paritária
- **Reciprocidade**: Forçada a cada iteração

### 5. Decision Support System

- **Framework**: Streamlit
- **Features**: Simulações What-If, Análise de Sensibilidade, Export de Relatórios

---

## 📖 Publicação e Citação

### Artigo (Submetido)

> **Pedroza, C. Y. M.; Vicente, A. D.** (2026). "XAI-AHP-Gaussian Framework for Strategic ESGE Decisions in the Mining Sector: An Application to Teck Resources Ltd." *Expert Systems with Applications* (em revisão).

### BibTeX

```bibtex
@article{pedroza2026xai,
  title={XAI-AHP-Gaussian Framework for Strategic ESGE Decisions in the Mining Sector},
  author={Pedroza, Cesar Yoshio Machado and Vicente, Arthur Damasceno},
  journal={Expert Systems with Applications},
  year={2026},
  publisher={Elsevier}
}
```

### DOI (Zenodo)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

*Será atualizado após publicação no Zenodo*

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

### Padrões de Código

- **PEP 8**: Formatação Python
- **Black**: Formatador automático
- **Type hints**: Obrigatório para funções públicas
- **Docstrings**: Google Style

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Cesar Yoshio Machado Pedroza**

- 📧 Email: cesar.pedroza@usp.br
- 💼 LinkedIn: [linkedin.com/in/cesarpedroza](https://linkedin.com/in/cesarpedroza)
- 🐱 GitHub: [@cesarpedroza](https://github.com/cesarpedroza)

### Orientador

**Prof. Dr. Arthur Damasceno Vicente**
- Instituição: USP/Esalq
- Email: arthur.vicente@usp.br

---

## 🙏 Agradecimentos

- **USP/Esalq** - Programa de MBA em Data Science and Analytics
- **Teck Resources Ltd.** - Disponibilização pública de relatórios de sustentabilidade
- **Comunidade Open Source** - Pacotes utilizados (pandas, scikit-learn, SHAP, LIME, DiCE, Streamlit)

---

## 📚 Referências Principais

1. **Lundberg, S. M.; Lee, S. I.** (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

2. **Ribeiro, M. T.; Singh, S.; Guestrin, C.** (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD*, 1135-1144.

3. **Mothilal, R. K. et al.** (2020). Explaining machine learning predictions through counterfactual examples. *Proceedings of the AAAI Conference on AI*, 34(4), 6116-6123.

4. **Santos, L. F. O. M.; Oishi, J.; Yoshizawa, M.** (2023). Gaussian Analytic Hierarchy Process: A probabilistic approach for decision making under uncertainty. *Expert Systems with Applications*, 214, 119130.

5. **Saaty, T. L.** (1980). *The Analytic Hierarchy Process*. New York: McGraw-Hill.

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub! ⭐**

Desenvolvido com ❤️ para o avanço da Ciência de Dados em Sustentabilidade Corporativa

</div>
