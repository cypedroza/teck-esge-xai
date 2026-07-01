"""
════════════════════════════════════════════════════════════════════════════
SCRIPT 1: ANÁLISE ESTATÍSTICA COMPLETA (2001-2025)
════════════════════════════════════════════════════════════════════════════

Análise estatística profunda dos 25 anos de dados ESG da Teck Resources:
- Normalização de features (nomes descritivos)
- Estatísticas descritivas completas
- Testes de normalidade (Shapiro-Wilk, Kolmogorov-Smirnov)
- Análise de quebra estrutural (Chow test 2018)
- Testes de diferenças temporais (ANOVA/Kruskal-Wallis)
- Análise de correlações (Pearson/Spearman)
- Detecção de outliers (IQR, Z-score)
- Comparação pré-TCFD (2001-2017) vs pós-TCFD (2018-2025)

Autor: Cesar Yoshio Machado Pedroza
Data: 2026-04-21
════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, normaltest, kruskal, f_oneway
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(r"C:\Users\user\Documents\_MBA_Data_Science_Analytics\00 - Temas TCC\teck-esge-xai")
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUTS_FIGURES = BASE_DIR / "outputs" / "figures"

# Criar diretórios se não existirem
OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# MAPEAMENTO E NORMALIZAÇÃO DE FEATURES
# ════════════════════════════════════════════════════════════════════════════

FEATURE_MAPPING = {
    # Dimensão Environmental (E)
    'esg_disclosure_score': {
        'normalized_name': 'ESG_Disclosure_Index',
        'display_name': 'ESG Disclosure Index',
        'description': 'Frequência de menções a termos ESG nos relatórios anuais',
        'dimension': 'Environmental',
        'unit': 'contagem',
        'interpretation': 'Maior = maior transparência ESG'
    },
    
    # Dimensão Social (S)
    'char_count': {
        'normalized_name': 'Report_Quality_Score',
        'display_name': 'Report Quality Score',
        'description': 'Extensão e detalhamento dos relatórios de sustentabilidade',
        'dimension': 'Social',
        'unit': 'caracteres',
        'interpretation': 'Maior = relatórios mais detalhados'
    },
    
    # Dimensão Economic (Ec)
    'annual_return_%': {
        'normalized_name': 'Annual_Return_Pct',
        'display_name': 'Annual Return (%)',
        'description': 'Retorno financeiro anualizado ajustado',
        'dimension': 'Economic',
        'unit': 'percentual',
        'interpretation': 'Maior = melhor performance financeira'
    },
    
    # Dimensão Governance (G)
    'volume': {
        'normalized_name': 'Market_Liquidity',
        'display_name': 'Market Liquidity',
        'description': 'Volume médio anual de transações acionárias',
        'dimension': 'Governance',
        'unit': 'volume negociado',
        'interpretation': 'Maior = maior liquidez de mercado'
    }
}


def normalize_feature_names(df):
    """Normaliza nomes de features para versões descritivas."""
    rename_dict = {old: mapping['normalized_name'] 
                   for old, mapping in FEATURE_MAPPING.items() 
                   if old in df.columns}
    return df.rename(columns=rename_dict)


def get_feature_description(feature_name):
    """Retorna descrição completa de uma feature."""
    for old_name, mapping in FEATURE_MAPPING.items():
        if mapping['normalized_name'] == feature_name or old_name == feature_name:
            return mapping
    return None


# ════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

class ComprehensiveStatisticalAnalysis:
    """Análise estatística completa dos dados ESGE (25 anos)."""
    
    def __init__(self, data_path: Path):
        """Inicializa com dataset master."""
        print("═" * 80)
        print("ANÁLISE ESTATÍSTICA COMPLETA - DADOS ESGE (2001-2025)")
        print("═" * 80)
        
        # Carregar dados
        self.df_raw = pd.read_csv(data_path)
        print(f"\n✅ Dataset carregado: {self.df_raw.shape}")
        print(f"   Anos: {self.df_raw['year'].min()} - {self.df_raw['year'].max()}")
        
        # Normalizar nomes
        self.df = normalize_feature_names(self.df_raw.copy())
        print(f"\n✅ Features normalizadas:")
        for old, new in zip(self.df_raw.columns, self.df.columns):
            if old != new:
                print(f"   {old:30} → {new}")
        
        # Features numéricas (excluindo year e filename)
        self.numeric_features = [col for col in self.df.columns 
                                if col not in ['year', 'filename'] and 
                                self.df[col].dtype in ['int64', 'float64']]
        
        # Períodos de análise
        self.period_pre_tcfd = self.df[self.df['year'] <= 2017].copy()
        self.period_post_tcfd = self.df[self.df['year'] >= 2018].copy()
        
        print(f"\n✅ Períodos definidos:")
        print(f"   Pré-TCFD (2001-2017): {len(self.period_pre_tcfd)} anos")
        print(f"   Pós-TCFD (2018-2025): {len(self.period_post_tcfd)} anos")
        
        # Resultados
        self.results = {}
    
    def descriptive_statistics(self):
        """Calcula estatísticas descritivas completas."""
        print("\n" + "─" * 80)
        print("1. ESTATÍSTICAS DESCRITIVAS")
        print("─" * 80)
        
        stats_list = []
        
        for feature in self.numeric_features:
            data = self.df[feature].dropna()
            
            # Calcular estatísticas
            stats_dict = {
                'Feature': feature,
                'N': len(data),
                'Mean (μ)': data.mean(),
                'Std (σ)': data.std(),
                'Min': data.min(),
                'Q1 (25%)': data.quantile(0.25),
                'Median': data.median(),
                'Q3 (75%)': data.quantile(0.75),
                'Max': data.max(),
                'Range': data.max() - data.min(),
                'IQR': data.quantile(0.75) - data.quantile(0.25),
                'CV (%)': (data.std() / data.mean()) * 100 if data.mean() != 0 else np.nan,
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            }
            
            stats_list.append(stats_dict)
            
            # Exibir
            desc = get_feature_description(feature)
            print(f"\n{feature}")
            print(f"  Dimensão: {desc['dimension'] if desc else 'N/A'}")
            print(f"  μ = {stats_dict['Mean (μ)']:.4f} ± σ = {stats_dict['Std (σ)']:.4f}")
            print(f"  Range: [{stats_dict['Min']:.4f}, {stats_dict['Max']:.4f}]")
            print(f"  CV = {stats_dict['CV (%)']:.2f}%")
        
        # Salvar
        df_stats = pd.DataFrame(stats_list)
        df_stats.to_csv(OUTPUTS_TABLES / "descriptive_statistics.csv", index=False)
        print(f"\n✅ Salvo: descriptive_statistics.csv")
        
        self.results['descriptive_stats'] = df_stats
        return df_stats
    
    def normality_tests(self):
        """Testa normalidade das distribuições."""
        print("\n" + "─" * 80)
        print("2. TESTES DE NORMALIDADE")
        print("─" * 80)
        
        normality_results = []
        
        for feature in self.numeric_features:
            data = self.df[feature].dropna()
            
            # Shapiro-Wilk (melhor para n < 50)
            shapiro_stat, shapiro_p = shapiro(data)
            
            # Kolmogorov-Smirnov
            ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
            
            # D'Agostino-Pearson
            da_stat, da_p = normaltest(data)
            
            result = {
                'Feature': feature,
                'Shapiro_W': shapiro_stat,
                'Shapiro_p': shapiro_p,
                'Shapiro_Normal': 'Sim' if shapiro_p > 0.05 else 'Não',
                'KS_Stat': ks_stat,
                'KS_p': ks_p,
                'KS_Normal': 'Sim' if ks_p > 0.05 else 'Não',
                'DA_Stat': da_stat,
                'DA_p': da_p,
                'DA_Normal': 'Sim' if da_p > 0.05 else 'Não'
            }
            
            normality_results.append(result)
            
            # Exibir
            print(f"\n{feature}:")
            print(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f} → {'Normal ✓' if shapiro_p > 0.05 else 'Não-normal ✗'}")
            print(f"  Kolmogorov-Smirnov: D={ks_stat:.4f}, p={ks_p:.4f}")
        
        # Salvar
        df_normality = pd.DataFrame(normality_results)
        df_normality.to_csv(OUTPUTS_TABLES / "normality_tests.csv", index=False)
        print(f"\n✅ Salvo: normality_tests.csv")
        
        self.results['normality'] = df_normality
        return df_normality
    
    def chow_test_structural_break(self):
        """Teste de quebra estrutural (Chow test) em 2018."""
        print("\n" + "─" * 80)
        print("3. TESTE DE QUEBRA ESTRUTURAL (CHOW TEST 2018)")
        print("─" * 80)
        
        chow_results = []
        
        for feature in self.numeric_features:
            # Dados completos
            full_data = self.df[['year', feature]].dropna()
            
            # Período 1 (2001-2017) e Período 2 (2018-2025)
            period1 = full_data[full_data['year'] <= 2017]
            period2 = full_data[full_data['year'] >= 2018]
            
            if len(period1) < 3 or len(period2) < 3:
                print(f"  ⚠️ {feature}: dados insuficientes")
                continue
            
            # Regressão no período completo
            X_full = sm.add_constant(full_data['year'])
            model_full = sm.OLS(full_data[feature], X_full).fit()
            rss_full = model_full.ssr
            
            # Regressões nos sub-períodos
            X_p1 = sm.add_constant(period1['year'])
            model_p1 = sm.OLS(period1[feature], X_p1).fit()
            rss_p1 = model_p1.ssr
            
            X_p2 = sm.add_constant(period2['year'])
            model_p2 = sm.OLS(period2[feature], X_p2).fit()
            rss_p2 = model_p2.ssr
            
            # Estatística de Chow
            rss_split = rss_p1 + rss_p2
            k = 2  # número de parâmetros
            n = len(full_data)
            chow_stat = ((rss_full - rss_split) / k) / (rss_split / (n - 2*k))
            
            # P-value
            chow_p = 1 - stats.f.cdf(chow_stat, k, n - 2*k)
            
            result = {
                'Feature': feature,
                'Chow_F': chow_stat,
                'p_value': chow_p,
                'Quebra_Estrutural': 'Sim (p<0.05)' if chow_p < 0.05 else 'Não (p≥0.05)',
                'Mean_Pre_TCFD': period1[feature].mean(),
                'Mean_Post_TCFD': period2[feature].mean(),
                'Delta_%': ((period2[feature].mean() - period1[feature].mean()) / period1[feature].mean()) * 100
            }
            
            chow_results.append(result)
            
            # Exibir
            print(f"\n{feature}:")
            print(f"  Chow F-statistic: {chow_stat:.4f}, p-value: {chow_p:.4f}")
            print(f"  Quebra estrutural: {'✓ Sim' if chow_p < 0.05 else '✗ Não'}")
            print(f"  Δ Pré→Pós: {result['Delta_%']:.2f}%")
        
        # Salvar
        df_chow = pd.DataFrame(chow_results)
        df_chow.to_csv(OUTPUTS_TABLES / "chow_test_structural_break.csv", index=False)
        print(f"\n✅ Salvo: chow_test_structural_break.csv")
        
        self.results['chow_test'] = df_chow
        return df_chow
    
    def temporal_differences_tests(self):
        """Testa diferenças entre períodos pré/pós TCFD."""
        print("\n" + "─" * 80)
        print("4. TESTES DE DIFERENÇAS TEMPORAIS (PRÉ vs PÓS TCFD)")
        print("─" * 80)
        
        temporal_results = []
        
        for feature in self.numeric_features:
            pre = self.period_pre_tcfd[feature].dropna()
            post = self.period_post_tcfd[feature].dropna()
            
            if len(pre) < 3 or len(post) < 3:
                continue
            
            # Mann-Whitney U (não-paramétrico, robusto)
            mw_stat, mw_p = stats.mannwhitneyu(pre, post, alternative='two-sided')
            
            # T-test independente (paramétrico)
            t_stat, t_p = stats.ttest_ind(pre, post)
            
            # Effect size (Cohen's d)
            cohens_d = (post.mean() - pre.mean()) / np.sqrt((pre.std()**2 + post.std()**2) / 2)
            
            result = {
                'Feature': feature,
                'Mean_Pre': pre.mean(),
                'Std_Pre': pre.std(),
                'Mean_Post': post.mean(),
                'Std_Post': post.std(),
                'Mann_Whitney_U': mw_stat,
                'MW_p_value': mw_p,
                'T_test_stat': t_stat,
                'T_p_value': t_p,
                'Cohens_d': cohens_d,
                'Effect_Size': 'Pequeno' if abs(cohens_d) < 0.5 else 'Médio' if abs(cohens_d) < 0.8 else 'Grande',
                'Diferença_Significativa': 'Sim (p<0.05)' if mw_p < 0.05 else 'Não'
            }
            
            temporal_results.append(result)
            
            # Exibir
            print(f"\n{feature}:")
            print(f"  Pré-TCFD:  μ={pre.mean():.4f} ± σ={pre.std():.4f}")
            print(f"  Pós-TCFD: μ={post.mean():.4f} ± σ={post.std():.4f}")
            print(f"  Mann-Whitney p={mw_p:.4f} → {'Diferença significativa ✓' if mw_p < 0.05 else 'Sem diferença ✗'}")
            print(f"  Cohen's d={cohens_d:.4f} ({result['Effect_Size']})")
        
        # Salvar
        df_temporal = pd.DataFrame(temporal_results)
        df_temporal.to_csv(OUTPUTS_TABLES / "temporal_differences_tests.csv", index=False)
        print(f"\n✅ Salvo: temporal_differences_tests.csv")
        
        self.results['temporal_tests'] = df_temporal
        return df_temporal
    
    def correlation_analysis(self):
        """Análise de correlações (Pearson e Spearman)."""
        print("\n" + "─" * 80)
        print("5. ANÁLISE DE CORRELAÇÕES")
        print("─" * 80)
        
        # Pearson (linear)
        corr_pearson = self.df[self.numeric_features].corr(method='pearson')
        
        # Spearman (monotônica, não-paramétrica)
        corr_spearman = self.df[self.numeric_features].corr(method='spearman')
        
        # P-values para Pearson
        n = len(self.df)
        p_values = pd.DataFrame(np.zeros_like(corr_pearson), 
                                columns=corr_pearson.columns, 
                                index=corr_pearson.index)
        
        for i in range(len(corr_pearson)):
            for j in range(len(corr_pearson)):
                if i != j:
                    r = corr_pearson.iloc[i, j]
                    t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                    p_values.iloc[i, j] = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Salvar
        corr_pearson.to_csv(OUTPUTS_TABLES / "correlation_pearson.csv")
        corr_spearman.to_csv(OUTPUTS_TABLES / "correlation_spearman.csv")
        p_values.to_csv(OUTPUTS_TABLES / "correlation_p_values.csv")
        
        print("\n✅ Correlações Pearson:")
        print(corr_pearson.round(3))
        
        print("\n✅ Salvo: correlation_pearson.csv, correlation_spearman.csv, correlation_p_values.csv")
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_pearson, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title('Matriz de Correlação (Pearson) - Features ESGE', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Salvo: correlation_matrix.png")
        
        self.results['correlations'] = {
            'pearson': corr_pearson,
            'spearman': corr_spearman,
            'p_values': p_values
        }
        
        return corr_pearson, corr_spearman
    
    def outlier_detection(self):
        """Detecta outliers usando IQR e Z-score."""
        print("\n" + "─" * 80)
        print("6. DETECÇÃO DE OUTLIERS")
        print("─" * 80)
        
        outlier_summary = []
        
        for feature in self.numeric_features:
            data = self.df[feature].dropna()
            
            # Método IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
            
            # Método Z-score
            z_scores = np.abs(stats.zscore(data))
            outliers_z = data[z_scores > 3]
            
            result = {
                'Feature': feature,
                'N_Total': len(data),
                'N_Outliers_IQR': len(outliers_iqr),
                'Pct_Outliers_IQR': (len(outliers_iqr) / len(data)) * 100,
                'N_Outliers_ZScore': len(outliers_z),
                'Pct_Outliers_ZScore': (len(outliers_z) / len(data)) * 100,
                'IQR_Lower': lower_bound,
                'IQR_Upper': upper_bound
            }
            
            outlier_summary.append(result)
            
            # Exibir
            print(f"\n{feature}:")
            print(f"  Outliers (IQR): {len(outliers_iqr)} ({result['Pct_Outliers_IQR']:.1f}%)")
            print(f"  Outliers (Z>3): {len(outliers_z)} ({result['Pct_Outliers_ZScore']:.1f}%)")
        
        # Salvar
        df_outliers = pd.DataFrame(outlier_summary)
        df_outliers.to_csv(OUTPUTS_TABLES / "outlier_detection.csv", index=False)
        print(f"\n✅ Salvo: outlier_detection.csv")
        
        self.results['outliers'] = df_outliers
        return df_outliers
    
    def generate_summary_report(self):
        """Gera relatório resumo consolidado."""
        print("\n" + "═" * 80)
        print("RELATÓRIO RESUMO CONSOLIDADO")
        print("═" * 80)
        
        report = []
        
        report.append("=" * 80)
        report.append("ANÁLISE ESTATÍSTICA COMPLETA - TECK RESOURCES (2001-2025)")
        report.append("=" * 80)
        report.append(f"\nDataset: {len(self.df)} observações, {len(self.numeric_features)} features")
        report.append(f"Período: {self.df['year'].min()} - {self.df['year'].max()}")
        
        report.append("\n" + "-" * 80)
        report.append("PRINCIPAIS ACHADOS")
        report.append("-" * 80)
        
        # Quebra estrutural
        if 'chow_test' in self.results:
            df_chow = self.results['chow_test']
            quebras = df_chow[df_chow['Quebra_Estrutural'].str.contains('Sim')]
            report.append(f"\n1. QUEBRA ESTRUTURAL (2018):")
            report.append(f"   Features com quebra significativa: {len(quebras)}/{len(df_chow)}")
            for _, row in quebras.iterrows():
                report.append(f"   - {row['Feature']}: Δ={row['Delta_%']:.2f}% (p={row['p_value']:.4f})")
        
        # Normalidade
        if 'normality' in self.results:
            df_norm = self.results['normality']
            normais = df_norm[df_norm['Shapiro_Normal'] == 'Sim']
            report.append(f"\n2. NORMALIDADE:")
            report.append(f"   Features com distribuição normal: {len(normais)}/{len(df_norm)}")
        
        # Correlações fortes
        if 'correlations' in self.results:
            corr = self.results['correlations']['pearson']
            strong_corr = []
            for i in range(len(corr)):
                for j in range(i+1, len(corr)):
                    r = corr.iloc[i, j]
                    if abs(r) > 0.7:
                        strong_corr.append((corr.index[i], corr.columns[j], r))
            
            report.append(f"\n3. CORRELAÇÕES FORTES (|r| > 0.7):")
            if strong_corr:
                for f1, f2, r in strong_corr:
                    report.append(f"   - {f1} <-> {f2}: r={r:.3f}")
            else:
                report.append("   Nenhuma correlação forte detectada")
        
        report.append("\n" + "=" * 80)
        
        # Salvar
        report_text = "\n".join(report)
        with open(OUTPUTS_TABLES / "statistical_analysis_summary.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✅ Relatório salvo: statistical_analysis_summary.txt")


# ════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Carregar dados
    data_path = DATA_PROCESSED / "esge_master.csv"
    
    if not data_path.exists():
        print(f"❌ ERRO: Arquivo não encontrado: {data_path}")
        print("   Execute o notebook 02_data_extraction.ipynb primeiro!")
        sys.exit(1)
    
    # Criar análise
    analysis = ComprehensiveStatisticalAnalysis(data_path)
    
    # Executar todas as análises
    analysis.descriptive_statistics()
    analysis.normality_tests()
    analysis.chow_test_structural_break()
    analysis.temporal_differences_tests()
    analysis.correlation_analysis()
    analysis.outlier_detection()
    analysis.generate_summary_report()
    
    print("\n" + "═" * 80)
    print("✅ ANÁLISE ESTATÍSTICA COMPLETA CONCLUÍDA!")
    print("═" * 80)
    print(f"\nArquivos gerados em: {OUTPUTS_TABLES}")
    print("\nPróximo passo: Execute 02_ml_advanced_validation.py")
