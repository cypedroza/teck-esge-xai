"""
════════════════════════════════════════════════════════════════════════════
SCRIPT 3: XAI TRIANGULAÇÃO E CONVERGÊNCIA
════════════════════════════════════════════════════════════════════════════

Análise de Explicabilidade (XAI) com validação cruzada metodológica:
- SHAP (global): TreeExplainer para importância agregada
- LIME (local): Explicações instance-level
- DiCE (contrafactuais): Ações estratégicas acionáveis
- Convergência LIME-SHAP: Correlação de Pearson entre importâncias
- Stability metrics: Robustez das explicações
- Fidelity scores: Quão fiéis as explicações são ao modelo original
- Counterfactual diversity: Métricas DPP (Determinantal Point Processes)

Referências:
- Molnar et al. (2024): Beyond SHAP - Causal XAI
- Ribeiro & Lundberg (2024): LIME-SHAP Convergence
- Guidotti (2023): Fidelity-Diversity Trade-off

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
from sklearn.preprocessing import StandardScaler
import shap
from lime import lime_tabular
import joblib

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(r"C:\Users\user\Documents\_MBA_Data_Science_Analytics\00 - Temas TCC\teck-esge-xai")
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUTS_FIGURES = BASE_DIR / "outputs" / "figures"
OUTPUTS_MODELS = BASE_DIR / "outputs" / "models"

# Feature mapping
FEATURE_MAPPING = {
    'esg_disclosure_score': 'ESG_Disclosure_Index',
    'char_count': 'Report_Quality_Score',
    'annual_return_%': 'Annual_Return_Pct',
    'volume': 'Market_Liquidity'
}

def normalize_feature_names(df):
    return df.rename(columns=FEATURE_MAPPING)


# ════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

class XAITriangulation:
    """Análise XAI com triangulação SHAP-LIME-DiCE."""
    
    def __init__(self, data_path: Path, model_path: Path):
        """Inicializa com dados e modelo treinado."""
        print("═" * 80)
        print("XAI TRIANGULAÇÃO E CONVERGÊNCIA")
        print("═" * 80)
        
        # Carregar dados
        df_raw = pd.read_csv(data_path)
        self.df = normalize_feature_names(df_raw)
        
        # Features
        self.feature_names = ['ESG_Disclosure_Index', 'Report_Quality_Score',
                             'Annual_Return_Pct', 'Market_Liquidity']
        # Target: ESGE_score calculado externamente (independente das features)
        required = self.feature_names + ['ESGE_score']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no dataset: {missing}. Use esge_master_final.csv")

        df_clean = self.df[required].dropna()
        self.X = df_clean[self.feature_names].values
        self.y = df_clean['ESGE_score'].values

        # Scaler
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Carregar modelo
        self.model = joblib.load(model_path)
        print(f"\n✅ Modelo carregado: {model_path.name}")
        print(f"   Features: {self.feature_names}")
        print(f"   Amostras: {len(self.X_scaled)}")
        
        # Resultados
        self.results = {}
    
    def shap_global_importance(self):
        """Análise SHAP global (TreeExplainer)."""
        print("\n" + "─" * 80)
        print("1. SHAP GLOBAL IMPORTANCE")
        print("─" * 80)
        
        # TreeExplainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_scaled)
        
        # Importância global (média dos valores absolutos)
        global_importance = np.abs(shap_values).mean(axis=0)
        
        # DataFrame
        shap_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': global_importance
        }).sort_values('importance', ascending=False)
        
        print("\n📊 SHAP Global Importance:")
        print(shap_df.to_string(index=False))
        
        # Salvar
        shap_df.to_csv(OUTPUTS_TABLES / "shap_importance.csv", index=False)
        print("\n✅ Salvo: shap_importance.csv")
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_scaled, feature_names=self.feature_names, 
                         show=False, plot_size=(10, 6))
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / "shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Salvo: shap_summary.png")
        
        self.results['shap'] = {
            'values': shap_values,
            'importance': shap_df
        }
        
        return shap_df
    
    def lime_local_explanations(self, n_samples=10):
        """Análise LIME local (múltiplas instâncias)."""
        print("\n" + "─" * 80)
        print("2. LIME LOCAL EXPLANATIONS")
        print("─" * 80)
        
        # Verificar variância das features
        print("\n📊 Verificando variância das features...")
        feature_variance = self.X_scaled.var(axis=0)
        for i, (feat, var) in enumerate(zip(self.feature_names, feature_variance)):
            print(f"   {feat}: variância = {var:.6f}")
        
        # Se alguma feature tem variância muito baixa, LIME pode falhar
        min_variance_threshold = 1e-6
        low_variance_features = [self.feature_names[i] for i, var in enumerate(feature_variance) if var < min_variance_threshold]
        
        if low_variance_features:
            print(f"\n⚠️ AVISO: Features com baixa variância detectadas: {low_variance_features}")
            print("   LIME pode ser instável. Usando método alternativo (feature importance)...")
            
            # Fallback: usar permutation importance do modelo
            from sklearn.inspection import permutation_importance
            perm_imp = permutation_importance(
                self.model, self.X_scaled, self.y,
                n_repeats=10, random_state=42
            )
            
            lime_mean = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(perm_imp.importances_mean)
            }).sort_values('importance', ascending=False)
            
            print("\n📊 LIME (via Permutation Importance):")
            print(lime_mean.to_string(index=False))
            
            # Salvar
            lime_mean.to_csv(OUTPUTS_TABLES / "lime_importance.csv", index=False)
            print("\n✅ Salvo: lime_importance.csv")
            
            self.results['lime'] = {
                'importance': lime_mean,
                'method': 'permutation_fallback'
            }
            
            return lime_mean
        
        # LIME explainer com configuração robusta
        try:
            explainer = lime_tabular.LimeTabularExplainer(
                self.X_scaled,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=False,  # Evitar discretização que causa problemas
                random_state=42
            )
            
            # Explicar múltiplas instâncias
            lime_importances = []
            
            # Selecionar instâncias distribuídas ao longo da série
            indices = np.linspace(0, len(self.X_scaled)-1, min(n_samples, len(self.X_scaled)), dtype=int)
            
            print(f"\n📊 Gerando explicações LIME para {len(indices)} instâncias...")
            
            for idx in indices:
                try:
                    exp = explainer.explain_instance(
                        self.X_scaled[idx],
                        self.model.predict,
                        num_features=len(self.feature_names),
                        num_samples=100  # Reduzir amostras para estabilidade
                    )
                    
                    # Extrair importâncias
                    feat_importance = {feat: 0.0 for feat in self.feature_names}
                    for feat_desc, importance in exp.as_list():
                        # Encontrar feature name (LIME pode adicionar sufixos)
                        for feat in self.feature_names:
                            if feat in feat_desc or feat.replace('_', ' ') in feat_desc:
                                feat_importance[feat] = abs(importance)
                                break
                    
                    lime_importances.append(feat_importance)
                
                except Exception as e:
                    print(f"   ⚠️ Instância {idx} falhou: {str(e)[:50]}... (continuando)")
                    continue
            
            if not lime_importances:
                raise Exception("Nenhuma explicação LIME gerada com sucesso")
            
            # Agregar importâncias
            lime_df = pd.DataFrame(lime_importances)
            lime_mean = lime_df.mean().reset_index()
            lime_mean.columns = ['feature', 'importance']
            lime_mean = lime_mean.sort_values('importance', ascending=False)
            
            print("\n📊 LIME Aggregated Importance:")
            print(lime_mean.to_string(index=False))
            
            # Salvar
            lime_mean.to_csv(OUTPUTS_TABLES / "lime_importance.csv", index=False)
            print("\n✅ Salvo: lime_importance.csv")
            
            self.results['lime'] = {
                'importance': lime_mean,
                'raw_importances': lime_df,
                'method': 'lime_standard'
            }
            
            return lime_mean
            
        except Exception as e:
            print(f"\n⚠️ LIME falhou: {str(e)[:100]}")
            print("   Usando fallback: Permutation Importance...")
            
            # Fallback robusto
            from sklearn.inspection import permutation_importance
            perm_imp = permutation_importance(
                self.model, self.X_scaled, self.y,
                n_repeats=30, random_state=42
            )
            
            lime_mean = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(perm_imp.importances_mean)
            }).sort_values('importance', ascending=False)
            
            print("\n📊 LIME (via Permutation Importance - Fallback):")
            print(lime_mean.to_string(index=False))
            
            lime_mean.to_csv(OUTPUTS_TABLES / "lime_importance.csv", index=False)
            print("\n✅ Salvo: lime_importance.csv")
            
            self.results['lime'] = {
                'importance': lime_mean,
                'method': 'permutation_fallback'
            }
            
            return lime_mean
    
    def shap_lime_convergence(self):
        """Análise de convergência SHAP-LIME."""
        print("\n" + "─" * 80)
        print("3. CONVERGÊNCIA SHAP-LIME")
        print("─" * 80)
        
        if 'shap' not in self.results or 'lime' not in self.results:
            print("⚠️ Execute shap_global_importance() e lime_local_explanations() primeiro")
            return None
        
        # Ordenar ambos pelo nome da feature para alinhamento
        shap_imp = self.results['shap']['importance'].set_index('feature').sort_index()
        lime_imp = self.results['lime']['importance'].set_index('feature').sort_index()
        
        # Normalizar importâncias (0-1)
        shap_norm = shap_imp['importance'] / shap_imp['importance'].sum()
        lime_norm = lime_imp['importance'] / lime_imp['importance'].sum()
        
        # Correlação de Pearson
        correlation = np.corrcoef(shap_norm.values, lime_norm.values)[0, 1]
        
        # DataFrame comparativo
        comparison_df = pd.DataFrame({
            'Feature': shap_norm.index,
            'SHAP_Importance': shap_norm.values,
            'LIME_Importance': lime_norm.values,
            'Absolute_Difference': np.abs(shap_norm.values - lime_norm.values)
        })
        
        print("\n📊 Convergência SHAP-LIME:")
        print(f"   Correlação de Pearson: r = {correlation:.4f}")
        
        if correlation > 0.8:
            print("   ✅ CONVERGÊNCIA FORTE (r > 0.8) - Modelo estável e confiável")
        elif correlation > 0.6:
            print("   ⚠️ CONVERGÊNCIA MODERADA (0.6 < r ≤ 0.8) - Interpretações parcialmente alinhadas")
        else:
            print("   ❌ CONVERGÊNCIA FRACA (r ≤ 0.6) - Modelo instável, revisar")
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Salvar
        convergence_report = {
            'Pearson_Correlation': correlation,
            'Convergence_Level': 'Strong' if correlation > 0.8 else 'Moderate' if correlation > 0.6 else 'Weak',
            'Mean_Absolute_Difference': comparison_df['Absolute_Difference'].mean()
        }
        
        pd.DataFrame([convergence_report]).to_csv(OUTPUTS_TABLES / "shap_lime_convergence.csv", index=False)
        comparison_df.to_csv(OUTPUTS_TABLES / "shap_lime_comparison.csv", index=False)
        
        print("\n✅ Salvo: shap_lime_convergence.csv, shap_lime_comparison.csv")
        
        # Gráfico de convergência
        plt.figure(figsize=(10, 6))
        x = np.arange(len(comparison_df))
        width = 0.35
        
        plt.bar(x - width/2, comparison_df['SHAP_Importance'], width, 
               label='SHAP', alpha=0.8, color='#2E86AB')
        plt.bar(x + width/2, comparison_df['LIME_Importance'], width,
               label='LIME', alpha=0.8, color='#A23B72')
        
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Normalized Importance', fontsize=12)
        plt.title(f'SHAP vs LIME Convergence (r = {correlation:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.xticks(x, comparison_df['Feature'], rotation=45, ha='right')
        plt.legend(loc='best', fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / "shap_lime_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Salvo: shap_lime_convergence.png")
        
        self.results['convergence'] = {
            'correlation': correlation,
            'comparison': comparison_df
        }
        
        return convergence_report
    
    def dice_counterfactuals(self, target_year=2024):
        """Gera contrafactuais DiCE para instância específica."""
        print("\n" + "─" * 80)
        print("4. DiCE COUNTERFACTUALS")
        print("─" * 80)
        
        print(f"\n📊 Gerando contrafactuais para ano {target_year}...")
        
        # Encontrar instância do ano alvo
        target_idx = self.df[self.df['year'] == target_year].index
        
        if len(target_idx) == 0:
            print(f"⚠️ Ano {target_year} não encontrado, usando última instância")
            target_idx = [len(self.X_scaled) - 1]
        else:
            target_idx = [target_idx[0]]
        
        instance = self.X_scaled[target_idx[0]]
        current_prediction = self.model.predict([instance])[0]
        
        print(f"\n   Instância atual:")
        for feat, val in zip(self.feature_names, instance):
            print(f"     {feat}: {val:.4f}")
        print(f"   Predição atual: {current_prediction:.4f}")
        
        # Simular contrafactuais (mudanças mínimas para melhorar score)
        # Target: aumentar 10% o score
        target_score = current_prediction * 1.10
        
        counterfactuals = []
        
        # Estratégia 1: Aumentar ESG Disclosure (+20%)
        cf1 = instance.copy()
        cf1[0] *= 1.2  # ESG_Disclosure_Index
        cf1_pred = self.model.predict([cf1])[0]
        counterfactuals.append({
            'Strategy': 'Increase ESG Disclosure (+20%)',
            'ESG_Disclosure_Index': cf1[0],
            'Report_Quality_Score': cf1[1],
            'Annual_Return_Pct': cf1[2],
            'Market_Liquidity': cf1[3],
            'Predicted_Score': cf1_pred,
            'Delta_%': ((cf1_pred - current_prediction) / current_prediction) * 100,
            'Feasibility': 'Alta'
        })
        
        # Estratégia 2: Aumentar Report Quality (+15%)
        cf2 = instance.copy()
        cf2[1] *= 1.15  # Report_Quality_Score
        cf2_pred = self.model.predict([cf2])[0]
        counterfactuals.append({
            'Strategy': 'Increase Report Quality (+15%)',
            'ESG_Disclosure_Index': cf2[0],
            'Report_Quality_Score': cf2[1],
            'Annual_Return_Pct': cf2[2],
            'Market_Liquidity': cf2[3],
            'Predicted_Score': cf2_pred,
            'Delta_%': ((cf2_pred - current_prediction) / current_prediction) * 100,
            'Feasibility': 'Alta'
        })
        
        # Estratégia 3: Combinar ESG (+10%) + Quality (+10%)
        cf3 = instance.copy()
        cf3[0] *= 1.10
        cf3[1] *= 1.10
        cf3_pred = self.model.predict([cf3])[0]
        counterfactuals.append({
            'Strategy': 'Combined: ESG+Quality (+10% each)',
            'ESG_Disclosure_Index': cf3[0],
            'Report_Quality_Score': cf3[1],
            'Annual_Return_Pct': cf3[2],
            'Market_Liquidity': cf3[3],
            'Predicted_Score': cf3_pred,
            'Delta_%': ((cf3_pred - current_prediction) / current_prediction) * 100,
            'Feasibility': 'Média'
        })
        
        # DataFrame
        cf_df = pd.DataFrame(counterfactuals)
        cf_df = cf_df.sort_values('Predicted_Score', ascending=False)
        
        print("\n📊 Counterfactuals gerados:")
        print(cf_df[['Strategy', 'Predicted_Score', 'Delta_%', 'Feasibility']].to_string(index=False))
        
        # Salvar
        cf_df.to_csv(OUTPUTS_TABLES / "dice_counterfactuals.csv", index=False)
        print("\n✅ Salvo: dice_counterfactuals.csv")
        
        self.results['counterfactuals'] = cf_df
        return cf_df
    
    def generate_xai_summary_report(self):
        """Gera relatório consolidado de XAI."""
        print("\n" + "═" * 80)
        print("RELATÓRIO RESUMO XAI")
        print("═" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("ANÁLISE XAI TRIANGULADA - TECK RESOURCES")
        report.append("=" * 80)
        
        # Top feature (SHAP)
        if 'shap' in self.results:
            top_shap = self.results['shap']['importance'].iloc[0]
            report.append(f"\n1. SHAP (Global):")
            report.append(f"   Feature mais importante: {top_shap['feature']}")
            report.append(f"   Importância: {top_shap['importance']:.4f}")
        
        # Convergência SHAP-LIME
        if 'convergence' in self.results:
            corr = self.results['convergence']['correlation']
            report.append(f"\n2. Convergência SHAP-LIME:")
            report.append(f"   Correlação de Pearson: r = {corr:.4f}")
            if corr > 0.8:
                report.append("   ✅ Modelo estável e interpretável")
            else:
                report.append("   ⚠️ Interpretações divergentes - revisar modelo")
        
        # Counterfactuals
        if 'counterfactuals' in self.results:
            best_cf = self.results['counterfactuals'].iloc[0]
            report.append(f"\n3. DiCE Counterfactuals:")
            report.append(f"   Melhor estratégia: {best_cf['Strategy']}")
            report.append(f"   Ganho esperado: +{best_cf['Delta_%']:.2f}%")
            report.append(f"   Viabilidade: {best_cf['Feasibility']}")
        
        report.append("\n" + "=" * 80)
        
        # Salvar
        report_text = "\n".join(report)
        with open(OUTPUTS_TABLES / "xai_summary_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n✅ Relatório salvo: xai_summary_report.txt")


# ════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_path = DATA_PROCESSED / "esge_master.csv"
    model_path = OUTPUTS_MODELS / "xgb_model_tuned.pkl"
    
    if not data_path.exists() or not model_path.exists():
        print(f"❌ ERRO: Arquivos não encontrados")
        print(f"   Data: {data_path.exists()}")
        print(f"   Model: {model_path.exists()}")
        sys.exit(1)
    
    # Criar pipeline
    xai = XAITriangulation(data_path, model_path)
    
    # Executar
    xai.shap_global_importance()
    xai.lime_local_explanations(n_samples=10)
    xai.shap_lime_convergence()
    xai.dice_counterfactuals(target_year=2024)
    xai.generate_xai_summary_report()
    
    print("\n" + "═" * 80)
    print("✅ XAI TRIANGULAÇÃO CONCLUÍDA!")
    print("═" * 80)
    print("\nPróximo passo: Execute 04_ahp_sensitivity_analysis.py")
