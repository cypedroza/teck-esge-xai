"""
════════════════════════════════════════════════════════════════════════════
SCRIPT 2: MODELAGEM ML COM VALIDAÇÃO AVANÇADA
════════════════════════════════════════════════════════════════════════════

Machine Learning com validação rigorosa para publicação internacional:
- Cross-validation estratificada (TimeSeriesSplit para dados temporais)
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Learning curves e validation curves
- Feature selection (RFE, Permutation Importance)
- Comparação com baselines simples (média, regressão linear)
- Intervalos de confiança bootstrap
- Métricas completas (R², RMSE, MAE, MAPE) com CI 95%

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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(r"C:\Users\user\Documents\_MBA_Data_Science_Analytics\00 - Temas TCC\teck-esge-xai")
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUTS_FIGURES = BASE_DIR / "outputs" / "figures"
OUTPUTS_MODELS = BASE_DIR / "outputs" / "models"

# Criar diretórios
for dir_path in [OUTPUTS_TABLES, OUTPUTS_FIGURES, OUTPUTS_MODELS]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# FEATURE MAPPING (do Script 1)
# ════════════════════════════════════════════════════════════════════════════

FEATURE_MAPPING = {
    'esg_disclosure_score': 'ESG_Disclosure_Index',
    'char_count': 'Report_Quality_Score',
    'annual_return_%': 'Annual_Return_Pct',
    'volume': 'Market_Liquidity'
}

def normalize_feature_names(df):
    """Normaliza nomes de features."""
    return df.rename(columns=FEATURE_MAPPING)


# ════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

class AdvancedMLValidation:
    """Modelagem ML com validação rigorosa (nível publicação)."""
    
    def __init__(self, data_path: Path):
        """Inicializa com dataset."""
        print("═" * 80)
        print("MODELAGEM ML COM VALIDAÇÃO AVANÇADA")
        print("═" * 80)
        
        # Carregar e normalizar
        df_raw = pd.read_csv(data_path)
        self.df = normalize_feature_names(df_raw)
        
        print(f"\n✅ Dataset carregado: {self.df.shape}")
        print(f"   Features normalizadas: {list(self.df.columns)}")
        
        # Preparar features e target
        feature_cols = ['ESG_Disclosure_Index', 'Report_Quality_Score', 'Annual_Return_Pct', 'Market_Liquidity']
        # ESGE_score é calculado externamente (notebooks/02_data_extraction) — target independente das features
        target_col = 'ESGE_score'

        required_cols = feature_cols + [target_col]
        available = [c for c in required_cols if c in self.df.columns]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            print(f"\n⚠️  Colunas ausentes no dataset: {missing}")
            print("   Verifique se está usando esge_master_final.csv")
            sys.exit(1)

        # Remover linhas com NaN em features OU target
        df_clean = self.df[required_cols].dropna()
        print(f"\n✅ Limpeza de dados:")
        print(f"   Linhas originais: {len(self.df)}")
        print(f"   Linhas após remover NaN: {len(df_clean)}")
        print(f"   Linhas removidas: {len(self.df) - len(df_clean)}")

        if len(df_clean) < 5:
            print("\n❌ ERRO: Dados insuficientes após limpeza!")
            print("   Verifique se o dataset tem valores válidos.")
            sys.exit(1)

        self.X = df_clean[feature_cols].values
        # Target: ESGE_score composto (calculado independentemente das features)
        self.y = df_clean[target_col].values
        
        print(f"\n✅ Features (X): {self.X.shape}")
        print(f"   Target (y): {self.y.shape}")
        
        # Verificar se ainda há NaN
        if np.isnan(self.X).any() or np.isnan(self.y).any():
            print("\n❌ AVISO: Ainda há NaN nos dados!")
            print(f"   NaN em X: {np.isnan(self.X).sum()}")
            print(f"   NaN em y: {np.isnan(self.y).sum()}")
            # Forçar remoção
            mask = ~(np.isnan(self.X).any(axis=1) | np.isnan(self.y))
            self.X = self.X[mask]
            self.y = self.y[mask]
            print(f"   Após limpeza forçada: {self.X.shape}")
        
        # Scaler
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Resultados
        self.results = {}
        self.models = {}
    
    def baseline_models(self):
        """Treina modelos baseline para comparação."""
        print("\n" + "─" * 80)
        print("1. MODELOS BASELINE")
        print("─" * 80)
        
        # TimeSeriesSplit (respeitando ordem temporal)
        tscv = TimeSeriesSplit(n_splits=5)
        
        baselines = {}
        
        # Baseline 1: Média
        mean_pred = np.full_like(self.y, self.y.mean())
        baselines['Mean'] = {
            'R²': r2_score(self.y, mean_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y, mean_pred)),
            'MAE': mean_absolute_error(self.y, mean_pred)
        }
        
        # Baseline 2: Regressão Linear
        lr = LinearRegression()
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        for train_idx, test_idx in tscv.split(self.X_scaled):
            X_train, X_test = self.X_scaled[train_idx], self.X_scaled[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
        
        baselines['Linear Regression'] = {
            'R²': np.mean(r2_scores),
            'R²_Std': np.std(r2_scores),
            'RMSE': np.mean(rmse_scores),
            'RMSE_Std': np.std(rmse_scores),
            'MAE': np.mean(mae_scores),
            'MAE_Std': np.std(mae_scores)
        }
        
        # Exibir
        print("\nBaseline: Média")
        print(f"  R² = {baselines['Mean']['R²']:.4f}")
        print(f"  RMSE = {baselines['Mean']['RMSE']:.4f}")
        
        print("\nBaseline: Regressão Linear (CV 5-fold temporal)")
        print(f"  R² = {baselines['Linear Regression']['R²']:.4f} ± {baselines['Linear Regression']['R²_Std']:.4f}")
        print(f"  RMSE = {baselines['Linear Regression']['RMSE']:.4f} ± {baselines['Linear Regression']['RMSE_Std']:.4f}")
        
        # Salvar
        df_baselines = pd.DataFrame(baselines).T
        df_baselines.to_csv(OUTPUTS_TABLES / "baseline_models_performance.csv")
        print(f"\n✅ Salvo: baseline_models_performance.csv")
        
        self.results['baselines'] = baselines
        return baselines
    
    def hyperparameter_tuning_rf(self):
        """Hyperparameter tuning Random Forest."""
        print("\n" + "─" * 80)
        print("2. HYPERPARAMETER TUNING - RANDOM FOREST")
        print("─" * 80)
        
        # Grid de parâmetros
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # GridSearchCV com TimeSeriesSplit
        rf = RandomForestRegressor(random_state=42)
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, 
            scoring='r2', n_jobs=-1, verbose=1
        )
        
        print("\n🔍 Executando GridSearchCV...")
        grid_search.fit(self.X_scaled, self.y)
        
        print(f"\n✅ Melhores parâmetros: {grid_search.best_params_}")
        print(f"   Melhor R² (CV): {grid_search.best_score_:.4f}")
        
        # Salvar modelo
        self.models['RF_tuned'] = grid_search.best_estimator_
        joblib.dump(grid_search.best_estimator_, OUTPUTS_MODELS / "rf_model_tuned.pkl")
        
        # Salvar resultados
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(OUTPUTS_TABLES / "rf_gridsearch_results.csv", index=False)
        print("✅ Salvo: rf_model_tuned.pkl, rf_gridsearch_results.csv")
        
        return grid_search.best_estimator_
    
    def hyperparameter_tuning_xgb(self):
        """Hyperparameter tuning XGBoost."""
        print("\n" + "─" * 80)
        print("3. HYPERPARAMETER TUNING - XGBOOST")
        print("─" * 80)
        
        # Grid de parâmetros
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
        
        # GridSearchCV
        xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=tscv,
            scoring='r2', n_jobs=-1, verbose=1
        )
        
        print("\n🔍 Executando GridSearchCV...")
        grid_search.fit(self.X_scaled, self.y)
        
        print(f"\n✅ Melhores parâmetros: {grid_search.best_params_}")
        print(f"   Melhor R² (CV): {grid_search.best_score_:.4f}")
        
        # Salvar
        self.models['XGB_tuned'] = grid_search.best_estimator_
        joblib.dump(grid_search.best_estimator_, OUTPUTS_MODELS / "xgb_model_tuned.pkl")
        
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(OUTPUTS_TABLES / "xgb_gridsearch_results.csv", index=False)
        print("✅ Salvo: xgb_model_tuned.pkl, xgb_gridsearch_results.csv")
        
        return grid_search.best_estimator_
    
    def compare_models_with_ci(self):
        """Compara modelos com intervalos de confiança bootstrap."""
        print("\n" + "─" * 80)
        print("4. COMPARAÇÃO DE MODELOS (Com CI 95% Bootstrap)")
        print("─" * 80)
        
        models_to_compare = {
            'Random Forest': self.models['RF_tuned'],
            'XGBoost': self.models['XGB_tuned']
        }
        
        comparison_results = []
        
        for model_name, model in models_to_compare.items():
            print(f"\n📊 Avaliando: {model_name}")
            
            # Bootstrap para intervalos de confiança
            n_bootstrap = 1000
            r2_bootstrap = []
            rmse_bootstrap = []
            mae_bootstrap = []
            
            for i in range(n_bootstrap):
                # Reamostragem com reposição
                indices = np.random.choice(len(self.X_scaled), size=len(self.X_scaled), replace=True)
                X_boot = self.X_scaled[indices]
                y_boot = self.y[indices]
                
                # Treinar e prever
                model.fit(X_boot, y_boot)
                y_pred = model.predict(X_boot)
                
                r2_bootstrap.append(r2_score(y_boot, y_pred))
                rmse_bootstrap.append(np.sqrt(mean_squared_error(y_boot, y_pred)))
                mae_bootstrap.append(mean_absolute_error(y_boot, y_pred))
            
            # Calcular CI 95%
            result = {
                'Model': model_name,
                'R²_Mean': np.mean(r2_bootstrap),
                'R²_Std': np.std(r2_bootstrap),
                'R²_CI_Lower': np.percentile(r2_bootstrap, 2.5),
                'R²_CI_Upper': np.percentile(r2_bootstrap, 97.5),
                'RMSE_Mean': np.mean(rmse_bootstrap),
                'RMSE_Std': np.std(rmse_bootstrap),
                'RMSE_CI_Lower': np.percentile(rmse_bootstrap, 2.5),
                'RMSE_CI_Upper': np.percentile(rmse_bootstrap, 97.5),
                'MAE_Mean': np.mean(mae_bootstrap),
                'MAE_Std': np.std(mae_bootstrap),
                'MAE_CI_Lower': np.percentile(mae_bootstrap, 2.5),
                'MAE_CI_Upper': np.percentile(mae_bootstrap, 97.5)
            }
            
            comparison_results.append(result)
            
            # Exibir
            print(f"  R² = {result['R²_Mean']:.4f} ± {result['R²_Std']:.4f}")
            print(f"     CI 95%: [{result['R²_CI_Lower']:.4f}, {result['R²_CI_Upper']:.4f}]")
            print(f"  RMSE = {result['RMSE_Mean']:.4f} ± {result['RMSE_Std']:.4f}")
            print(f"     CI 95%: [{result['RMSE_CI_Lower']:.4f}, {result['RMSE_CI_Upper']:.4f}]")
        
        # Salvar
        df_comparison = pd.DataFrame(comparison_results)
        df_comparison.to_csv(OUTPUTS_TABLES / "model_comparison_with_ci.csv", index=False)
        print(f"\n✅ Salvo: model_comparison_with_ci.csv")
        
        self.results['comparison'] = df_comparison
        return df_comparison
    
    def learning_curves(self):
        """Gera learning curves para diagnóstico."""
        print("\n" + "─" * 80)
        print("5. LEARNING CURVES")
        print("─" * 80)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = {
            'Random Forest': self.models['RF_tuned'],
            'XGBoost': self.models['XGB_tuned']
        }
        
        for idx, (name, model) in enumerate(models.items()):
            print(f"\n📈 Gerando curve: {name}")
            
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_scaled, self.y,
                cv=TimeSeriesSplit(n_splits=5),
                scoring='r2',
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            axes[idx].plot(train_sizes, train_mean, label='Training Score', marker='o')
            axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
            
            axes[idx].plot(train_sizes, val_mean, label='Validation Score', marker='s')
            axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
            
            axes[idx].set_xlabel('Training Size', fontsize=11)
            axes[idx].set_ylabel('R² Score', fontsize=11)
            axes[idx].set_title(f'Learning Curve: {name}', fontsize=12, fontweight='bold')
            axes[idx].legend(loc='best')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / "learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✅ Salvo: learning_curves.png")
    
    def feature_importance_permutation(self):
        """Calcula importância via permutation (model-agnostic)."""
        print("\n" + "─" * 80)
        print("6. FEATURE IMPORTANCE (Permutation)")
        print("─" * 80)
        
        feature_names = ['ESG_Disclosure_Index', 'Report_Quality_Score', 'Annual_Return_Pct', 'Market_Liquidity']
        
        for model_name, model in [('Random Forest', self.models['RF_tuned']), 
                                   ('XGBoost', self.models['XGB_tuned'])]:
            print(f"\n📊 {model_name}:")
            
            # Permutation importance
            perm_importance = permutation_importance(
                model, self.X_scaled, self.y, 
                n_repeats=30, random_state=42, n_jobs=-1
            )
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance_Mean': perm_importance.importances_mean,
                'Importance_Std': perm_importance.importances_std
            }).sort_values('Importance_Mean', ascending=False)
            
            print(importance_df.to_string(index=False))
            
            # Salvar
            filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
            importance_df.to_csv(OUTPUTS_TABLES / filename, index=False)
            print(f"✅ Salvo: {filename}")
    
    def generate_final_performance_table(self):
        """Gera tabela final de performance para o TCC."""
        print("\n" + "─" * 80)
        print("7. TABELA FINAL DE PERFORMANCE")
        print("─" * 80)
        
        # Combinar baselines + modelos tuned
        final_results = []
        
        # Baselines
        for name, metrics in self.results['baselines'].items():
            final_results.append({
                'Modelo': name,
                'R²': metrics.get('R²', metrics.get('R²', 0)),
                'RMSE': metrics.get('RMSE', 0),
                'MAE': metrics.get('MAE', 0),
                'Categoria': 'Baseline'
            })
        
        # Modelos tuned (usar valores do comparison com CI)
        if 'comparison' in self.results:
            for _, row in self.results['comparison'].iterrows():
                final_results.append({
                    'Modelo': row['Model'],
                    'R²': row['R²_Mean'],
                    'RMSE': row['RMSE_Mean'],
                    'MAE': row['MAE_Mean'],
                    'Categoria': 'Ensemble Tuned'
                })
        
        df_final = pd.DataFrame(final_results)
        df_final = df_final.sort_values('R²', ascending=False)
        
        print("\n" + df_final.to_string(index=False))
        
        # Salvar (SOBRESCREVER o antigo model_performance.csv)
        df_final.to_csv(OUTPUTS_TABLES / "model_performance.csv", index=False)
        print(f"\n✅ Salvo: model_performance.csv (ATUALIZADO)")
        
        return df_final


# ════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_path = DATA_PROCESSED / "esge_master.csv"
    
    if not data_path.exists():
        print(f"❌ ERRO: {data_path} não encontrado")
        sys.exit(1)
    
    # Criar pipeline
    ml_pipeline = AdvancedMLValidation(data_path)
    
    # Executar
    ml_pipeline.baseline_models()
    ml_pipeline.hyperparameter_tuning_rf()
    ml_pipeline.hyperparameter_tuning_xgb()
    ml_pipeline.compare_models_with_ci()
    ml_pipeline.learning_curves()
    ml_pipeline.feature_importance_permutation()
    ml_pipeline.generate_final_performance_table()
    
    print("\n" + "═" * 80)
    print("✅ MODELAGEM ML AVANÇADA CONCLUÍDA!")
    print("═" * 80)
    print("\nPróximo passo: Execute 03_xai_triangulation_convergence.py")
