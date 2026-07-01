"""
════════════════════════════════════════════════════════════════════════════
SCRIPT 4: AHP-GAUSSIANO COM ANÁLISE DE SENSIBILIDADE
════════════════════════════════════════════════════════════════════════════

Análise de sensibilidade completa do AHP-Gaussiano:
- Simulação Monte Carlo com múltiplos níveis de ruído (σ = 0.05, 0.10, 0.15, 0.20)
- Análise de rank reversal (estabilidade de ordenação)
- Distribuição de CR (Consistency Ratio)
- P(CR < 0.10) para cada nível de σ
- Comparação visual de pesos sob diferentes incertezas
- Validação de robustez probabilística

Referências:
- Santos et al. (2023): AHP-Gaussiano Monte Carlo
- Saaty (1980): AHP clássico e CR

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
from scipy.linalg import eig

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(r"C:\Users\user\Documents\_MBA_Data_Science_Analytics\00 - Temas TCC\teck-esge-xai")
OUTPUTS_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUTS_FIGURES = BASE_DIR / "outputs" / "figures"

OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

class AHPGaussianSensitivity:
    """Análise de sensibilidade AHP-Gaussiano."""
    
    def __init__(self):
        """Inicializa com matriz paritária base."""
        print("═" * 80)
        print("AHP-GAUSSIANO COM ANÁLISE DE SENSIBILIDADE")
        print("═" * 80)
        
        # Critérios ESGE
        self.criteria = ['Environmental (E)', 'Social (S)', 'Governance (G)', 'Economic (Ec)']
        self.n = len(self.criteria)
        
        # Matriz paritária base (julgamentos a priori)
        # E > S > G > Ec (based on mining sector priorities)
        self.pairwise_base = np.array([
            [1.0,   3.0,  5.0,  7.0],   # Environmental
            [1/3.0, 1.0,  3.0,  5.0],   # Social
            [1/5.0, 1/3.0, 1.0, 3.0],   # Governance
            [1/7.0, 1/5.0, 1/3.0, 1.0]  # Economic
        ])
        
        print(f"\n✅ Critérios: {self.criteria}")
        print(f"   Matriz paritária base ({self.n}x{self.n}):")
        print(self.pairwise_base)
        
        # Resultados
        self.results = {}
    
    def compute_consistency_ratio(self, matrix):
        """Calcula Consistency Ratio (CR) de Saaty."""
        # Autovalor máximo
        eigenvalues, _ = eig(matrix)
        lambda_max = max(eigenvalues.real)
        
        # Consistency Index
        CI = (lambda_max - self.n) / (self.n - 1)
        
        # Random Index (Saaty 1980)
        RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
        
        # Consistency Ratio
        CR = CI / RI.get(self.n, 1.0)
        
        return CR
    
    def monte_carlo_simulation(self, n_simulations=10000, noise_std=0.1):
        """Simulação Monte Carlo com ruído gaussiano."""
        weights_all = []
        cr_all = []
        
        for _ in range(n_simulations):
            # Ruído gaussiano N(1.0, σ)
            noise = np.random.normal(1.0, noise_std, self.pairwise_base.shape)
            noisy_matrix = self.pairwise_base * noise
            
            # Forçar reciprocidade: a_ji = 1/a_ij
            for i in range(self.n):
                for j in range(i+1, self.n):
                    noisy_matrix[j, i] = 1.0 / noisy_matrix[i, j]
            
            # Calcular pesos (autovetor principal)
            eigenvalues, eigenvectors = eig(noisy_matrix)
            idx_max = np.argmax(eigenvalues.real)
            weights = eigenvectors[:, idx_max].real
            weights = weights / weights.sum()  # Normalizar
            
            weights_all.append(weights)
            
            # CR
            cr = self.compute_consistency_ratio(noisy_matrix)
            cr_all.append(cr)
        
        # Converter para array
        weights_array = np.array(weights_all)
        cr_array = np.array(cr_all)
        
        return weights_array, cr_array
    
    def sensitivity_analysis(self):
        """Análise de sensibilidade para diferentes níveis de σ."""
        print("\n" + "─" * 80)
        print("1. ANÁLISE DE SENSIBILIDADE (Múltiplos níveis de σ)")
        print("─" * 80)
        
        sigma_levels = [0.05, 0.10, 0.15, 0.20]
        n_simulations = 10000
        
        results_by_sigma = []
        
        for sigma in sigma_levels:
            print(f"\n📊 Simulando com σ = {sigma:.2f}...")
            
            weights, cr_values = self.monte_carlo_simulation(n_simulations, sigma)
            
            # Estatísticas dos pesos (variável "result" evita shadowing de scipy.stats)
            result = {
                'Sigma': sigma,
                'P(CR<0.10)_%': (np.sum(cr_values < 0.10) / len(cr_values)) * 100,
                'Mean_CR': cr_values.mean(),
                'Std_CR': cr_values.std()
            }
            
            # Para cada critério
            for i, criterion in enumerate(self.criteria):
                result[f'{criterion}_Mean'] = weights[:, i].mean()
                result[f'{criterion}_Std'] = weights[:, i].std()
                result[f'{criterion}_CV%'] = (weights[:, i].std() / weights[:, i].mean()) * 100

            results_by_sigma.append(result)

            # Exibir
            print(f"   P(CR < 0.10) = {result['P(CR<0.10)_%']:.2f}%")
            print(f"   Mean CR = {result['Mean_CR']:.4f}")
            
            # Armazenar para plots
            self.results[f'sigma_{sigma}'] = {
                'weights': weights,
                'cr': cr_values
            }
        
        # DataFrame
        df_sensitivity = pd.DataFrame(results_by_sigma)
        df_sensitivity.to_csv(OUTPUTS_TABLES / "ahp_sensitivity_analysis.csv", index=False)
        print(f"\n✅ Salvo: ahp_sensitivity_analysis.csv")
        
        self.results['sensitivity'] = df_sensitivity
        return df_sensitivity
    
    def rank_reversal_analysis(self):
        """Testa rank reversal (mudança de ordenação dos critérios)."""
        print("\n" + "─" * 80)
        print("2. ANÁLISE DE RANK REVERSAL")
        print("─" * 80)
        
        sigma_levels = [0.05, 0.10, 0.15, 0.20]
        
        rank_changes = []
        
        for sigma in sigma_levels:
            weights = self.results[f'sigma_{sigma}']['weights']
            
            # Calcular ranking médio
            mean_weights = weights.mean(axis=0)
            base_ranking = np.argsort(-mean_weights)  # Ordem decrescente
            
            # Contar quantas simulações tiveram ranking diferente
            reversals = 0
            for sim_weights in weights:
                sim_ranking = np.argsort(-sim_weights)
                if not np.array_equal(base_ranking, sim_ranking):
                    reversals += 1
            
            reversal_rate = (reversals / len(weights)) * 100
            
            rank_changes.append({
                'Sigma': sigma,
                'Reversal_Rate_%': reversal_rate,
                'Base_Ranking': ' > '.join([self.criteria[i] for i in base_ranking])
            })
            
            print(f"\nσ = {sigma:.2f}:")
            print(f"  Ranking base: {rank_changes[-1]['Base_Ranking']}")
            print(f"  Taxa de reversão: {reversal_rate:.2f}%")
        
        # Salvar
        df_rank = pd.DataFrame(rank_changes)
        df_rank.to_csv(OUTPUTS_TABLES / "rank_reversal_analysis.csv", index=False)
        print(f"\n✅ Salvo: rank_reversal_analysis.csv")
        
        return df_rank
    
    def plot_weights_distribution(self):
        """Plota distribuição de pesos para diferentes σ."""
        print("\n" + "─" * 80)
        print("3. VISUALIZAÇÃO - DISTRIBUIÇÃO DE PESOS")
        print("─" * 80)
        
        sigma_levels = [0.05, 0.10, 0.15, 0.20]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, sigma in enumerate(sigma_levels):
            weights = self.results[f'sigma_{sigma}']['weights']
            
            # Boxplot
            df_weights = pd.DataFrame(weights, columns=self.criteria)
            
            axes[idx].boxplot([df_weights[c] for c in self.criteria], labels=self.criteria)
            axes[idx].set_title(f'σ = {sigma:.2f}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Weight', fontsize=11)
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Distribuição de Pesos AHP-Gaussiano (Diferentes níveis de σ)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / "ahp_weights_distribution_by_sigma.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Salvo: ahp_weights_distribution_by_sigma.png")
    
    def plot_cr_distribution(self):
        """Plota distribuição de CR."""
        print("\n" + "─" * 80)
        print("4. VISUALIZAÇÃO - DISTRIBUIÇÃO DE CR")
        print("─" * 80)
        
        sigma_levels = [0.05, 0.10, 0.15, 0.20]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, sigma in enumerate(sigma_levels):
            cr_values = self.results[f'sigma_{sigma}']['cr']
            
            axes[idx].hist(cr_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[idx].axvline(0.10, color='red', linestyle='--', linewidth=2, label='CR = 0.10 (limiar)')
            
            p_consistent = (np.sum(cr_values < 0.10) / len(cr_values)) * 100
            
            axes[idx].set_title(f'σ = {sigma:.2f} | P(CR<0.10) = {p_consistent:.1f}%',
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Consistency Ratio (CR)', fontsize=11)
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].legend(loc='best')
            axes[idx].grid(alpha=0.3)
        
        plt.suptitle('Distribuição do Consistency Ratio (CR) - AHP-Gaussiano',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / "cr_distribution_by_sigma.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Salvo: cr_distribution_by_sigma.png")
    
    def generate_final_ahp_weights(self, sigma=0.10):
        """Gera tabela final de pesos para o TCC (σ padrão = 0.10)."""
        print("\n" + "─" * 80)
        print(f"5. TABELA FINAL DE PESOS (σ = {sigma:.2f})")
        print("─" * 80)
        
        weights = self.results[f'sigma_{sigma}']['weights']
        
        # Estatísticas
        ahp_table = []
        for i, criterion in enumerate(self.criteria):
            ahp_table.append({
                'Criterion': criterion,
                'Mean': weights[:, i].mean(),
                'Std': weights[:, i].std(),
                'CI_95_Lower': np.percentile(weights[:, i], 2.5),
                'CI_95_Upper': np.percentile(weights[:, i], 97.5),
                'CV_%': (weights[:, i].std() / weights[:, i].mean()) * 100
            })
        
        df_ahp = pd.DataFrame(ahp_table)
        df_ahp = df_ahp.sort_values('Mean', ascending=False)
        
        print("\n" + df_ahp.to_string(index=False))
        
        # Salvar (SOBRESCREVER o antigo ahp_weights.csv)
        df_ahp.to_csv(OUTPUTS_TABLES / "ahp_weights.csv", index=False)
        print(f"\n✅ Salvo: ahp_weights.csv (ATUALIZADO)")
        
        # Gráfico de barras
        plt.figure(figsize=(10, 6))
        x = np.arange(len(df_ahp))
        plt.bar(x, df_ahp['Mean'], yerr=df_ahp['Std'], capsize=5, alpha=0.8, color='teal')
        plt.xlabel('Critério ESGE', fontsize=12)
        plt.ylabel('Peso (μ ± σ)', fontsize=12)
        plt.title(f'Pesos AHP-Gaussiano (σ = {sigma:.2f}, N = 10.000 simulações)',
                 fontsize=14, fontweight='bold')
        plt.xticks(x, df_ahp['Criterion'], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / "ahp_weights.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Salvo: ahp_weights.png (ATUALIZADO)")
        
        return df_ahp


# ════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Criar análise
    ahp = AHPGaussianSensitivity()
    
    # Executar
    ahp.sensitivity_analysis()
    ahp.rank_reversal_analysis()
    ahp.plot_weights_distribution()
    ahp.plot_cr_distribution()
    ahp.generate_final_ahp_weights(sigma=0.10)
    
    print("\n" + "═" * 80)
    print("✅ ANÁLISE DE SENSIBILIDADE AHP CONCLUÍDA!")
    print("═" * 80)
    print("\nPróximo passo: Execute 05_financial_benchmarking_tsx.py")
