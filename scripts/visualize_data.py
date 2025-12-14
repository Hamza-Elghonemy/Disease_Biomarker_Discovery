
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_DIR = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/data'
RESULTS_DIR = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/results/figures'
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

def load_data(filename, sheet_name=None):
    path = os.path.join(DATA_DIR, filename)
    if filename.endswith('.csv'):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path, sheet_name=sheet_name)

def visualize_proteome():
    print("Visualizing Proteome...")
    df = load_data('crc_gut_proteome.csv')
    
    sample_ids = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    groups = ['Healthy' if 'H' in s else 'Patient' for s in sample_ids]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Group'] = groups
    pca_df['Sample'] = sample_ids
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Group', style='Group', s=100, palette='viridis')
    plt.title('PCA of Gut Proteome')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'proteome_pca.png'))
    plt.close()
    variances = data.var(axis=0)
    top_50_vars = variances.nlargest(50).index
    top_50_data = data[top_50_vars]
    
    group_colors = {'Healthy': '#1f77b4', 'Patient': '#ff7f0e'}
    row_colors = pd.Series(groups).map(group_colors)
    
    plt.figure(figsize=(12, 10))
    cg = sns.clustermap(top_50_data.T, 
                   col_colors=row_colors.values,
                   standard_scale=0,
                   cmap='vlag',
                   figsize=(12, 12),
                   xticklabels=False)
    cg.ax_heatmap.set_title("Top 50 Variable Proteins")
    plt.savefig(os.path.join(RESULTS_DIR, 'proteome_heatmap.png'))
    plt.close()

def visualize_differential_analysis():
    print("Visualizing Differential Analysis...")
    df = load_data('differential_analysis_results.xlsx', sheet_name='Bacillota-specific KOs')
    df.columns = df.columns.str.strip()
    df['nlog10_q'] = -np.log10(df['q-value'])
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='log ratio', y='nlog10_q', hue='higher in', palette='coolwarm', edgecolor='k')
    
    plt.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='q=0.05')
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(-1, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Volcano Plot: Bacillota-specific KOs')
    plt.xlabel('Log Ratio')
    plt.ylabel('-Log10 q-value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'volcano_bacillota_kos.png'))
    plt.close()
    df_path = load_data('differential_analysis_results.xlsx', sheet_name='Bacillota-specific pathways')
    if not df_path.empty:
        df_path.columns = df_path.columns.str.strip()
        top_path = df_path.sort_values(by='p-value').head(20)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_path, y='pathway name', x='log ratio', hue='higher in', palette='coolwarm', dodge=False)
        plt.title('Top Differential Pathways (Bacillota)')
        plt.xlabel('Log Ratio')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'barplot_bacillota_pathways.png'))
        plt.close()

def visualize_abundance():
    print("Visualizing Abundance...")
    df = load_data('abundance_tables.xlsx', sheet_name='phylum')
    
    sample_cols = [c for c in df.columns if c not in ['phylum', 'normalized abundance (no unassigned)']]
    df['mean_abundance'] = df[sample_cols].mean(axis=1)
    top_phyla = df.sort_values('mean_abundance', ascending=False).head(10)
    
    melted = top_phyla.melt(id_vars='phylum', value_vars=sample_cols, var_name='Sample', value_name='Abundance')
    
    plt.figure(figsize=(14, 8))
    top_phyla.set_index('phylum')[sample_cols].T.plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, 8))
    plt.title('Relative Abundance of Top 10 Phyla across Samples')
    plt.ylabel('Abundance')
    plt.xlabel('Sample')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'abundance_phylum_stacked.png'))
    plt.close()

def visualize_correlations():
    print("Visualizing Correlations...")
    df = load_data('correlations.xlsx', sheet_name='phyla vs phyla')
    
    sig_df = df[df['FDR'] < 0.05]
    
    if not sig_df.empty:
        plt.figure(figsize=(10, 8))
        pivot = sig_df.pivot(index='phylum 1', columns='phylum 2', values='rho')
        sns.heatmap(pivot, cmap='coolwarm', center=0, annot=False)
        plt.title('Significant Phyla-Phyla Correlations (rho)')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'correlation_phyla_heatmap.png'))
        plt.close()
    df_age = load_data('correlations.xlsx', sheet_name='phyla vs age')
    sig_age = df_age[df_age['FDR'] < 0.05]
    
    if not sig_age.empty:
         plt.figure(figsize=(10, 6))
         sns.barplot(data=sig_age, x='rho', y='phylum 1', palette='coolwarm')
         plt.title('Significant Correlations between Phyla and Age')
         plt.xlabel('Correlation Coefficient (Rho)')
         plt.tight_layout()
         plt.savefig(os.path.join(RESULTS_DIR, 'correlation_phyla_age.png'))
         plt.close()

if __name__ == "__main__":
    try:
        visualize_proteome()
    except Exception as e:
        print(f"Error in proteome: {e}")
        
    try:
        visualize_differential_analysis()
    except Exception as e:
        print(f"Error in differential: {e}")
        
    try:
        visualize_abundance()
    except Exception as e:
        print(f"Error in abundance: {e}")
        
    try:
        visualize_correlations()
    except Exception as e:
        print(f"Error in correlations: {e}")

    print("Visualization complete.")
