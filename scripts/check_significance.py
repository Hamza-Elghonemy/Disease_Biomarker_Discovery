
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA_FILE = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/data/crc_gut_proteome.csv'

def check_significance():
    df = pd.read_csv(DATA_FILE)
    
    h_samples = [c for c in df.iloc[:, 0] if 'H' in c]
    p_samples = [c for c in df.iloc[:, 0] if 'P' in c]
    
    print(f"H samples: {len(h_samples)}")
    print(f"P samples: {len(p_samples)}")
    
    data = df.iloc[:, 1:].values
    labels = [0 if 'H' in s else 1 for s in df.iloc[:, 0]]
    
    h_data = data[np.array(labels) == 0]
    p_data = data[np.array(labels) == 1]
    
    p_values = []
    features = df.columns[1:]
    
    for i in range(data.shape[1]):
        try:
            _, p = stats.ttest_ind(h_data[:, i], p_data[:, i], equal_var=False)
            p_values.append(p)
        except:
            p_values.append(1.0)
            
    p_values = np.array(p_values)
    p_values[np.isnan(p_values)] = 1.0
    
    reject, q_values, _, _ = multipletests(p_values, method='fdr_bh')
    
    sig_count = sum(reject)
    print(f"Number of significant features (FDR < 0.05): {sig_count}")
    
    indices = np.argsort(p_values)
    print("\nTop 10 features (uncorrected p-value):")
    for i in indices[:10]:
        print(f"{features[i]}: p={p_values[i]:.2e}")

    if sig_count < 10:
        print("\nWARNING: Very weak signal. Classification will be difficult.")

if __name__ == "__main__":
    check_significance()
