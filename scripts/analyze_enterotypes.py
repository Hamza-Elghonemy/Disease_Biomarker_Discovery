
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import os

# Configuration
DATA_DIR = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/data'
RESULTS_DIR = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/results/enterotypes'
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

def load_excel_sheet(filename, sheet_name):
    path = os.path.join(DATA_DIR, filename)
    print(f"Loading {sheet_name} from {filename}...")
    return pd.read_excel(path, sheet_name=sheet_name)

def determine_enterotypes(df_genus):
    print("Determining Enterotypes...")
    
    sample_cols = [c for c in df_genus.columns if isinstance(c, str) and (c.startswith('D') or c.startswith('H') or c.startswith('P')) and len(c) > 3]
    
    bact_row = None
    prev_row = None
    
    for col in df_genus.select_dtypes(include=['object']).columns:
        b_match = df_genus[df_genus[col].astype(str).str.contains('Bacteroides', case=False, na=False)]
        p_match = df_genus[df_genus[col].astype(str).str.contains('Prevotella', case=False, na=False)]
        
        if not b_match.empty:
            bact_row = b_match.index[0]
            if len(b_match) > 1:
                exact = df_genus[df_genus[col].astype(str).str.strip() == 'Bacteroides']
                if not exact.empty:
                    bact_row = exact.index[0]
            
        if not p_match.empty:
            prev_row = p_match.index[0]
            if len(p_match) > 1:
                exact = df_genus[df_genus[col].astype(str).str.strip() == 'Prevotella']
                if not exact.empty:
                    prev_row = exact.index[0]
        
        if bact_row is not None and prev_row is not None:
            break
            
    if bact_row is None or prev_row is None:
        raise ValueError("Could not find Bacteroides or Prevotella rows in genus sheet.")
        
    print(f"Bacteroides row index: {bact_row}")
    print(f"Prevotella row index: {prev_row}")
    
    bact_vals = df_genus.loc[bact_row, sample_cols].astype(float)
    prev_vals = df_genus.loc[prev_row, sample_cols].astype(float)
    
    epsilon = 1e-6
    ratios = bact_vals / (prev_vals + epsilon)
    
    labels = pd.Series(index=sample_cols, data=-1)
    labels[ratios >= 2.0] = 0  # Bacteroides-dominant
    labels[ratios <= 0.5] = 1  # Prevotella-dominant
    
    clean_labels = labels[labels != -1]
    
    print(f"Total samples: {len(sample_cols)}")
    print(f"Classified samples: {len(clean_labels)}")
    print(f"  Bacteroides (0): {sum(clean_labels==0)}")
    print(f"  Prevotella (1): {sum(clean_labels==1)}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(np.log10(ratios + epsilon), kde=True, bins=20)
    plt.axvline(np.log10(2.0), color='r', linestyle='--', label='Bacteroides Threshold')
    plt.axvline(np.log10(0.5), color='g', linestyle='--', label='Prevotella Threshold')
    plt.title('Distribution of Bacteroides/Prevotella Ratios (Log10)')
    plt.xlabel('Log10(Ratio)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'enterotype_ratios.png'))
    plt.close()
    
    return clean_labels

def run_ml_pipeline(labels):
    print("Running ML Pipeline using KO data...")
    df_ko = load_excel_sheet('abundance_tables.xlsx', 'KO')
    
    ko_id_col = None
    for c in df_ko.columns:
        if 'KO' in str(c) and 'phylum' not in str(c):
            ko_id_col = c
            break
    
    if ko_id_col is None:
        ko_id_col = df_ko.columns[0]
        
    df_ko = df_ko.set_index(ko_id_col)
    
    common_samples = [s for s in labels.index if s in df_ko.columns]
    
    y = labels[common_samples].values
    X = df_ko[common_samples].T.values
    X = np.nan_to_num(X)
    
    print(f"ML Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print(f"Original Training Distribution: {np.bincount(y_train.astype(int))}")
    
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled Training Distribution: {np.bincount(y_train_resampled.astype(int))}")
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42) 
    clf.fit(X_train_resampled, y_train_resampled)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, y_pred, target_names=['Bacteroides', 'Prevotella']))
    
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: Predicting Enterotype from Functions')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(RESULTS_DIR, 'roc_enterotype.png'))
        plt.close()
        
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    
    top_kos = df_ko.index[indices]
    top_scores = importances[indices]
    
    try:
        diff_df = load_excel_sheet('differential_analysis_results.xlsx', 'genus-specific KOs') 
        ko_map = {}
        if 'KO' in diff_df.columns and 'KO name' in diff_df.columns:
             ko_map = dict(zip(diff_df['KO'], diff_df['KO name']))
             
        top_names = [ko_map.get(k, k) for k in top_kos]
    except:
        top_names = top_kos
        
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_scores, y=top_names, palette='viridis')
    plt.title('Top 20 Discriminative Functional Features (KOs)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'top_features_enterotype.png'))
    plt.close()

if __name__ == "__main__":
    df_genus = load_excel_sheet('abundance_tables.xlsx', 'genus')
    labels = determine_enterotypes(df_genus)
    run_ml_pipeline(labels)
