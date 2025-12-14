
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Configuration
DATA_DIR = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/data'
RESULTS_DIR = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/results/diet_health_model'
os.makedirs(RESULTS_DIR, exist_ok=True)

# KEGG Pathway Mapping
PATHWAY_MAP = {
    'map00500': 'Starch and sucrose metabolism',
    'map00531': 'Glycosaminoglycan degradation',
    'map00790': 'Folate biosynthesis',
    'map00540': 'Lipopolysaccharide biosynthesis',
    'map00290': 'Valine, leucine and isoleucine biosynthesis', # BCAA
    'map02030': 'Bacterial chemotaxis',
    'map00633': 'Nitrotoluene degradation',
    'map00350': 'Tyrosine metabolism',
    'map00440': 'Phosphonate and phosphinate metabolism',
    'map00791': 'Atrazine degradation',
    'map00670': 'One carbon pool by folate',
    'map02040': 'Flagellar assembly',
    'map01210': '2-Oxocarboxylic acid metabolism',
    'map00903': 'Limonene and pinene degradation'
}

def load_data():
    path = os.path.join(DATA_DIR, 'abundance_tables.xlsx')
    print("Loading abundance tables...")
    df_genus = pd.read_excel(path, sheet_name='genus')
    df_pathway = pd.read_excel(path, sheet_name='pathway')
    return df_genus, df_pathway

def determine_diet_labels(df_genus):
    print("Determining Diet Labels based on Enterotypes...")
    
    sample_cols = [c for c in df_genus.columns if isinstance(c, str) and (c.startswith('D') or c.startswith('H') or c.startswith('P')) and len(c) > 3]
    bact_row = None
    prev_row = None
    
    for col in df_genus.select_dtypes(include=['object']).columns:
        b_match = df_genus[df_genus[col].astype(str).str.contains('Bacteroides', case=False, na=False)]
        p_match = df_genus[df_genus[col].astype(str).str.contains('Prevotella', case=False, na=False)]
        
        if not b_match.empty:
            bact_row = b_match.index[0]
            exact = df_genus[df_genus[col].astype(str).str.strip() == 'Bacteroides']
            if not exact.empty:
                bact_row = exact.index[0]

        if not p_match.empty:
            prev_row = p_match.index[0]
            exact = df_genus[df_genus[col].astype(str).str.strip() == 'Prevotella']
            if not exact.empty:
                prev_row = exact.index[0]
                
        if bact_row is not None and prev_row is not None:
            break
            
    if bact_row is None or prev_row is None:
        raise ValueError("Could not find Bacteroides or Prevotella rows.")

    bact_vals = df_genus.loc[bact_row, sample_cols].astype(float)
    prev_vals = df_genus.loc[prev_row, sample_cols].astype(float)
    
    epsilon = 1e-6
    ratios = bact_vals / (prev_vals + epsilon)
    
    labels = pd.Series(index=sample_cols, data=-1)
    labels[ratios >= 2.0] = 0  # Western (Bacteroides dominant)
    labels[ratios <= 0.5] = 1  # Plant-Based (Prevotella dominant) 
    
    clean_labels = labels[labels != -1]
    
    print(f"Total samples: {len(sample_cols)}")
    print(f"Labeled samples: {len(clean_labels)}")
    print(f"  Western Diet (Bacteroides): {sum(clean_labels==0)}")
    print(f"  Plant-Based Diet (Prevotella): {sum(clean_labels==1)}")
    
    return clean_labels

def train_diet_model(df_pathway, labels):
    print("\nTRAINING DIET PATTERN MODEL...")
    
    if pd.api.types.is_integer_dtype(df_pathway.index):
        id_col = df_pathway.columns[0]
        df_pathway = df_pathway.set_index(id_col)
    common_samples = [s for s in labels.index if s in df_pathway.columns]
    y = labels[common_samples].values
    X = df_pathway[common_samples].T.values
    X = np.nan_to_num(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_res, y_train_res)
    
    score = clf.score(X_test, y_test)
    print(f"Model Accuracy: {score:.4f}")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = df_pathway.index[indices]
    
    print("Top Predictive Pathways for Diet:")
    for i, idx in enumerate(indices):
        pid = df_pathway.index[idx]
        name = PATHWAY_MAP.get(pid, pid)
        print(f"  {i+1}. {name} ({importances[idx]:.4f})")
        
    return clf, X_test, y_test, df_pathway.index

def simulate_user_app(clf, x_sample, sample_id, pathway_index, df_pathway, full_samples):
    probs = clf.predict_proba(x_sample.reshape(1, -1))[0]
    western_score = probs[0] * 100
    plant_score = probs[1] * 100
    
    print(f"\n--- USER REPORT: {sample_id} ---")
    if western_score > plant_score:
        print(f"Dietary Score: Your microbiome suggests a {western_score:.1f}% match with a Western Diet profile.")
    else:
        print(f"Dietary Score: Your microbiome suggests a {plant_score:.1f}% match with a Plant-Based Diet profile.")
        
    risk_flags = []
    def get_abundance(pid):
        try:
            loc = list(pathway_index).index(pid)
            return x_sample[loc]
        except ValueError:
            return 0.0

    lps_val = get_abundance('map00540')
    bcaa_val = get_abundance('map00290')
    
    try:
        all_lps = df_pathway.loc['map00540', full_samples].values
        all_bcaa = df_pathway.loc['map00290', full_samples].values
        
        lps_high_thresh = np.percentile(all_lps, 75)
        bcaa_high_thresh = np.percentile(all_bcaa, 75)
        
        if lps_val > lps_high_thresh:
            risk_flags.append(f"Inflammation Risk (High LPS Biosynthesis: {lps_val:.2e} > {lps_high_thresh:.2e})")
            
        if bcaa_val > bcaa_high_thresh:
            risk_flags.append(f"Insulin Resistance Risk (High BCAA Biosynthesis: {bcaa_val:.2e} > {bcaa_high_thresh:.2e})")
            
    except KeyError:
        print("Warning: Could not calculate risk thresholds due to missing pathway data.")

    if risk_flags:
        print("Metabolic Flags:")
        for flag in risk_flags:
            print(f"  Cauton: {flag}")
    else:
        print("Metabolic Flags: None detected.")

def main():
    df_genus, df_pathway = load_data()
    df_pathway = df_pathway.set_index(df_pathway.columns[0])
    
    labels = determine_diet_labels(df_genus)
    
    clf, X_test, y_test, pathway_index = train_diet_model(df_pathway, labels)
    
    print("\n--- SIMULATING APP FOR TEST USERS ---")
    common_samples = [s for s in labels.index if s in df_pathway.columns]
    
    western_samples = labels[labels==0].index.intersection(common_samples)
    plant_samples = labels[labels==1].index.intersection(common_samples)
    if len(western_samples) > 0:
        s_west = western_samples[0]
        x_west = df_pathway[s_west].values
        x_west = np.nan_to_num(x_west)
        simulate_user_app(clf, x_west, s_west, df_pathway.index, df_pathway, common_samples)
        
    if len(plant_samples) > 0:
        s_plant = plant_samples[0]
        x_plant = df_pathway[s_plant].values
        x_plant = np.nan_to_num(x_plant)
        simulate_user_app(clf, x_plant, s_plant, df_pathway.index, df_pathway, common_samples)

if __name__ == "__main__":
    main()
