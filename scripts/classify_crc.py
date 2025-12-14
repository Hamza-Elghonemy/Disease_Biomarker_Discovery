
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
import os

# Configuration
DATA_FILE = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/data/crc_gut_proteome.csv'
RESULTS_DIR = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/results/ml'
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

def load_and_prepare_data():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    sample_ids = df.iloc[:, 0]
    X = df.iloc[:, 1:].values
    feature_names = df.columns[1:]
    
    y = np.array([0 if 'H' in s else 1 for s in sample_ids])
    
    print(f"Data Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Healthy={sum(y==0)}, Patient={sum(y==1)}")
    
    return X, y, feature_names, sample_ids

def train_and_evaluate(X, y, feature_names):
    print("Training Random Forest with LOOCV (Leave-One-Out Cross-Validation)...")
    
    from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('dropper', VarianceThreshold(threshold=0)),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=10)), 
        ('clf', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=3
        ))
    ])
    
    loo = LeaveOneOut()
    
    y_pred = cross_val_predict(pipeline, X, y, cv=loo)
    try:
        y_proba = cross_val_predict(pipeline, X, y, cv=loo, method='predict_proba')[:, 1]
    except:
        y_proba = np.zeros_like(y, dtype=float)
    acc = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("\n--- Model Performance ---")
    print(f"Accuracy: {acc:.2%}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.2%}")
    print(f"Specificity (True Negative Rate): {specificity:.2%}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Healthy', 'Patient']))
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (LOOCV)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
    plt.close()
    
    print("Extracting Top Features...")
    pipeline.fit(X, y)
    
    dropper = pipeline.named_steps['dropper']
    selector = pipeline.named_steps['selector']
    clf = pipeline.named_steps['clf']
    
    features_after_drop = feature_names[dropper.get_support(indices=True)]
    selected_indices = selector.get_support(indices=True)
    selected_features = features_after_drop[selected_indices]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_features = selected_features[indices]
    top_scores = importances[indices]
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_scores, y=top_features, palette='viridis')
    plt.title(f'Top {len(top_features)} Discriminative Proteins')
    plt.xlabel('Feature Importance (Gini)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))
    plt.close()
    
    # Save Feature List
    importance_df = pd.DataFrame({
        'Protein_ID': top_features,
        'Importance': top_scores
    })
    importance_df.to_csv(os.path.join(RESULTS_DIR, 'top_biomarkers.csv'), index=False)
    print(f"Top biomarkers saved to {os.path.join(RESULTS_DIR, 'top_biomarkers.csv')}")

if __name__ == "__main__":
    X, y, feature_names, sample_ids = load_and_prepare_data()
    train_and_evaluate(X, y, feature_names)
