# Disease Biomarker Discovery & Microbiome Diet Tracker

## Project Overview

This project analyzes **metaproteomics and gut microbiome data** to identify functional protein signatures and health biomarkers associated with gut enterotypes and disease states. The project includes:

1. **Enterotype Classification**: Analysis of 134 healthy individuals to identify functional signatures of Bacteroides- vs Prevotella-dominant gut communities
2. **Disease Biomarker Discovery**: Classification of healthy vs CRC patients using gut proteome data
3. **Interactive Health App**: A Streamlit web application that predicts diet patterns and health risks from microbiome profiles

### Data Sources

**Healthy Gut Metaproteomics**: Data from *"Metaproteomic portrait of the healthy human gut microbiota"* (Microbiome, 2024) - 134 healthy individuals across 10 independent datasets with harmonized abundance tables at taxonomic and functional levels.

**CRC Gut Proteome**: Proteomic data comparing healthy individuals (H samples) with colorectal cancer patients (P samples).

---

## Project Structure

```
Disease_Biomarker_Discovery/
├── data/                           # Data files (Excel, CSV)
│   ├── abundance_tables.xlsx       # Taxonomic & functional abundances
│   ├── correlations.xlsx           # Phyla correlations
│   ├── crc_gut_proteome.csv       # CRC vs Healthy proteome data
│   ├── differential_analysis_results.xlsx
│   └── rank_data.xlsx
├── scripts/                        # Analysis scripts
│   ├── analyze_enterotypes.py     # Enterotype classification (RF + SMOTE)
│   ├── check_significance.py      # Statistical significance testing
│   ├── classify_crc.py            # CRC classification with LOOCV
│   ├── diet_and_health_model.py   # Diet pattern prediction model
│   ├── inspect_data.py            # Data exploration
│   └── visualize_data.py          # Data visualization
├── results/                        # Output files
│   ├── figures/                   # Visualizations
│   ├── enterotypes/               # Enterotype analysis results
│   ├── ml/                        # Machine learning outputs
│   └── diet_health_model/         # Diet model results
├── diet_app.py                     # Streamlit web application
├── pathway_names.txt               # KEGG pathway mappings
└── README.md
```

---

## Key Features

### 1. Enterotype Classification

**Objective**: Predict enterotype (Bacteroides- vs Prevotella-dominant) from functional KO features

**Methods**:
- Enterotype definition based on Bacteroides/Prevotella ratio thresholds (≥2.0 for Bacteroides, ≤0.5 for Prevotella)
- Random Forest classifier with SMOTE for class imbalance
- 1,151 KO features (filtered by ≥30% prevalence)
- Stratified train/test split (75:25)

**Results**:
- Test AUC: 0.846
- PR-AUC: 0.953
- 68 subjects classified (52 Bacteroides-dominant, 16 Prevotella-dominant)

**Top Discriminative Features** (Prevotella-associated):
- K00925: Acetate kinase (SCFA metabolism)
- K02650: ABC transport permease
- K00927: Phosphate acetyltransferase
- K15771: Lactate 2-monooxygenase

### 2. CRC Classification

**Objective**: Distinguish healthy from CRC patients using gut proteome

**Methods**:
- Leave-One-Out Cross-Validation (LOOCV)
- Feature selection pipeline (VarianceThreshold + SelectKBest)
- Random Forest with balanced class weights
- Top 10 most discriminative proteins identified

### 3. Diet & Health Prediction App

**Interactive Streamlit Application** that provides:
- **Diet Pattern Analysis**: Predicts Western vs Plant-Based diet signatures
- **Health Risk Assessment**: Inflammation and insulin resistance indicators
- **Metabolic Pathway Visualization**: Key functional features
- **AI Health Coach**: Personalized recommendations using Google Gemini

**Live Demo**: Run `streamlit run diet_app.py`

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn imbalanced-learn
pip install matplotlib seaborn plotly
pip install streamlit google-generativeai python-dotenv
pip install openpyxl scipy statsmodels
```

### Environment Configuration

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## Usage

### Run Analysis Scripts

```bash
# Enterotype classification
python scripts/analyze_enterotypes.py

# CRC classification
python scripts/classify_crc.py

# Diet and health model
python scripts/diet_and_health_model.py

# Data visualization
python scripts/visualize_data.py

# Statistical significance testing
python scripts/check_significance.py
```

### Launch Web Application

```bash
streamlit run diet_app.py
```

The app will open in your browser at `http://localhost:8501`

**Features**:
- Select any sample ID from the dataset
- View diet pattern predictions (Western vs Plant-Based)
- Check health risk flags (inflammation, insulin resistance)
- Explore metabolic pathway analysis
- Get AI-powered personalized health recommendations

---

## Methods Summary

### Enterotype Definition

1. Compute **Bacteroides / Prevotella ratio** for each subject
2. Classify based on thresholds:
   - Bacteroides-dominant: ratio ≥ 2.0 (label 0)
   - Prevotella-dominant: ratio ≤ 0.5 (label 1)
   - Intermediate: excluded

### Feature Processing

- **Prevalence filtering**: Keep KOs present in ≥30% of samples
- **Transformation**: log1p + StandardScaler
- **Class imbalance**: SMOTE applied to training set only

### Model Configuration

**Random Forest Hyperparameters**:
- `n_estimators = 200`
- `max_depth = 2` (shallow for regularization)
- `min_samples_split = 25`
- `min_samples_leaf = 15`
- `class_weight = 'balanced'`

---

## Results & Biological Insights

### Functional Signatures

**Prevotella-dominant communities**:
- Enriched in acetate/lactate metabolism enzymes
- Higher ABC transporter activity (nutrient uptake)
- Enhanced pH homeostasis machinery
- Associated with high-fiber, plant-rich diets

**Bacteroides-dominant communities**:
- Mucin degradation pathways
- Protein fermentation
- Associated with Western-type diets

### Health Implications

The models identify:
- **High LPS biosynthesis** → Inflammation risk marker
- **Elevated BCAA biosynthesis** → Insulin resistance indicator
- Pathway-level biomarkers for metabolic health assessment

---

## Key Findings

1. **Multi-omics Integration**: Successfully bridges compositional (genus-level) and functional (KO-level) microbiome data
2. **Interpretable ML**: Uses Random Forest with feature importance analysis for biological insights
3. **Practical Application**: Provides actionable health insights through interactive web interface
4. **Rigorous Validation**: SMOTE, stratified CV, LOOCV for small-sample robustness

---

## Future Extensions

1. **Disease Cohorts**: Extend to IBD, metabolic syndrome classification
2. **Temporal Analysis**: Track enterotype stability over time
3. **Diet Intervention**: Predict microbiome response to dietary changes
4. **Host Integration**: Incorporate host metabolomics/transcriptomics

---

## Citation

**Original Data**:
> Xiaoyu et al. (2024). "Metaproteomic portrait of the healthy human gut microbiota." *Microbiome*, 12:526. https://doi.org/10.1186/s40168-024-01782-2

**Tools & Libraries**:
- Scikit-learn: Pedregosa et al. (2011). *JMLR*
- SMOTE (imbalanced-learn): Lemaître et al. (2017)
- Streamlit: https://streamlit.io
- Plotly: https://plotly.com

---

## Collaborators

| Name            | Code    |
| --------------- | ------- |
| Habiba Mamdouh  | 4230192 |
| Hamza Elghonemy | 1210218 |
| Mohamed Mostafa | 4230197 |

---

## License

This project uses publicly available data and open-source tools. Please cite the original data sources when using this code.

---

## Glossary

| Term                    | Definition                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| **Enterotype**          | Gut microbial community type defined by dominant bacterial genus (Bacteroides, Prevotella) |
| **Metaproteomics**      | Proteomics applied to microbial communities; measures expressed proteins                   |
| **KO (KEGG Orthology)** | Functional gene/protein group identifier in KEGG database                                  |
| **SMOTE**               | Synthetic Minority Over-sampling Technique for imbalanced datasets                         |
| **LOOCV**               | Leave-One-Out Cross-Validation                                                             |
| **LPS**                 | Lipopolysaccharide biosynthesis (inflammation marker)                                      |
| **BCAA**                | Branched-Chain Amino Acids (metabolic health marker)                                       |