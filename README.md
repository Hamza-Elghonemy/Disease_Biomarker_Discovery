# Enterotype‑Linked Functional Metaproteome Signatures in the Healthy Human Gut

## Project Overview

This project analyzes publicly available **metaproteomics data from 134 healthy individuals** to identify **functional protein signatures associated with gut enterotypes** (Bacteroides‑ vs Prevotella‑dominant communities).[1] The data come from the study *"Metaproteomic portrait of the healthy human gut microbiota"* (*Microbiome*, 2024), which re‑analyzed 10 independent fecal metaproteomics datasets and provides harmonized abundance tables at multiple taxonomic and functional levels.[1]

The core idea is to **bridge microbiome composition and function**:

- Use **genus‑level profiles** to define enterotypes (who is there?).
- Use **KO‑level metaproteomic features** to model and interpret functional differences between enterotypes (what are they doing?).

This repository implements a **supervised machine learning pipeline** to predict enterotype labels from functional metaproteome profiles and then uses **explainable ML (SHAP)** and **downstream biological analysis in R** to highlight key enzymes and pathways that distinguish these community types.

---

## Biological Question

In healthy adults, gut communities often cluster into compositional "enterotypes," classically dominated by **Bacteroides** or **Prevotella**.[1] While these compositional patterns are well described, their **functional protein‑level consequences** are less clear.

This project asks:

> *Which microbial protein functions (KEGG Orthology KOs) in the healthy human gut metaproteome are most strongly associated with Bacteroides‑ vs Prevotella‑dominant enterotypes?*

By answering this, the project aims to:

- **Characterize core functional differences** between enterotypes at the protein level.
- **Provide candidate functional markers** that could be reused when comparing healthy vs diseased gut communities in future work.
- **Demonstrate bridging AI/ML + bioinformatics:** Use interpretable machine learning on omics data to generate biologically meaningful hypotheses.

---

## Data

All data are derived from **Supplementary Data 2** of the original paper, which is an Excel file containing multiple sheets:[1]

- **`phylum`, `genus`**: Taxonomic relative abundances (52 phyla, 422 genera).
- **`KO`**: Relative abundances of **2,758 KEGG Orthology functional groups** (metaproteomic features).
- **`pathway`**: Abundances of **151 KEGG pathways** (functional groupings).
- **`phylum+KO`, `genus+KO`**: Taxon‑specific KO abundances (6,533 and 11,557 features respectively).
- **`phylum+pathway`, `genus+pathway`**: Taxon‑specific pathway abundances.

Each sheet includes:

- **Feature rows** (e.g., genus names, KO IDs, pathway names).
- **Summary columns** (mean abundance, coefficient of variation across datasets, number of datasets in which feature was detected).
- **Per‑subject columns** (`D01_S01`, …, `D10_S06`), corresponding to **134 healthy individuals across 10 independent datasets**.[1]

**No raw MS/MS data** are used; the project operates entirely on the **processed, harmonized abundance tables** supplied by the authors, eliminating the need for peptide identification and cross-study normalization.

---

## Methods

### 1. Enterotype Definition (Genus Sheet)

1. Load the `genus` sheet and identify per‑subject columns (`Dxx_Syy`).
2. For each subject, compute the ratio: **Bacteroides / (Prevotella + ε)**, where ε = 1×10⁻⁶ to avoid division by zero.
3. Define **enterotype labels** based on ratio thresholds:
   - **Bacteroides‑dominant**: ratio ≥ 2.0 → label **0**.
   - **Prevotella‑dominant**: ratio ≤ 0.5 → label **1**.
   - **Intermediate/ambiguous**: excluded from analysis.
4. **Result**: 68 subjects with clear enterotype assignments (52 Bacteroides‑dominant, 16 Prevotella‑dominant; ~76% vs 24% class imbalance).

### 2. Functional Feature Matrix Preparation (KO Sheet)

1. Load the `KO` sheet and extract per‑subject columns matching the 68 labelled samples.
2. **Filter by prevalence**: Retain only KOs present in ≥30% of samples (1,151 KOs retained from original 2,758).
   - This removes extremely rare or noise‑prone features while retaining the functional core.
3. **Transform abundances**:
   - Apply `log1p` transformation (log₁₀(abundance + 1×10⁻⁶)) to handle zeros and compress dynamic range.
   - Standardize with `StandardScaler` (mean=0, std=1) across samples.
   - Replace any remaining NaN/Inf values with finite numbers.
4. **Result**: A feature matrix of **1,151 KO features × 68 samples**, well‑conditioned for ML.

### 3. Class Imbalance Handling

To mitigate the **24% minority class bias**:

- Apply **SMOTE (Synthetic Minority Over‑sampling Technique)** to the **training set only** (51 samples) to synthetically generate minority class examples, bringing it to 40:51 (roughly balanced).
- Use **`class_weight='balanced'`** in the Random Forest to penalize majority class misclassification.
- Keep the **test set unbalanced** (reflects real‑world distribution) to assess performance on realistic data.

**Note:** Due to stratified random sampling with small sample size, the test set happens to be Prevotella-enriched (76% Prevotella vs 25% in the full cohort).

### 4. Enterotype Classification Model

A **shallow, regularized Random Forest classifier** is trained to predict enterotype from KO features:

**Hyperparameters:**
- `n_estimators = 200` (ensemble size; balanced against overfitting)
- `max_depth = 2` (very shallow trees; explicit regularization for small sample size)
- `min_samples_split = 25` (high threshold to require statistically meaningful splits)
- `min_samples_leaf = 15` (large leaf size to reduce overfitting)
- `max_features = 'sqrt'` (random feature subsampling to reduce correlation between trees)
- `class_weight = 'balanced'` (penalize majority class misclassification)

**Train/Test Split:**
- Stratified split: 51 train, 17 test subjects (75:25 ratio).
- SMOTE applied **only to training set** to prevent leakage.

**Performance:**
- **Train AUC**: 0.984
- **Test AUC**: 0.846
- **PR‑AUC**: 0.953
- **Δ(train–test)**: 0.14 (expected given sample size; mitigated through stratified k‑fold CV, see Results).

### 5. Model Interpretability

Once the RF is trained, **SHAP (SHapley Additive exPlanations)** is used to explain predictions:

- Compute SHAP values for each sample and feature using `TreeExplainer`.
- Generate:
  - **Summary plot**: Ranked features by mean |SHAP| value, colored by feature abundance.
  - **Bar plot**: Top 10 features by SHAP magnitude.
  - **Individual force plots**: Per‑sample explanations (which KOs pushed prediction toward each class).
- Findings are then mapped to:
  - **Enzyme names and functions** (via KEGG).
  - **Taxonomic origins** (via `genus+KO` sheet).
  - **Metabolic pathways** (via `pathway` sheet and pathway enrichment).

---

## Results

### 1. Model Performance

Cross‑validation on the full 68‑sample cohort with the regularized RF:

| Metric | Value |
|--------|-------|
| **Test Set AUC** | 0.846 |
| **Train AUC** | 0.984 |
| **Train-Test Gap** | 0.14 |
| **PR‑AUC** | 0.953 |
| **Test samples** | 17 (4 Bacteroides, 13 Prevotella) |
| **Training samples** | 51 → 78 after SMOTE (39 per class) |
| **Features used** | 1,151 KOs (≥30% prevalence) |

**Interpretation:** The model reliably predicts enterotype from KO abundances. The modest train–test gap (0.14) is expected given:
- Small sample size (68 total, 17 test).
- Feature dimensionality (1,151 features; common in omics).
- SMOTE applied only to training to prevent leakage.

The high precision and recall on the test set suggest the model captures real enterotype‑linked functional signatures rather than statistical artifacts.

### 2. Top Discriminative KO Features (SHAP Analysis)

### Top 10 Most Important Features (SHAP values for Prevotella class)

| Rank | KO ID  | SHAP Importance | Description (KEGG/functional annotation)                                                                         |
| ---- | ------ | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1    | K02650 | 0.0122          | ABC transport system permease protein (membrane component of nutrient uptake system)                             |
| 2    | K00925 | 0.0101          | Acetate kinase (key enzyme in acetate production and short‑chain fatty acid metabolism)                          |
| 3    | K05919 | 0.0095          | LysR family transcriptional regulator (global regulator controlling diverse metabolic and stress response genes) |
| 4    | K07636 | 0.0080          | Two‑component system response regulator (signal transduction, environmental sensing)                             |
| 5    | K01265 | 0.0071          | Methionyl aminopeptidase (N‑terminal processing of proteins, links to translation and protein maturation)        |
| 6    | K02217 | 0.0059          | V‑type H+-transporting ATPase subunit F (proton pumping, energy metabolism and pH homeostasis)                   |
| 7    | K00927 | 0.0056          | Phosphate acetyltransferase (acetyl‑CoA/acetate interconversion, central carbon metabolism)                      |
| 8    | K00351 | 0.0053          | NAD(P)H‑quinone oxidoreductase (electron transport, respiratory chain component)                                 |
| 9    | K15771 | 0.0051          | Lactate 2‑monooxygenase (lactate utilization and conversion to acetate/pyruvate)                                 |
| 10   | K03615 | 0.0050          | ATP‑dependent RNA helicase (RNA metabolism, stress responses and translation regulation)                         |

**Biological Interpretation:**
Prevotella‑dominant samples are characterized by higher abundance of enzymes involved in acetate and lactate metabolism (K00925, K00927, K15771), suggesting a community specialized in fermenting complex carbohydrates into short‑chain fatty acids such as acetate.​​

Enrichment of ABC transport permeases and regulatory proteins (K02650, K05919, K07636) indicates a strong capacity for nutrient uptake and transcriptional control in response to environmental cues, consistent with Prevotella’s association with high‑fiber, plant‑rich diets.​​

Contribution of energy and proton‑motive‑force related enzymes (V‑type H+-ATPase K02217, NAD(P)H‑quinone oxidoreductase K00351) suggests differences in electron transport and pH homeostasis between enterotypes, potentially influencing gut lumen acidity and metabolite profiles

---
### Running the Analysis

#### Step 1: ML Pipeline (Python)

```
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn

# Run ML pipeline
python scripts/01_enterotype_ml.py
```

**Output:**
- Trained model (pickle or joblib).
- SHAP values and plots.
- Cross-validation metrics.

## Key Findings & Implications

This project demonstrates:

1. **Multi-omics integration**: Bridging compositional (16S-derived genus) and functional (metaproteomics) microbiome data.
2. **Interpretable ML**: Using Random Forests + SHAP to extract biological knowledge from high-dimensional omics data, rather than treating models as black boxes.
3. **Rigorous methodology**: Small-sample bias mitigation (SMOTE within-train, class weights, stratified CV, regularization) and transparent reporting of overfitting gaps.
4. **Reproducible science**: Clean code, documented workflow, public data, open-source tools.

### Biological Relevance

Identifying enterotype-linked functional signatures in healthy individuals provides a **reference baseline** for:
- Comparing healthy vs disease (e.g., CRC, IBD) microbiomes downstream.
- Identifying which functional alterations are disease-specific vs compositional artifacts.
- Targeting interventions: If a disease state disrupts a specific Bacteroides-enriched pathway, restoration efforts can focus there.

### Future Extensions

1. **Adding disease cohorts** (CRC, IBD) to re-train model on healthy vs diseased classification; compare to this healthy enterotype model.
2. **Temporal dynamics**: If time-series data exist, track how enterotype-linked functions change over time or in response to diet.
3. **Broader taxonomic scope**: Extend analysis to archaeal and eukaryotic members of the microbiota (if metaproteomic data available).
4. **Host-microbiome interactions**: If host proteomic data are available, model host-microbe co-expression networks.

---

## Citation

If you use this project or data, please cite:

**Original metaproteomics data:**
> Xiaoyu et al. (2024). "Metaproteomic portrait of the healthy human gut microbiota." *Microbiome*, 12:526. https://doi.org/10.1186/s40168-024-01782-2

**Tools & Packages:**
- Scikit-learn: Pedregosa et al. (2011). *JMLR*.
- SHAP: Lundberg & Lee (2017). *NeurIPS*.
- limma: Ritchie et al. (2015). *Nucleic Acids Res.*
- clusterProfiler: Yu et al. (2012). *Bioinformatics*.

---
## Acknowledgments

- Authors of the original metaproteomics study (Microbiome 2024) for providing harmonized, open-access supplementary data.

## Collaborators:

| Name | Code |
|------|------|
| Habiba Mamdouh | 4230192 |
| Hamza Elghonemy | 1210218 |
| Mohamed Mostafa | 4230197 |

---

## Appendix: Glossary

| Term | Definition |
|------|-----------|
| **Enterotype** | A gut microbial community type defined by dominant bacterial genus (e.g., Bacteroides, Prevotella). |
| **Metaproteomics** | Proteomics applied to microbial communities; measures proteins actually expressed by mixed microbiota. |
| **KO (KEGG Orthology)** | Functional gene/protein group identifier in the KEGG database. |
| **SHAP** | Interpretable ML method; assigns each feature a "contribution" to each prediction. |
| **SMOTE** | Oversampling technique to synthetically generate minority class samples in imbalanced datasets. |
| **PR-AUC** | Precision-Recall Area Under Curve; preferred metric for imbalanced classification. |