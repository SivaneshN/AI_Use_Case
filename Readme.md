# Customer Segmentation for Marketing Optimization

> **Machine Learning Project 1** - Unsupervised customer segmentation using KMeans clustering on credit card behavioural data.


## Overview

This project discovers meaningful customer segments from 6-month credit card activity data to enable targeted marketing campaigns. Starting from raw transactional features, it applies a full ML pipeline - EDA → Preprocessing → PCA → KMeans Clustering → Segment Profiling and translates the results into concrete, actionable marketing recommendations.

**Dataset:** 8,500 customers × 18 features covering balance, purchases, cash advances, payments, and credit behaviour.


## Repository Structure

```
AI_Use_Case/
├── data/
│   └── Project1_dataset.csv          # Raw credit card activity dataset
├── docs/
│   └── CustomerSegmentation_Presentation.pptx   # Project presentation
└── notebook/
    └── Project1_CustomerSegmentation_final.ipynb # Main analysis notebook
```


## Results — 4 Customer Segments

| Cluster | Segment Name | Size | Share | Key Behaviour |
|---------|-------------|------|-------|---------------|
| 0 | **Active Full-Payers** | 1,189 | 14.0% | High purchases, 81% full-payment rate, lowest credit risk |
| 1 | **Cash-Advance Dependents** | 2,572 | 30.3% | High balance (avg £2,667), relies on cash advances, minimum payments only |
| 2 | **Inactive / Dormant** | 1,122 | 13.2% | Low usage across all features, highest re-engagement potential |
| 3 | **Instalment Revolvers** | 3,617 | 42.6% | Largest segment, active shoppers carrying revolving balance |


## Pipeline Summary

| Step | Decision | Rationale |
|------|----------|-----------|
| Missing values | Median imputation | Robust to right-skewed distributions |
| Transformation | `log1p` on monetary/count columns | Reduces skew without losing zero values |
| Scaling | `RobustScaler` | IQR-based — handles remaining outliers |
| Feature engineering | 2 ratio features added | Adds relative context (utilisation, advance reliance) |
| Dimensionality reduction | PCA — 7 components (90% variance) | Removes noise, handles multicollinearity |
| Clustering | KMeans with k-means++ initialisation | Fast, interpretable, suited to numeric data |
| k selection | k = 4 | Best elbow + highest Calinski-Harabasz score + business interpretability |


## Key Marketing Recommendations

| Segment | Action |
|---------|--------|
| Active Full-Payers | Premium card upgrade, loyalty programme, higher credit limit offers |
| Cash-Advance Dependents | Debt consolidation products, financial health tools, instalment plan offers |
| Inactive / Dormant | Time-limited bonus offers, personalised category incentives, proactive outreach |
| Instalment Revolvers | Instalment plan promotions, merchant partner offers, balance transfer deals |


## Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualisation
- `scikit-learn` — preprocessing, PCA, KMeans, evaluation metrics
- `scipy` — statistical analysis


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SivaneshN/AI_Use_Case.git
   cd AI_Use_Case
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

3. Place `Project1_dataset.csv` in the `data/` folder.

4. Open and run the notebook:
   ```bash
   jupyter notebook notebook/Project1_CustomerSegmentation_final.ipynb
   ```


## Limitations

- 6 months of data may not capture seasonal behaviour or long-term trends
- Segments are purely behavioural - no demographic data included
- KMeans assumes roughly spherical clusters; GMM or DBSCAN may reveal different structure
- Silhouette score of 0.30 indicates meaningful but overlapping clusters, expected in real consumer data