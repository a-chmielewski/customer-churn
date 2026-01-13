# Model Comparison Report

## Cross-Validation Results (5-Fold Stratified)

| Model               |   ROC-AUC |   ROC-AUC Std |   PR-AUC |   PR-AUC Std |     F1 |   F1 Std |   Train Time (s) |
|:--------------------|----------:|--------------:|---------:|-------------:|-------:|---------:|-----------------:|
| Dummy (Stratified)  |    0.4922 |        0.0101 |   0.2627 |       0.0035 | 0.2498 |   0.0149 |             0.96 |
| Logistic Regression |    0.8485 |        0.0046 |   0.6735 |       0.0241 | 0.6296 |   0.0136 |             0.68 |
| Random Forest       |    0.8453 |        0.0033 |   0.6677 |       0.0235 | 0.6299 |   0.0079 |             1.01 |
| Gradient Boosting   |    0.8393 |        0.0033 |   0.6505 |       0.0197 | 0.5659 |   0.0226 |             2.27 |

## Validation Set Performance

| Model               |   ROC-AUC |   PR-AUC |     F1 |   Precision |   Recall |
|:--------------------|----------:|---------:|-------:|------------:|---------:|
| Dummy (Stratified)  |    0.502  |   0.2666 | 0.2679 |      0.2688 |   0.2669 |
| Logistic Regression |    0.8349 |   0.638  | 0.627  |      0.5294 |   0.7687 |
| Random Forest       |    0.8322 |   0.6368 | 0.6183 |      0.5291 |   0.7438 |
| Gradient Boosting   |    0.8351 |   0.6461 | 0.5674 |      0.6528 |   0.5018 |

## Key Findings

- **Best Model**: Gradient Boosting (ROC-AUC: 0.8351)
- **Improvement over baseline**: 66.3%
