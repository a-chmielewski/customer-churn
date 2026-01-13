# Final Test Set Evaluation Report

**Model**: Gradient Boosting

---

## Test Set Performance

| Metric | Score |
|:-------|------:|
| ROC-AUC | 0.8410 |
| PR-AUC | 0.6556 |
| F1 Score | 0.5545 |
| Precision | 0.6222 |
| Recall | 0.5000 |
| Accuracy | 0.7871 |

## Confusion Matrix

```
                Predicted
              No Churn  Churn
Actual No  |     692      85
       Yes |     140     140
```

- **True Negatives (TN)**: 692 - Correctly predicted no churn
- **False Positives (FP)**: 85 - Incorrectly predicted churn
- **False Negatives (FN)**: 140 - Missed churners
- **True Positives (TP)**: 140 - Correctly predicted churn

## Validation vs Test Comparison

| Metric    |   Validation |   Test |   Difference |   % Change |
|:----------|-------------:|-------:|-------------:|-----------:|
| ROC-AUC   |       0.8351 | 0.8410 |       0.0059 |     0.7107 |
| PR-AUC    |       0.6461 | 0.6556 |       0.0095 |     1.4759 |
| F1 Score  |       0.5674 | 0.5545 |      -0.0129 |    -2.2821 |
| Precision |       0.6528 | 0.6222 |      -0.0306 |    -4.6809 |
| Recall    |       0.5018 | 0.5000 |      -0.0018 |    -0.3546 |

## Interpretation

[OK] **Model Generalization**: Excellent - Test performance closely matches validation (ROC-AUC difference < 0.02)

### Business Impact

- **Test Set Size**: 1,057 customers
- **Actual Churners**: 280 (26.5%)
- **Predicted Churners**: 225 (21.3%)
- **Correctly Identified**: 140 churners (50.0% of all churners)
- **False Alarms**: 85 customers (37.8% of predictions)

### Recommendations

- **High Precision**: Good for targeted interventions with limited budget
