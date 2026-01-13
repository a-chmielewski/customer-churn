# Customer Churn Prediction

**Production-ready ML pipeline for predicting telecom customer churn with business-optimized threshold and $1.4M annual value.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Problem

Predict which customers will churn (cancel service) to enable proactive retention interventions. The model identifies high-risk customers and optimizes contact strategy for maximum ROI.

**Business Impact**: $118,000 monthly profit | $1.4M annual value | 476% ROI

---

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn shap

# Run full pipeline (Linux/Mac with make)
make all

# Or on Windows (PowerShell)
.\run.ps1 all

# Or step-by-step
make train         # Model training (or: .\run.ps1 train)
make evaluate      # Test set evaluation
make explain       # SHAP + threshold optimization
```

---

## Results

### Model Performance

| Metric | Score | Context |
|:-------|------:|:--------|
| **ROC-AUC** | 84.1% | Excellent discrimination |
| **PR-AUC** | 65.6% | Strong on imbalanced data |
| **Precision** | 62.2% | 3 of 5 predictions correct |
| **Recall** | 50.0% | Catches half of churners |

**Key Validation**: Test performance **exceeds** validation (no overfitting).

### Business Optimization

**Optimized Threshold: 0.110** (vs default 0.500)

| Metric | Value |
|:-------|------:|
| Monthly Net Profit | **$118,000** |
| Annual Value | **$1.4 Million** |
| ROI | **476%** |
| Contact Volume | 496 customers |
| Cost per Contact | $50 |

**Strategy**: Contact 496 highest-risk customers monthly, expect to retain 71 (238 identified × 30% success rate).

### Top Churn Drivers (SHAP)

![SHAP Feature Importance](reports/figures/shap_importance.png)

1. **charges_per_tenure** (26%) - Loyalty metric
2. **is_month_to_month** (24.7%) - Contract risk
3. **MonthlyCharges** (11.5%) - Price sensitivity
4. **TotalCharges** (10.1%) - Customer value
5. **Fiber optic internet** (6.3%) - Service type

**Insight**: Top 2 features (50% importance) are EDA-driven engineered features.

### Business Decision Curve

![Profit vs Threshold](reports/figures/profit_threshold_analysis.png)

**Finding**: Optimal threshold (0.110) maximizes profit while staying within contact capacity (500/month).

### Model Calibration

![Calibration Curve](reports/figures/calibration_curve.png)

**Result**: Model is well-calibrated (Brier score: 0.140) - probabilities are reliable.

---

## Project Structure

```
customer-churn/
├── data/
│   ├── raw/                    # Original data
│   └── processed/              # Preprocessed train/val/test splits
├── src/churn/
│   ├── config.py              # Configuration
│   ├── data_load.py           # Data loading + validation
│   ├── eda.py                 # Exploratory analysis
│   ├── preprocess.py          # Leakage-safe preprocessing
│   ├── train.py               # Model training + comparison
│   ├── evaluate.py            # Test set evaluation
│   └── explain.py             # SHAP + threshold optimization
├── models/
│   ├── best_model.pkl         # Trained Gradient Boosting
│   ├── calibrated_model.pkl   # Calibrated probabilities
│   └── preprocessor.pkl       # Fitted pipeline
├── reports/
│   ├── case_study.md          # Full documentation
│   ├── model_comparison.csv   # All models compared
│   ├── interpretation_report.md # Business optimization
│   └── figures/               # 16 visualizations
├── tests/
│   └── test_pipeline.py       # Unit tests
├── Makefile                   # Automation commands
└── README.md
```

---

## Key Features

**Technical Excellence:**
- ✅ Leakage-safe preprocessing (split → fit → transform)
- ✅ Proper imbalanced classification metrics (ROC-AUC, PR-AUC)
- ✅ 5-fold stratified cross-validation
- ✅ Test set generalization (no overfitting)
- ✅ Model calibration analysis

**Business Value:**
- ✅ SHAP explainability (feature importance)
- ✅ Threshold optimization for profit maximization
- ✅ Clear ROI calculation ($1.4M annual value)
- ✅ Top at-risk customers identified
- ✅ Deployment strategy with A/B testing plan

**Production Ready:**
- ✅ Complete preprocessing pipeline saved
- ✅ Unit tests covering key functionality
- ✅ Makefile for automation
- ✅ Comprehensive documentation

---

## Methodology

### 1. EDA-Driven Insights

**Key Findings:**
- Month-to-month contracts: 42.7% churn (15x riskier than 2-year)
- New customers (≤12 months): 47.4% churn rate
- Churned customers pay +$15/month on average

**Action**: 6 engineered features based on these insights.

### 2. Preprocessing

- Train/Val/Test: 70% / 15% / 15% (stratified)
- Feature engineering: 21 → 38 features
- Numeric: Median imputation + StandardScaler
- Categorical: Most frequent + One-hot encoding
- **Critical**: Fit transformers on training data only (no leakage)

### 3. Model Selection

Compared 4 models with 5-fold CV:
- Dummy Classifier (baseline)
- Logistic Regression
- Random Forest
- **Gradient Boosting** ← Selected (highest PR-AUC)

**Result**: 66% improvement over baseline.

### 4. Interpretation

- **SHAP**: Validates EDA-driven features as most important
- **Calibration**: Well-calibrated probabilities (Brier: 0.140)
- **Threshold**: Optimized from 0.500 → 0.110 for business value

---

## Reproducibility

**Run Tests:**
```bash
make test
```

**Reproduce Results:**
```bash
# Full pipeline
make all

# Or individual steps
make preprocess  # Creates data/processed/*.npy
make train       # Creates models/*.pkl, reports/model_comparison.*
make evaluate    # Creates reports/test_evaluation.md
make explain     # Creates reports/interpretation_report.md
```

**Expected Runtime:**
- Preprocessing: ~5 seconds
- Training: ~10 seconds
- Evaluation: ~5 seconds
- Interpretation: ~8 seconds
- **Total: ~30 seconds**

---

## Deployment Strategy

**Phase 1: Pilot** (Month 1)
- Deploy with threshold = 0.110
- A/B test: 100 control vs 100 treatment
- Track actual retention rates

**Phase 2: Rollout** (Months 2-3)
- Scale to 496 contacts/month
- Refine interventions based on results

**Phase 3: Optimization** (Months 4+)
- Retrain with new data
- Fine-tune threshold based on actual results

**Success Metrics:**
- Retention rate: Target 30%
- Monthly profit: Target $100K+
- ROI: Target >400%

---

## Limitations & Next Steps

### Current Limitations

1. **Static model**: Needs retraining with new data
2. **Assumptions**: 30% retention success rate is estimated
3. **Single CLV**: Assumes $2,000 average (actual varies)
4. **No seasonality**: Model doesn't capture temporal patterns

### Future Enhancements

**Short-term:**
- [ ] Add survival analysis (time-to-churn)
- [ ] Implement real-time scoring API
- [ ] Create monitoring dashboard
- [ ] Add individual SHAP explanations per customer

**Medium-term:**
- [ ] Incorporate customer feedback data
- [ ] Test intervention strategies (discount vs upgrade)
- [ ] Build segment-specific models
- [ ] Add causal inference for intervention effects

**Long-term:**
- [ ] Deploy as REST API with FastAPI
- [ ] Add MLOps pipeline (MLflow/Kubeflow)
- [ ] Implement online learning
- [ ] Create customer retention chatbot

---

## Requirements

```
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
shap < 0.50
joblib >= 1.0.0
```

---

## Citation

```bibtex
@misc{customer_churn_2026,
  title={Customer Churn Prediction with Business Optimization},
  author={[Aleksander Chmielewski]},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/a-chmielewski/customer-churn}}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*Built with scikit-learn • Optimized for business value • Production-ready*
