# Customer Churn - Exploratory Data Analysis Report

## Executive Summary

This EDA focuses on identifying key business drivers of customer churn in the telco dataset. Analysis reveals clear patterns that can inform both modeling and business strategy.

---

## Dataset Overview

- **Total Customers**: 7,043
- **Features**: 21 (after cleaning)
- **Target Variable**: Churn (Yes/No)
- **Overall Churn Rate**: **26.5%** (1,869 churned / 5,174 retained)
- **Class Imbalance**: 2.8:1 ratio (retained:churned)

---

## Data Quality Issues Addressed

### TotalCharges Column
- **Issue**: Non-numeric values (stored as string/object type)
- **Root Cause**: 11 customers with blank TotalCharges (likely new customers with tenure=0)
- **Resolution**: Converted to numeric, filled missing values with MonthlyCharges
- **Impact**: No data loss, improved data consistency

---

## Key Findings

### 1. Contract Type - Strongest Predictor

**Churn Rate by Contract Type:**
- **Month-to-month**: 42.7% (1,655 / 3,875 customers)
- **One year**: 11.3% (166 / 1,473 customers)  
- **Two year**: 2.8% (48 / 1,695 customers)

**Business Insight:**
- Month-to-month customers are **15x more likely** to churn than two-year contract customers
- Contract commitment is the single strongest retention factor
- 55% of customer base is on risky month-to-month contracts

**Visualization:** `01_churn_distribution.png`, `03_churn_vs_contract.png`

---

### 2. Monthly Charges - Price Sensitivity

**Median Monthly Charges:**
- **Churned customers**: $79.65
- **Retained customers**: $64.43
- **Difference**: +$15.22 (24% premium)

**Distribution Analysis:**
- Churned customers cluster at higher price points ($70-$100 range)
- Clear price sensitivity visible in histogram overlap
- Higher charges correlate with increased churn risk

**Business Insight:**
- Price is a significant churn driver
- Premium pricing without perceived value leads to defection
- Opportunity for value-based pricing strategies

**Visualization:** `02_churn_vs_monthly_charges.png`

---

### 3. Customer Tenure - The Critical First Year

**Median Tenure:**
- **Churned customers**: 10 months
- **Retained customers**: 38 months
- **Gap**: 28 months

**New Customer Risk:**
- **Customers ≤12 months**: 47.4% churn rate
- Nearly **2x the overall churn rate**
- Churn rate declines sharply after the first year

**Business Insight:**
- First 12 months are critical for retention
- Long-term customers are significantly more loyal
- Early-stage customer experience is make-or-break

**Visualization:** `04_churn_vs_tenure.png`

---

### 4. Combined Risk Factors - Highest Risk Segments

**Top Risk Segment:**
```
Month-to-Month + New Customer (<12mo) + High Charges
→ 69.95% churn rate (812 customers)
```

**Risk Tier Breakdown:**
1. **Critical Risk** (70% churn): Month-to-month, new, high charges
2. **High Risk** (42% churn): Month-to-month, established, high charges  
3. **Moderate Risk** (39% churn): Month-to-month, new, low charges
4. **Lower Risk** (18% churn): Month-to-month, established, low charges

**Business Insight:**
- Risk factors compound multiplicatively
- Nearly 70% of highest-risk segment will churn
- Targeted interventions needed for at-risk segments

**Visualization:** `05_risk_factors_heatmap.png`

---

## Modeling Implications

### Feature Importance Expectations
1. **Contract type** - Expect highest importance
2. **Tenure** - Strong predictor, especially early months
3. **MonthlyCharges** - Moderate importance with interactions
4. **TotalCharges** - May be redundant with tenure × MonthlyCharges

### Recommended Preprocessing
- **Encode contract type** with ordinal relationship awareness
- **Create tenure bins** to capture non-linear effects
- **Feature engineering**: 
  - Price-per-service ratio
  - Contract × Charges interactions
  - Early-customer flag (tenure ≤ 12)

### Class Imbalance Handling
- 2.8:1 imbalance is moderate
- Consider SMOTE or class weights
- Prioritize recall for churn class (cost of false negatives is high)

---

## Business Recommendations

### Immediate Actions
1. **Contract Incentives**: Promote annual contracts with discounts
2. **Price Review**: Examine value proposition for $70+ customers
3. **Onboarding Program**: Enhanced support in first 12 months

### Retention Strategy
- **High-Risk Segment**: Proactive outreach, personalized offers
- **Price-Sensitive**: Loyalty discounts, bundle options
- **New Customers**: White-glove service, engagement tracking

---

## Charts

1. `01_churn_distribution.png` - Target variable distribution
2. `02_churn_vs_monthly_charges.png` - Price sensitivity analysis
3. `03_churn_vs_contract.png` - Contract type impact
4. `04_churn_vs_tenure.png` - Tenure relationship with churn
5. `05_risk_factors_heatmap.png` - Combined risk factor analysis
6. `eda_summary.txt` - Detailed statistics summary

---

## Next Steps

1. ✅ Data loaded and validated
2. ✅ EDA completed with business insights
3. ⏭️ **Next**: Feature engineering and preprocessing
4. ⏭️ Model training with multiple algorithms
5. ⏭️ Model evaluation and selection
6. ⏭️ Model interpretation with SHAP/feature importance

---

*Analysis Date: January 9, 2026*  
*Script: `src/churn/eda.py`*
