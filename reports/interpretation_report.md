# Model Interpretation & Business Decision Report

---

## 1. Model Calibration

### Calibration Quality

- **Brier Score (Uncalibrated)**: 0.1398
- **Brier Score (Calibrated)**: 0.1505
- **Improvement**: -0.0107

Model was already well-calibrated.

---

## 2. Business-Driven Threshold Optimization

### Business Parameters

- **Customer Lifetime Value**: $2,000
- **Retention Success Rate**: 30%
- **Contact Cost per Customer**: $50
- **Monthly Budget**: $25,000
- **Contact Capacity**: 500 customers/month

### Optimal Threshold Analysis

**Recommended Threshold**: 0.110

**Expected Monthly Performance**:

- **Customers Contacted**: 496
- **True Positives (Churners Identified)**: 238
- **False Positives (False Alarms)**: 258
- **Expected Value Saved**: $142,800
- **Contact Costs**: $24,800
- **Net Profit**: $118,000
- **ROI**: 475.8%

### Business Impact

For every $50 spent on intervention:
- Expected value returned: $287.90
- Break-even if retention success rate > 2.5%

---

## 3. Top At-Risk Customers

Sample of highest-risk customers for immediate intervention:

| customer_id   |   churn_probability | risk_category   |   actual_churn |   MonthlyCharges |   tenure |   is_month_to_month |   charges_per_tenure |   TotalCharges |
|:--------------|--------------------:|:----------------|---------------:|-----------------:|---------:|--------------------:|---------------------:|---------------:|
| CUST_0878     |               0.997 | High            |              1 |         0.387373 | -1.27281 |            0.905628 |             3.66919  |      -0.97112  |
| CUST_0880     |               0.997 | High            |              1 |         0.410504 | -1.27281 |            0.905628 |             3.70883  |      -0.970812 |
| CUST_0213     |               0.996 | High            |              1 |         1.0152   | -1.27281 |            0.905628 |             4.74509  |      -0.96278  |
| CUST_0291     |               0.996 | High            |              1 |         1.14076  | -1.03001 |            0.905628 |             0.745154 |      -0.712578 |
| CUST_0628     |               0.995 | High            |              1 |         1.0218   | -1.27281 |            0.905628 |             4.75641  |      -0.962692 |
| CUST_0752     |               0.995 | High            |              1 |         1.19693  | -1.19188 |            0.905628 |             2.19832  |      -0.859846 |
| CUST_0304     |               0.995 | High            |              1 |         1.00363  | -1.27281 |            0.905628 |             4.72527  |      -0.962933 |
| CUST_0758     |               0.995 | High            |              1 |         0.465025 | -1.27281 |            0.905628 |             3.80226  |      -0.970088 |
| CUST_0751     |               0.995 | High            |              1 |         0.504677 | -1.27281 |            0.905628 |             3.87021  |      -0.969561 |
| CUST_0835     |               0.994 | High            |              1 |         0.322939 | -1.27281 |            0.905628 |             3.55877  |      -0.971976 |

---

## 4. Implementation Recommendations

### Immediate Actions

1. **Deploy model with threshold = 0.110**
2. **Contact top 496 highest-risk customers monthly**
3. **Allocate retention specialists to high-probability cases**
4. **Track intervention success rates to refine model**

### A/B Testing Strategy

- **Control Group**: No intervention (baseline churn rate)
- **Treatment Group**: Targeted retention interventions
- **Target**: Improve retention by 30%
- **Expected Monthly ROI**: 475.8%

