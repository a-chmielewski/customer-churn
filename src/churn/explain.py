"""
Model interpretation and business decision optimization.
Includes SHAP analysis, calibration, and threshold optimization.
"""
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
import shap

# Handle imports
try:
    from . import config
    from .preprocess import load_processed_data, load_feature_names
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.churn import config
    from src.churn.preprocess import load_processed_data, load_feature_names


# Business parameters (adjust based on actual business context)
BUSINESS_PARAMS = {
    'customer_lifetime_value': 2000,  # Average CLV
    'retention_success_rate': 0.3,    # 30% of contacted churners stay
    'contact_cost': 50,                # Cost per customer contact/intervention
    'monthly_budget': 25000,           # Monthly retention budget
    'max_contacts_per_month': 500      # Contact capacity constraint
}


def load_best_model():
    """Load the best trained model and metadata."""
    model = joblib.load(config.MODELS_DIR / 'best_model.pkl')
    metadata = joblib.load(config.MODELS_DIR / 'best_model_metadata.pkl')
    return model, metadata


def calibrate_model(model, X_train, y_train, X_val, y_val, method='sigmoid'):
    """
    Calibrate model probabilities using validation set.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        method: 'sigmoid' or 'isotonic'
        
    Returns:
        Calibrated model, calibration metrics
    """
    print(f"\n[Calibrating model using {method} method...]")
    
    # Get uncalibrated probabilities
    y_pred_proba_uncal = model.predict_proba(X_val)[:, 1]
    brier_uncal = brier_score_loss(y_val, y_pred_proba_uncal)
    
    # Calibrate model (use prefit since model is already trained)
    calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit')
    calibrated_model.fit(X_train, y_train)
    
    # Get calibrated probabilities
    y_pred_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
    brier_cal = brier_score_loss(y_val, y_pred_proba_cal)
    
    metrics = {
        'brier_score_uncalibrated': brier_uncal,
        'brier_score_calibrated': brier_cal,
        'improvement': brier_uncal - brier_cal
    }
    
    print(f"  Brier Score (uncalibrated): {brier_uncal:.4f}")
    print(f"  Brier Score (calibrated):   {brier_cal:.4f}")
    print(f"  Improvement:                {metrics['improvement']:.4f}")
    
    return calibrated_model, metrics


def plot_calibration_curve(model, calibrated_model, X_val, y_val, output_dir):
    """Plot calibration curves before and after calibration."""
    # Uncalibrated predictions
    y_pred_proba_uncal = model.predict_proba(X_val)[:, 1]
    fraction_of_positives_uncal, mean_predicted_value_uncal = calibration_curve(
        y_val, y_pred_proba_uncal, n_bins=10, strategy='uniform'
    )
    
    # Calibrated predictions
    y_pred_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
        y_val, y_pred_proba_cal, n_bins=10, strategy='uniform'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
    
    # Uncalibrated
    ax.plot(mean_predicted_value_uncal, fraction_of_positives_uncal, 
           marker='o', linewidth=2, markersize=8, label='Uncalibrated', color='coral')
    
    # Calibrated
    ax.plot(mean_predicted_value_cal, fraction_of_positives_cal,
           marker='s', linewidth=2, markersize=8, label='Calibrated', color='steelblue')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives (Actual)', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Curve Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: calibration_curve.png")


def calculate_profit_at_threshold(y_true, y_pred_proba, threshold, params):
    """
    Calculate expected profit at a given threshold.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        params: Business parameters dict
        
    Returns:
        Dictionary with profit metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix elements
    tp = ((y_pred == 1) & (y_true == 1)).sum()  # True positives
    fp = ((y_pred == 1) & (y_true == 0)).sum()  # False positives
    fn = ((y_pred == 0) & (y_true == 1)).sum()  # False negatives
    tn = ((y_pred == 0) & (y_true == 0)).sum()  # True negatives
    
    total_contacted = tp + fp
    
    # Expected value calculations
    # Value from preventing churn (TP * success_rate * CLV)
    value_saved = tp * params['retention_success_rate'] * params['customer_lifetime_value']
    
    # Cost of contacting customers
    contact_cost = total_contacted * params['contact_cost']
    
    # Net profit
    net_profit = value_saved - contact_cost
    
    # Cost of missed opportunities (FN who would have stayed if contacted)
    missed_value = fn * params['retention_success_rate'] * params['customer_lifetime_value']
    
    return {
        'threshold': threshold,
        'total_contacted': total_contacted,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'value_saved': value_saved,
        'contact_cost': contact_cost,
        'net_profit': net_profit,
        'missed_value': missed_value,
        'roi': (value_saved / contact_cost - 1) * 100 if contact_cost > 0 else 0
    }


def optimize_threshold_for_profit(y_true, y_pred_proba, params, thresholds=None):
    """
    Find optimal threshold based on expected profit.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        params: Business parameters
        thresholds: List of thresholds to evaluate
        
    Returns:
        DataFrame with profit analysis, optimal threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    
    results = []
    for threshold in thresholds:
        metrics = calculate_profit_at_threshold(y_true, y_pred_proba, threshold, params)
        
        # Check capacity constraint
        if metrics['total_contacted'] <= params['max_contacts_per_month']:
            metrics['within_capacity'] = True
        else:
            metrics['within_capacity'] = False
            
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold (max profit within capacity)
    feasible = results_df[results_df['within_capacity']]
    if len(feasible) > 0:
        optimal_idx = feasible['net_profit'].idxmax()
        optimal_threshold = feasible.loc[optimal_idx, 'threshold']
    else:
        optimal_idx = results_df['net_profit'].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    return results_df, optimal_threshold


def plot_profit_curve(results_df, optimal_threshold, params, output_dir):
    """Plot profit vs threshold with capacity constraint."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Profit curve
    ax1.plot(results_df['threshold'], results_df['net_profit'], 
            linewidth=2, color='seagreen', label='Net Profit')
    ax1.axvline(x=optimal_threshold, color='red', linestyle='--', 
               linewidth=2, label=f'Optimal Threshold ({optimal_threshold:.3f})')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax1.set_xlabel('Probability Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Expected Monthly Net Profit ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Expected Profit vs Classification Threshold', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Contacts vs threshold with capacity line
    ax2.plot(results_df['threshold'], results_df['total_contacted'],
            linewidth=2, color='steelblue', label='Customers Contacted')
    ax2.axhline(y=params['max_contacts_per_month'], color='red', 
               linestyle='--', linewidth=2, label=f'Capacity Limit ({params["max_contacts_per_month"]})')
    ax2.axvline(x=optimal_threshold, color='red', linestyle='--', 
               linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('Probability Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Customers Contacted', fontsize=12, fontweight='bold')
    ax2.set_title('Contact Volume vs Threshold', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'profit_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: profit_threshold_analysis.png")


def compute_shap_values(model, X_sample, feature_names, sample_size=100):
    """
    Compute SHAP values for model interpretation.
    
    Args:
        model: Trained model
        X_sample: Sample of features
        feature_names: List of feature names
        sample_size: Number of samples for SHAP analysis
        
    Returns:
        SHAP explainer and values
    """
    print(f"\n[Computing SHAP values on {sample_size} samples...]")
    
    # Sample data for efficiency
    if len(X_sample) > sample_size:
        idx = np.random.choice(len(X_sample), sample_size, replace=False)
        X_shap = X_sample[idx]
    else:
        X_shap = X_sample
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    print(f"  SHAP values computed for {len(X_shap)} samples")
    
    return explainer, shap_values, X_shap


def plot_shap_summary(shap_values, X_shap, feature_names, output_dir):
    """Plot SHAP summary plots."""
    # Summary plot (bar)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, 
                     plot_type='bar', show=False)
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: shap_importance.png")
    
    # Summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: shap_summary.png")


def generate_top_risk_customers(model, X_test, y_test, feature_names, 
                                calibrated_model=None, top_n=20):
    """
    Generate table of top at-risk customers with predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: Feature names
        calibrated_model: Calibrated model (optional)
        top_n: Number of customers to show
        
    Returns:
        DataFrame with top at-risk customers
    """
    # Get predictions
    if calibrated_model is not None:
        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'customer_id': [f'CUST_{i:04d}' for i in range(len(y_test))],
        'churn_probability': y_pred_proba,
        'actual_churn': y_test,
        'risk_category': pd.cut(y_pred_proba, 
                               bins=[0, 0.3, 0.6, 1.0],
                               labels=['Low', 'Medium', 'High'])
    })
    
    # Add key features
    df_features = pd.DataFrame(X_test, columns=feature_names)
    
    # Select most important features to display
    key_features = ['MonthlyCharges', 'tenure', 'is_month_to_month', 
                   'charges_per_tenure', 'TotalCharges']
    available_features = [f for f in key_features if f in df_features.columns]
    
    for feat in available_features:
        results[feat] = df_features[feat].values
    
    # Sort by probability and get top N
    top_risk = results.nlargest(top_n, 'churn_probability')
    
    return top_risk


def generate_interpretation_report(optimal_threshold, profit_metrics, 
                                  calibration_metrics, top_risk, output_path):
    """Generate comprehensive interpretation report."""
    with open(output_path, 'w') as f:
        f.write("# Model Interpretation & Business Decision Report\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Model Calibration\n\n")
        f.write("### Calibration Quality\n\n")
        f.write(f"- **Brier Score (Uncalibrated)**: {calibration_metrics['brier_score_uncalibrated']:.4f}\n")
        f.write(f"- **Brier Score (Calibrated)**: {calibration_metrics['brier_score_calibrated']:.4f}\n")
        f.write(f"- **Improvement**: {calibration_metrics['improvement']:.4f}\n\n")
        
        if calibration_metrics['improvement'] > 0:
            f.write("Calibration improved probability estimates (lower Brier score is better).\n\n")
        else:
            f.write("Model was already well-calibrated.\n\n")
        
        f.write("---\n\n")
        
        f.write("## 2. Business-Driven Threshold Optimization\n\n")
        f.write("### Business Parameters\n\n")
        f.write(f"- **Customer Lifetime Value**: ${BUSINESS_PARAMS['customer_lifetime_value']:,}\n")
        f.write(f"- **Retention Success Rate**: {BUSINESS_PARAMS['retention_success_rate']:.0%}\n")
        f.write(f"- **Contact Cost per Customer**: ${BUSINESS_PARAMS['contact_cost']}\n")
        f.write(f"- **Monthly Budget**: ${BUSINESS_PARAMS['monthly_budget']:,}\n")
        f.write(f"- **Contact Capacity**: {BUSINESS_PARAMS['max_contacts_per_month']} customers/month\n\n")
        
        f.write("### Optimal Threshold Analysis\n\n")
        f.write(f"**Recommended Threshold**: {optimal_threshold:.3f}\n\n")
        
        opt_metrics = profit_metrics[profit_metrics['threshold'] == optimal_threshold].iloc[0]
        
        f.write("**Expected Monthly Performance**:\n\n")
        f.write(f"- **Customers Contacted**: {opt_metrics['total_contacted']:.0f}\n")
        f.write(f"- **True Positives (Churners Identified)**: {opt_metrics['true_positives']:.0f}\n")
        f.write(f"- **False Positives (False Alarms)**: {opt_metrics['false_positives']:.0f}\n")
        f.write(f"- **Expected Value Saved**: ${opt_metrics['value_saved']:,.0f}\n")
        f.write(f"- **Contact Costs**: ${opt_metrics['contact_cost']:,.0f}\n")
        f.write(f"- **Net Profit**: ${opt_metrics['net_profit']:,.0f}\n")
        f.write(f"- **ROI**: {opt_metrics['roi']:.1f}%\n\n")
        
        f.write("### Business Impact\n\n")
        f.write(f"For every ${BUSINESS_PARAMS['contact_cost']} spent on intervention:\n")
        value_per_contact = opt_metrics['value_saved'] / opt_metrics['total_contacted'] if opt_metrics['total_contacted'] > 0 else 0
        f.write(f"- Expected value returned: ${value_per_contact:.2f}\n")
        f.write(f"- Break-even if retention success rate > {(BUSINESS_PARAMS['contact_cost'] / BUSINESS_PARAMS['customer_lifetime_value']):.1%}\n\n")
        
        f.write("---\n\n")
        
        f.write("## 3. Top At-Risk Customers\n\n")
        f.write("Sample of highest-risk customers for immediate intervention:\n\n")
        
        # Format table
        display_cols = ['customer_id', 'churn_probability', 'risk_category', 'actual_churn']
        display_cols += [col for col in top_risk.columns if col not in display_cols and col != 'Unnamed: 0']
        
        top_risk_display = top_risk[display_cols].head(10).copy()
        top_risk_display['churn_probability'] = top_risk_display['churn_probability'].apply(lambda x: f"{x:.3f}")
        
        f.write(top_risk_display.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("---\n\n")
        
        f.write("## 4. Implementation Recommendations\n\n")
        f.write("### Immediate Actions\n\n")
        f.write(f"1. **Deploy model with threshold = {optimal_threshold:.3f}**\n")
        f.write(f"2. **Contact top {int(opt_metrics['total_contacted'])} highest-risk customers monthly**\n")
        f.write("3. **Allocate retention specialists to high-probability cases**\n")
        f.write("4. **Track intervention success rates to refine model**\n\n")
        
        f.write("### A/B Testing Strategy\n\n")
        f.write("- **Control Group**: No intervention (baseline churn rate)\n")
        f.write("- **Treatment Group**: Targeted retention interventions\n")
        f.write(f"- **Target**: Improve retention by {BUSINESS_PARAMS['retention_success_rate']:.0%}\n")
        f.write(f"- **Expected Monthly ROI**: {opt_metrics['roi']:.1f}%\n\n")
        
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"[OK] Saved: {output_path.name}")


def main():
    """Main interpretation pipeline."""
    print("="*60)
    print("MODEL INTERPRETATION & BUSINESS OPTIMIZATION")
    print("="*60)
    
    # Load data
    print("\n[1/7] Loading data and model...")
    data = load_processed_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    model, metadata = load_best_model()
    feature_names = load_feature_names()
    print(f"  Model: {metadata['model_name']}")
    
    # Calibrate model
    print("\n[2/7] Calibrating probabilities...")
    calibrated_model, calibration_metrics = calibrate_model(
        model, X_train, y_train, X_val, y_val, method='sigmoid'
    )
    
    # Save calibrated model
    joblib.dump(calibrated_model, config.MODELS_DIR / 'calibrated_model.pkl')
    print("  Saved: calibrated_model.pkl")
    
    # Generate visualizations
    print("\n[3/7] Generating calibration plots...")
    figures_dir = config.REPORTS_DIR / 'figures'
    plot_calibration_curve(model, calibrated_model, X_val, y_val, figures_dir)
    
    # Optimize threshold for business profit
    print("\n[4/7] Optimizing threshold for business profit...")
    y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    profit_df, optimal_threshold = optimize_threshold_for_profit(
        y_test, y_pred_proba_cal, BUSINESS_PARAMS
    )
    
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    opt_metrics = profit_df[profit_df['threshold'] == optimal_threshold].iloc[0]
    print(f"  Expected monthly net profit: ${opt_metrics['net_profit']:,.0f}")
    print(f"  Customers to contact: {opt_metrics['total_contacted']:.0f}")
    print(f"  ROI: {opt_metrics['roi']:.1f}%")
    
    # Plot profit curves
    plot_profit_curve(profit_df, optimal_threshold, BUSINESS_PARAMS, figures_dir)
    
    # SHAP analysis
    print("\n[5/7] Computing SHAP values...")
    explainer, shap_values, X_shap = compute_shap_values(
        model, X_test, feature_names, sample_size=100
    )
    
    print("\n[6/7] Generating SHAP visualizations...")
    plot_shap_summary(shap_values, X_shap, feature_names, figures_dir)
    
    # Top at-risk customers
    print("\n[7/7] Identifying top at-risk customers...")
    top_risk = generate_top_risk_customers(
        model, X_test, y_test, feature_names, calibrated_model, top_n=50
    )
    
    # Save results
    top_risk.to_csv(config.REPORTS_DIR / 'top_risk_customers.csv', index=False)
    print("  Saved: top_risk_customers.csv")
    
    profit_df.to_csv(config.REPORTS_DIR / 'profit_analysis.csv', index=False)
    print("  Saved: profit_analysis.csv")
    
    # Generate comprehensive report
    report_path = config.REPORTS_DIR / 'interpretation_report.md'
    generate_interpretation_report(
        optimal_threshold, profit_df, calibration_metrics, top_risk, report_path
    )
    
    print("\n" + "="*60)
    print("INTERPRETATION COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  Expected Monthly Profit: ${opt_metrics['net_profit']:,.0f}")
    print(f"  Contact Volume: {opt_metrics['total_contacted']:.0f} customers")
    print(f"  ROI: {opt_metrics['roi']:.1f}%")
    print(f"\nArtifacts Generated:")
    print(f"  - calibrated_model.pkl")
    print(f"  - calibration_curve.png")
    print(f"  - profit_threshold_analysis.png")
    print(f"  - shap_importance.png")
    print(f"  - shap_summary.png")
    print(f"  - top_risk_customers.csv")
    print(f"  - profit_analysis.csv")
    print(f"  - interpretation_report.md")


if __name__ == "__main__":
    main()
