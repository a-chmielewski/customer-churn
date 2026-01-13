"""
Final test set evaluation for customer churn prediction.
Evaluate best model on held-out test data.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys

# Handle imports
try:
    from . import config
    from .preprocess import load_processed_data, load_feature_names
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.churn import config
    from src.churn.preprocess import load_processed_data, load_feature_names


def load_best_model():
    """Load the best trained model and its metadata."""
    model = joblib.load(config.MODELS_DIR / 'best_model.pkl')
    metadata = joblib.load(config.MODELS_DIR / 'best_model_metadata.pkl')
    return model, metadata


def evaluate_test_set(model, X_test, y_test):
    """
    Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of test metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'test_pr_auc': average_precision_score(y_test, y_pred_proba),
        'test_f1': f1_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def plot_roc_curve(y_test, y_pred_proba, output_dir):
    """Plot ROC curve for test set."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})', color='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Test Set', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: test_roc_curve.png")


def plot_precision_recall_curve(y_test, y_pred_proba, output_dir):
    """Plot Precision-Recall curve for test set."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    baseline = y_test.mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.4f})', color='coral')
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=2, 
               label=f'Baseline (No Skill = {baseline:.4f})', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve - Test Set', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: test_pr_curve.png")


def plot_confusion_matrix(cm, output_dir):
    """Plot confusion matrix with detailed annotations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               cbar=True, square=True,
               xticklabels=['No Churn (0)', 'Churn (1)'],
               yticklabels=['No Churn (0)', 'Churn (1)'])
    
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j+0.5, i+0.75, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    # Add metrics annotations
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    info_text = f'True Negatives: {tn}\nFalse Positives: {fp}\n'
    info_text += f'False Negatives: {fn}\nTrue Positives: {tp}\n\n'
    info_text += f'Sensitivity (Recall): {sensitivity:.3f}\n'
    info_text += f'Specificity: {specificity:.3f}'
    
    ax.text(1.4, 0.5, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: test_confusion_matrix.png")


def plot_threshold_analysis(y_test, y_pred_proba, output_dir):
    """Analyze different probability thresholds."""
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        if y_pred_thresh.sum() > 0:  # Avoid division by zero
            precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
            recalls.append(recall_score(y_test, y_pred_thresh))
            f1_scores.append(f1_score(y_test, y_pred_thresh))
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label='Precision', linewidth=2, color='steelblue')
    ax.plot(thresholds, recalls, label='Recall', linewidth=2, color='coral')
    ax.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, color='seagreen')
    
    # Mark default threshold
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Default (0.5)')
    
    ax.set_xlabel('Probability Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Metrics vs. Probability Threshold', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: test_threshold_analysis.png")


def compare_val_vs_test(metadata, test_metrics):
    """
    Compare validation vs test performance.
    
    Args:
        metadata: Model metadata with validation metrics
        test_metrics: Test set metrics
        
    Returns:
        DataFrame comparing metrics
    """
    # Load validation metrics from model comparison CSV
    try:
        model_comp = pd.read_csv(config.REPORTS_DIR / 'model_comparison.csv')
        best_row = model_comp[model_comp['model'] == metadata['model_name']].iloc[0]
        val_precision = best_row['val_precision']
        val_recall = best_row['val_recall']
    except:
        val_precision = metadata.get('val_precision', 0)
        val_recall = metadata.get('val_recall', 0)
    
    comparison = pd.DataFrame({
        'Metric': ['ROC-AUC', 'PR-AUC', 'F1 Score', 'Precision', 'Recall'],
        'Validation': [
            metadata['val_roc_auc'],
            metadata['val_pr_auc'],
            metadata['val_f1'],
            val_precision,
            val_recall
        ],
        'Test': [
            test_metrics['test_roc_auc'],
            test_metrics['test_pr_auc'],
            test_metrics['test_f1'],
            test_metrics['test_precision'],
            test_metrics['test_recall']
        ]
    })
    
    comparison['Difference'] = comparison['Test'] - comparison['Validation']
    comparison['% Change'] = (comparison['Difference'] / comparison['Validation'] * 100).replace([np.inf, -np.inf], 0)
    
    return comparison


def generate_test_report(model_name, test_metrics, comparison_df, output_path):
    """Generate comprehensive test evaluation report."""
    with open(output_path, 'w') as f:
        f.write("# Final Test Set Evaluation Report\n\n")
        f.write(f"**Model**: {model_name}\n\n")
        f.write("---\n\n")
        
        f.write("## Test Set Performance\n\n")
        f.write("| Metric | Score |\n")
        f.write("|:-------|------:|\n")
        f.write(f"| ROC-AUC | {test_metrics['test_roc_auc']:.4f} |\n")
        f.write(f"| PR-AUC | {test_metrics['test_pr_auc']:.4f} |\n")
        f.write(f"| F1 Score | {test_metrics['test_f1']:.4f} |\n")
        f.write(f"| Precision | {test_metrics['test_precision']:.4f} |\n")
        f.write(f"| Recall | {test_metrics['test_recall']:.4f} |\n")
        f.write(f"| Accuracy | {test_metrics['test_accuracy']:.4f} |\n\n")
        
        f.write("## Confusion Matrix\n\n")
        cm = test_metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        f.write("```\n")
        f.write("                Predicted\n")
        f.write("              No Churn  Churn\n")
        f.write(f"Actual No  |    {tn:4d}    {fp:4d}\n")
        f.write(f"       Yes |    {fn:4d}    {tp:4d}\n")
        f.write("```\n\n")
        
        f.write(f"- **True Negatives (TN)**: {tn} - Correctly predicted no churn\n")
        f.write(f"- **False Positives (FP)**: {fp} - Incorrectly predicted churn\n")
        f.write(f"- **False Negatives (FN)**: {fn} - Missed churners\n")
        f.write(f"- **True Positives (TP)**: {tp} - Correctly predicted churn\n\n")
        
        f.write("## Validation vs Test Comparison\n\n")
        f.write(comparison_df.to_markdown(index=False, floatfmt='.4f'))
        f.write("\n\n")
        
        f.write("## Interpretation\n\n")
        
        # Check for overfitting
        roc_diff = comparison_df[comparison_df['Metric'] == 'ROC-AUC']['Difference'].values[0]
        if abs(roc_diff) < 0.02:
            f.write("[OK] **Model Generalization**: Excellent - Test performance closely matches validation (ROC-AUC difference < 0.02)\n\n")
        elif abs(roc_diff) < 0.05:
            f.write("[OK] **Model Generalization**: Good - Minor difference between validation and test\n\n")
        else:
            f.write("[WARNING] **Model Generalization**: Test performance differs significantly from validation\n\n")
        
        # Business interpretation
        f.write("### Business Impact\n\n")
        total_test = len(test_metrics['y_pred'])
        churn_rate = test_metrics['y_pred'].sum() / total_test
        actual_churners = test_metrics['confusion_matrix'][1, :].sum()
        
        f.write(f"- **Test Set Size**: {total_test:,} customers\n")
        f.write(f"- **Actual Churners**: {actual_churners} ({actual_churners/total_test:.1%})\n")
        f.write(f"- **Predicted Churners**: {tp + fp} ({(tp + fp)/total_test:.1%})\n")
        f.write(f"- **Correctly Identified**: {tp} churners ({tp/actual_churners:.1%} of all churners)\n")
        f.write(f"- **False Alarms**: {fp} customers ({fp/(tp+fp) if (tp+fp) > 0 else 0:.1%} of predictions)\n\n")
        
        f.write("### Recommendations\n\n")
        if test_metrics['test_precision'] > 0.6:
            f.write("- **High Precision**: Good for targeted interventions with limited budget\n")
        if test_metrics['test_recall'] > 0.5:
            f.write("- **Moderate Recall**: Captures majority of at-risk customers\n")
        
        f.write(f"\n*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"[OK] Saved: {output_path.name}")


def get_feature_importance(model, feature_names, top_n=15):
    """Extract feature importance from model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        return feature_imp
    return None


def plot_feature_importance(feature_imp, output_dir):
    """Plot top feature importances."""
    if feature_imp is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(feature_imp)), feature_imp['importance'], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(feature_imp)))
    ax.set_yticklabels(feature_imp['feature'])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: feature_importance.png")


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading test data...")
    data = load_processed_data()
    X_test = data['X_test']
    y_test = data['y_test']
    print(f"  Test set: {len(y_test):,} samples")
    
    # Load model
    print("\n[2/5] Loading best model...")
    model, metadata = load_best_model()
    print(f"  Model: {metadata['model_name']}")
    print(f"  Validation ROC-AUC: {metadata['val_roc_auc']:.4f}")
    
    # Evaluate on test set
    print("\n[3/5] Evaluating on test set...")
    test_metrics = evaluate_test_set(model, X_test, y_test)
    
    print("\nTest Set Metrics:")
    print(f"  ROC-AUC:   {test_metrics['test_roc_auc']:.4f}")
    print(f"  PR-AUC:    {test_metrics['test_pr_auc']:.4f}")
    print(f"  F1 Score:  {test_metrics['test_f1']:.4f}")
    print(f"  Precision: {test_metrics['test_precision']:.4f}")
    print(f"  Recall:    {test_metrics['test_recall']:.4f}")
    print(f"  Accuracy:  {test_metrics['test_accuracy']:.4f}")
    
    # Generate visualizations
    print("\n[4/5] Generating visualizations...")
    figures_dir = config.REPORTS_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_roc_curve(y_test, test_metrics['y_pred_proba'], figures_dir)
    plot_precision_recall_curve(y_test, test_metrics['y_pred_proba'], figures_dir)
    plot_confusion_matrix(test_metrics['confusion_matrix'], figures_dir)
    plot_threshold_analysis(y_test, test_metrics['y_pred_proba'], figures_dir)
    
    # Feature importance
    feature_names = load_feature_names()
    feature_imp = get_feature_importance(model, feature_names)
    if feature_imp is not None:
        plot_feature_importance(feature_imp, figures_dir)
        # Save to CSV
        feature_imp.to_csv(config.REPORTS_DIR / 'feature_importance.csv', index=False)
        print("[OK] Saved: feature_importance.csv")
    
    # Compare validation vs test
    print("\n[5/5] Generating final report...")
    comparison_df = compare_val_vs_test(metadata, test_metrics)
    
    print("\nValidation vs Test Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save reports
    report_path = config.REPORTS_DIR / 'test_evaluation.md'
    generate_test_report(metadata['model_name'], test_metrics, comparison_df, report_path)
    
    # Save test metrics
    test_results = {
        'model_name': metadata['model_name'],
        **{k: v for k, v in test_metrics.items() if k not in ['confusion_matrix', 'y_pred', 'y_pred_proba']}
    }
    pd.DataFrame([test_results]).to_csv(config.REPORTS_DIR / 'test_results.csv', index=False)
    print("[OK] Saved: test_results.csv")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nFinal Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
    print(f"Model ready for deployment: {metadata['model_name']}")


if __name__ == "__main__":
    main()
