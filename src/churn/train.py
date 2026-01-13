"""
Model training and evaluation for customer churn prediction.
Cross-validated comparison with proper metrics for imbalanced classification.
"""
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
import time

# Handle imports
try:
    from . import config
    from .preprocess import load_processed_data, load_feature_names
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.churn import config
    from src.churn.preprocess import load_processed_data, load_feature_names


def get_model_configs():
    """
    Define model configurations with hyperparameters.
    
    Returns:
        Dictionary of model instances
    """
    return {
        'Dummy (Stratified)': DummyClassifier(strategy='stratified', random_state=config.RANDOM_SEED),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Handle imbalance
            random_state=config.RANDOM_SEED
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=config.RANDOM_SEED
        )
    }


def evaluate_model_cv(model, X, y, cv=5):
    """
    Evaluate model using stratified cross-validation.
    
    Args:
        model: Scikit-learn estimator
        X: Feature matrix
        y: Target vector
        cv: Number of folds
        
    Returns:
        Dictionary of cross-validated metrics
    """
    # Define scoring metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'accuracy': 'accuracy'
    }
    
    # Stratified K-Fold
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)
    
    # Cross-validate
    cv_results = cross_validate(
        model, X, y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Aggregate results
    results = {
        'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
        'cv_roc_auc_std': cv_results['test_roc_auc'].std(),
        'cv_pr_auc_mean': cv_results['test_pr_auc'].mean(),
        'cv_pr_auc_std': cv_results['test_pr_auc'].std(),
        'cv_f1_mean': cv_results['test_f1'].mean(),
        'cv_f1_std': cv_results['test_f1'].std(),
        'cv_precision_mean': cv_results['test_precision'].mean(),
        'cv_precision_std': cv_results['test_precision'].std(),
        'cv_recall_mean': cv_results['test_recall'].mean(),
        'cv_recall_std': cv_results['test_recall'].std(),
        'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
        'cv_accuracy_std': cv_results['test_accuracy'].std(),
    }
    
    return results


def evaluate_model_holdout(model, X_train, y_train, X_val, y_val):
    """
    Train on full training set and evaluate on validation set.
    
    Args:
        model: Scikit-learn estimator
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Dictionary of validation metrics
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    results = {
        'val_roc_auc': roc_auc_score(y_val, y_pred_proba),
        'val_pr_auc': average_precision_score(y_val, y_pred_proba),
        'val_f1': f1_score(y_val, y_pred),
        'val_precision': precision_score(y_val, y_pred),
        'val_recall': recall_score(y_val, y_pred),
        'val_accuracy': (y_pred == y_val).mean(),
        'confusion_matrix': confusion_matrix(y_val, y_pred)
    }
    
    return results, model


def train_and_evaluate_models(X_train, y_train, X_val, y_val, cv_folds=5):
    """
    Train and evaluate all models with cross-validation and holdout validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        cv_folds: Number of CV folds
        
    Returns:
        DataFrame with comparison results, dictionary of trained models
    """
    print("="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)
    
    models = get_model_configs()
    results_list = []
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n[{model_name}]")
        start_time = time.time()
        
        # Cross-validation on training set
        print("  Running cross-validation...")
        cv_results = evaluate_model_cv(model, X_train, y_train, cv=cv_folds)
        
        # Holdout validation
        print("  Training on full training set...")
        val_results, trained_model = evaluate_model_holdout(model, X_train, y_train, X_val, y_val)
        
        elapsed = time.time() - start_time
        
        # Combine results
        result = {
            'model': model_name,
            'train_time_sec': elapsed,
            **cv_results,
            **{k: v for k, v in val_results.items() if k != 'confusion_matrix'}
        }
        results_list.append(result)
        trained_models[model_name] = trained_model
        
        # Print summary
        print(f"  CV ROC-AUC: {cv_results['cv_roc_auc_mean']:.4f} (+/- {cv_results['cv_roc_auc_std']:.4f})")
        print(f"  Val ROC-AUC: {val_results['val_roc_auc']:.4f}")
        print(f"  Val PR-AUC: {val_results['val_pr_auc']:.4f}")
        print(f"  Val F1: {val_results['val_f1']:.4f}")
        print(f"  Time: {elapsed:.2f}s")
    
    results_df = pd.DataFrame(results_list)
    
    return results_df, trained_models


def plot_model_comparison(results_df, output_dir):
    """
    Create visualization comparing model performance.
    
    Args:
        results_df: DataFrame with model results
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC-AUC comparison
    ax = axes[0, 0]
    x_pos = range(len(results_df))
    ax.bar(x_pos, results_df['cv_roc_auc_mean'], 
           yerr=results_df['cv_roc_auc_std'],
           alpha=0.7, capsize=5, color='steelblue', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Cross-Validated ROC-AUC', fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # PR-AUC comparison
    ax = axes[0, 1]
    ax.bar(x_pos, results_df['cv_pr_auc_mean'],
           yerr=results_df['cv_pr_auc_std'],
           alpha=0.7, capsize=5, color='coral', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.set_ylabel('PR-AUC Score')
    ax.set_title('Cross-Validated PR-AUC', fontweight='bold')
    baseline = results_df[results_df['model'] == 'Dummy (Stratified)']['cv_pr_auc_mean'].values[0]
    ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # F1 Score comparison
    ax = axes[1, 0]
    ax.bar(x_pos, results_df['cv_f1_mean'],
           yerr=results_df['cv_f1_std'],
           alpha=0.7, capsize=5, color='seagreen', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('Cross-Validated F1 Score', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Precision vs Recall tradeoff
    ax = axes[1, 1]
    for idx, row in results_df.iterrows():
        ax.scatter(row['cv_recall_mean'], row['cv_precision_mean'], 
                  s=200, alpha=0.7, edgecolor='black', linewidth=2)
        ax.annotate(row['model'], 
                   (row['cv_recall_mean'], row['cv_precision_mean']),
                   textcoords="offset points", xytext=(0,10), ha='center',
                   fontsize=8)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Tradeoff', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved: model_comparison.png")


def plot_confusion_matrices(trained_models, X_val, y_val, output_dir):
    """
    Plot confusion matrices for all models.
    
    Args:
        trained_models: Dictionary of trained model instances
        X_val: Validation features
        y_val: Validation target
        output_dir: Directory to save plots
    """
    n_models = len(trained_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, model) in zip(axes, trained_models.items()):
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=False, square=True,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        
        ax.set_title(f'{model_name}\n(Validation Set)', fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                ax.text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                       ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: confusion_matrices.png")


def save_best_model(trained_models, results_df):
    """
    Save the best performing model based on ROC-AUC.
    
    Args:
        trained_models: Dictionary of trained models
        results_df: DataFrame with model results
    """
    # Select best model by validation ROC-AUC
    best_idx = results_df['val_roc_auc'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_model = trained_models[best_model_name]
    best_score = results_df.loc[best_idx, 'val_roc_auc']
    
    # Save model
    model_path = config.MODELS_DIR / 'best_model.pkl'
    joblib.dump(best_model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'val_roc_auc': best_score,
        'val_pr_auc': results_df.loc[best_idx, 'val_pr_auc'],
        'val_f1': results_df.loc[best_idx, 'val_f1'],
        'model_params': best_model.get_params()
    }
    joblib.dump(metadata, config.MODELS_DIR / 'best_model_metadata.pkl')
    
    print(f"\n[OK] Best model: {best_model_name} (ROC-AUC: {best_score:.4f})")
    print(f"[OK] Saved to: {model_path.name}")
    
    return best_model_name, best_model


def generate_markdown_report(results_df, output_path):
    """
    Generate markdown report with model comparison table.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save markdown file
    """
    with open(output_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write("## Cross-Validation Results (5-Fold Stratified)\n\n")
        
        # Prepare table
        table_df = results_df[['model', 'cv_roc_auc_mean', 'cv_roc_auc_std', 
                              'cv_pr_auc_mean', 'cv_pr_auc_std',
                              'cv_f1_mean', 'cv_f1_std', 'train_time_sec']].copy()
        
        table_df.columns = ['Model', 'ROC-AUC', 'ROC-AUC Std', 'PR-AUC', 'PR-AUC Std', 
                           'F1', 'F1 Std', 'Train Time (s)']
        
        # Format numbers
        for col in ['ROC-AUC', 'PR-AUC', 'F1']:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")
        for col in ['ROC-AUC Std', 'PR-AUC Std', 'F1 Std']:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")
        table_df['Train Time (s)'] = table_df['Train Time (s)'].apply(lambda x: f"{x:.2f}")
        
        f.write(table_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Validation Set Performance\n\n")
        
        val_df = results_df[['model', 'val_roc_auc', 'val_pr_auc', 
                            'val_f1', 'val_precision', 'val_recall']].copy()
        val_df.columns = ['Model', 'ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall']
        
        for col in ['ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall']:
            val_df[col] = val_df[col].apply(lambda x: f"{x:.4f}")
        
        f.write(val_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Key Findings\n\n")
        best_model = results_df.loc[results_df['val_roc_auc'].idxmax(), 'model']
        best_roc = results_df['val_roc_auc'].max()
        f.write(f"- **Best Model**: {best_model} (ROC-AUC: {best_roc:.4f})\n")
        
        baseline_roc = results_df[results_df['model'] == 'Dummy (Stratified)']['val_roc_auc'].values[0]
        improvement = ((best_roc - baseline_roc) / baseline_roc) * 100
        f.write(f"- **Improvement over baseline**: {improvement:.1f}%\n")
        
        f.write(f"\n*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"[OK] Saved markdown report: {output_path.name}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("CUSTOMER CHURN - MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading preprocessed data...")
    data = load_processed_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    
    # Train models
    print("\n[2/5] Training models with cross-validation...")
    results_df, trained_models = train_and_evaluate_models(X_train, y_train, X_val, y_val)
    
    # Save results
    print("\n[3/5] Saving results...")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = config.REPORTS_DIR / 'model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path.name}")
    
    # Generate plots
    print("\n[4/5] Generating visualizations...")
    figures_dir = config.REPORTS_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(results_df, figures_dir)
    plot_confusion_matrices(trained_models, X_val, y_val, figures_dir)
    
    # Save best model and generate report
    print("\n[5/5] Finalizing...")
    best_model_name, best_model = save_best_model(trained_models, results_df)
    
    markdown_path = config.REPORTS_DIR / 'model_comparison.md'
    generate_markdown_report(results_df, markdown_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - {markdown_path}")
    print(f"  - {figures_dir}/model_comparison.png")
    print(f"  - {figures_dir}/confusion_matrices.png")


if __name__ == "__main__":
    main()
