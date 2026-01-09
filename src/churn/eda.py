"""
Exploratory Data Analysis for customer churn prediction.
Focus on business-relevant insights, not exhaustive exploration.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Handle imports for both direct execution and module import
try:
    from . import config
    from .data_load import load_and_validate_data
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.churn import config
    from src.churn.data_load import load_and_validate_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data issues identified during validation.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Fix TotalCharges - convert to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing TotalCharges (likely new customers with tenure=0)
    # Fill with MonthlyCharges for consistency
    missing_mask = df['TotalCharges'].isna()
    df.loc[missing_mask, 'TotalCharges'] = df.loc[missing_mask, 'MonthlyCharges']
    
    print(f"Cleaned {missing_mask.sum()} missing TotalCharges values")
    
    return df


def analyze_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze and visualize target variable distribution.
    
    Args:
        df: DataFrame with cleaned data
        output_dir: Directory to save figures
    """
    churn_counts = df[config.TARGET_COLUMN].value_counts()
    churn_rate = (df[config.TARGET_COLUMN] == 'Yes').mean()
    
    print(f"\n{'='*60}")
    print("TARGET DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Total customers: {len(df):,}")
    print(f"Churned: {churn_counts.get('Yes', 0):,} ({churn_rate:.1%})")
    print(f"Retained: {churn_counts.get('No', 0):,} ({1-churn_rate:.1%})")
    print(f"Class imbalance ratio: {churn_counts['No'] / churn_counts['Yes']:.1f}:1")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(churn_counts.index, churn_counts.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Customer Status', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Add percentage labels on bars
    for bar, count in zip(bars, churn_counts.values):
        height = bar.get_height()
        percentage = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: 01_churn_distribution.png")


def analyze_churn_vs_monthly_charges(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze relationship between churn and monthly charges.
    
    Args:
        df: DataFrame with cleaned data
        output_dir: Directory to save figures
    """
    print(f"\n{'='*60}")
    print("CHURN VS MONTHLY CHARGES")
    print(f"{'='*60}")
    
    # Statistics
    churn_stats = df.groupby(config.TARGET_COLUMN)['MonthlyCharges'].describe()
    print(churn_stats[['mean', '50%', 'std']])
    
    # Visualization: Box plot + violin
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    box_data = [df[df[config.TARGET_COLUMN] == 'No']['MonthlyCharges'],
                df[df[config.TARGET_COLUMN] == 'Yes']['MonthlyCharges']]
    bp = ax1.boxplot(box_data, labels=['No Churn', 'Churned'],
                     patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Monthly Charges ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Monthly Charges Distribution by Churn', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Histogram with overlay
    ax2.hist(df[df[config.TARGET_COLUMN] == 'No']['MonthlyCharges'],
             bins=30, alpha=0.6, label='No Churn', color='#2ecc71', edgecolor='black')
    ax2.hist(df[df[config.TARGET_COLUMN] == 'Yes']['MonthlyCharges'],
             bins=30, alpha=0.6, label='Churned', color='#e74c3c', edgecolor='black')
    ax2.set_xlabel('Monthly Charges ($)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
    ax2.set_title('Monthly Charges Overlap', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_churn_vs_monthly_charges.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: 02_churn_vs_monthly_charges.png")


def analyze_churn_vs_contract(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze relationship between churn and contract type.
    
    Args:
        df: DataFrame with cleaned data
        output_dir: Directory to save figures
    """
    print(f"\n{'='*60}")
    print("CHURN VS CONTRACT TYPE")
    print(f"{'='*60}")
    
    # Calculate churn rate by contract type
    contract_churn = df.groupby('Contract')[config.TARGET_COLUMN].apply(
        lambda x: (x == 'Yes').sum()
    )
    contract_total = df.groupby('Contract').size()
    contract_rate = (contract_churn / contract_total * 100).sort_values(ascending=False)
    
    print("\nChurn Rate by Contract Type:")
    for contract, rate in contract_rate.items():
        count = contract_churn[contract]
        total = contract_total[contract]
        print(f"  {contract:20s}: {rate:5.1f}% ({count:,} / {total:,})")
    
    # Visualization: Grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    contract_pivot = df.groupby(['Contract', config.TARGET_COLUMN]).size().unstack(fill_value=0)
    contract_pivot = contract_pivot.reindex(contract_rate.index)  # Sort by churn rate
    
    x = np.arange(len(contract_pivot))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, contract_pivot['No'], width, 
                   label='No Churn', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, contract_pivot['Yes'], width,
                   label='Churned', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Contract Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Customer Churn by Contract Type', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(contract_pivot.index, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add churn rate labels
    for i, (contract, rate) in enumerate(contract_rate.items()):
        ax.text(i, contract_pivot.loc[contract].sum() + 100,
                f'{rate:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#c0392b')
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_churn_vs_contract.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: 03_churn_vs_contract.png")


def analyze_churn_vs_tenure(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze relationship between churn and customer tenure.
    
    Args:
        df: DataFrame with cleaned data
        output_dir: Directory to save figures
    """
    print(f"\n{'='*60}")
    print("CHURN VS TENURE")
    print(f"{'='*60}")
    
    # Statistics
    tenure_stats = df.groupby(config.TARGET_COLUMN)['tenure'].describe()
    print(tenure_stats[['mean', '50%', 'std']])
    
    # Create tenure bins for analysis
    df['tenure_group'] = pd.cut(df['tenure'], 
                                 bins=[0, 12, 24, 48, 72],
                                 labels=['0-12 months', '13-24 months', 
                                        '25-48 months', '49-72 months'])
    
    # Visualization: Histogram + Line plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram
    ax1.hist(df[df[config.TARGET_COLUMN] == 'No']['tenure'],
             bins=72, alpha=0.6, label='No Churn', color='#2ecc71', edgecolor='black')
    ax1.hist(df[df[config.TARGET_COLUMN] == 'Yes']['tenure'],
             bins=72, alpha=0.6, label='Churned', color='#e74c3c', edgecolor='black')
    ax1.set_xlabel('Tenure (months)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
    ax1.set_title('Tenure Distribution by Churn Status', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Churn rate by tenure
    tenure_bins = range(0, 73, 6)  # 6-month bins
    churn_by_tenure = []
    bin_centers = []
    
    for i in range(len(tenure_bins)-1):
        mask = (df['tenure'] >= tenure_bins[i]) & (df['tenure'] < tenure_bins[i+1])
        if mask.sum() > 0:
            churn_rate = (df[mask][config.TARGET_COLUMN] == 'Yes').mean() * 100
            churn_by_tenure.append(churn_rate)
            bin_centers.append((tenure_bins[i] + tenure_bins[i+1]) / 2)
    
    ax2.plot(bin_centers, churn_by_tenure, marker='o', linewidth=2, 
             markersize=8, color='#e74c3c')
    ax2.fill_between(bin_centers, churn_by_tenure, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Tenure (months)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Churn Rate Trend by Tenure', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=df[config.TARGET_COLUMN].eq('Yes').mean()*100, 
                color='gray', linestyle='--', label='Overall Churn Rate')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_churn_vs_tenure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: 04_churn_vs_tenure.png")


def analyze_combined_risk_factors(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze combined impact of key risk factors.
    
    Args:
        df: DataFrame with cleaned data
        output_dir: Directory to save figures
    """
    print(f"\n{'='*60}")
    print("COMBINED RISK FACTORS")
    print(f"{'='*60}")
    
    # Create risk segments
    df['high_charges'] = df['MonthlyCharges'] > df['MonthlyCharges'].median()
    df['new_customer'] = df['tenure'] <= 12
    df['month_to_month'] = df['Contract'] == 'Month-to-month'
    
    # Calculate churn rate for different combinations
    risk_factors = []
    for mtm in [True, False]:
        for new in [True, False]:
            for high in [True, False]:
                mask = (df['month_to_month'] == mtm) & \
                       (df['new_customer'] == new) & \
                       (df['high_charges'] == high)
                if mask.sum() > 0:
                    churn_rate = (df[mask][config.TARGET_COLUMN] == 'Yes').mean() * 100
                    count = mask.sum()
                    risk_factors.append({
                        'Month-to-Month': 'Yes' if mtm else 'No',
                        'New (<12mo)': 'Yes' if new else 'No',
                        'High Charges': 'Yes' if high else 'No',
                        'Churn Rate': churn_rate,
                        'Count': count
                    })
    
    risk_df = pd.DataFrame(risk_factors).sort_values('Churn Rate', ascending=False)
    
    print("\nTop Risk Segments:")
    print(risk_df.head(5).to_string(index=False))
    
    # Visualization: Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot for heatmap
    pivot_data = risk_df[risk_df['Month-to-Month'] == 'Yes'].pivot_table(
        values='Churn Rate',
        index='New (<12mo)',
        columns='High Charges'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Churn Rate (%)'},
                linewidths=2, linecolor='white',
                vmin=0, vmax=100, ax=ax)
    
    ax.set_title('Churn Risk Heatmap\n(Month-to-Month Contracts Only)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('High Monthly Charges', fontsize=12, fontweight='bold')
    ax.set_ylabel('New Customer', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_risk_factors_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: 05_risk_factors_heatmap.png")


def generate_summary_report(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate summary statistics report.
    
    Args:
        df: DataFrame with cleaned data
        output_dir: Directory to save report
    """
    print(f"\n{'='*60}")
    print("DATA QUALITY SUMMARY")
    print(f"{'='*60}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing Values:")
        print(missing[missing > 0])
    else:
        print("\n[OK] No missing values after cleaning")
    
    # Key metrics
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {df.shape[1]}")
    print(f"Samples: {df.shape[0]:,}")
    
    # Save summary to file
    summary_path = output_dir.parent / 'eda_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset: {config.TELCO_CHURN_CSV.name}\n")
        f.write(f"Total Customers: {len(df):,}\n")
        f.write(f"Features: {df.shape[1]}\n\n")
        
        churn_rate = (df[config.TARGET_COLUMN] == 'Yes').mean()
        f.write(f"Overall Churn Rate: {churn_rate:.1%}\n\n")
        
        f.write("KEY INSIGHTS:\n")
        f.write("-" * 60 + "\n")
        
        # Contract insight
        contract_rates = df.groupby('Contract')[config.TARGET_COLUMN].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        f.write(f"\n1. Contract Type (Highest Risk First):\n")
        for contract, rate in contract_rates.items():
            f.write(f"   - {contract}: {rate:.1f}% churn\n")
        
        # Charges insight
        median_churn = df[df[config.TARGET_COLUMN] == 'Yes']['MonthlyCharges'].median()
        median_no_churn = df[df[config.TARGET_COLUMN] == 'No']['MonthlyCharges'].median()
        f.write(f"\n2. Monthly Charges:\n")
        f.write(f"   - Churned customers median: ${median_churn:.2f}\n")
        f.write(f"   - Retained customers median: ${median_no_churn:.2f}\n")
        f.write(f"   - Difference: ${median_churn - median_no_churn:.2f}\n")
        
        # Tenure insight
        median_tenure_churn = df[df[config.TARGET_COLUMN] == 'Yes']['tenure'].median()
        median_tenure_no = df[df[config.TARGET_COLUMN] == 'No']['tenure'].median()
        f.write(f"\n3. Customer Tenure:\n")
        f.write(f"   - Churned customers median: {median_tenure_churn:.0f} months\n")
        f.write(f"   - Retained customers median: {median_tenure_no:.0f} months\n")
        
        new_customer_churn = (df[df['tenure'] <= 12][config.TARGET_COLUMN] == 'Yes').mean()
        f.write(f"   - New customers (<=12mo) churn rate: {new_customer_churn:.1%}\n")
    
    print(f"\n[OK] Summary report saved to: {summary_path.name}")


def main():
    """Run complete EDA pipeline."""
    print("="*60)
    print("CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_and_validate_data()
    
    # Clean data
    print("\nCleaning data...")
    df = clean_data(df)
    
    # Create output directory
    output_dir = config.REPORTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Run analyses
    analyze_target_distribution(df, output_dir)
    analyze_churn_vs_monthly_charges(df, output_dir)
    analyze_churn_vs_contract(df, output_dir)
    analyze_churn_vs_tenure(df, output_dir)
    analyze_combined_risk_factors(df, output_dir)
    generate_summary_report(df, output_dir)
    
    print(f"\n{'='*60}")
    print("EDA COMPLETE")
    print(f"{'='*60}")
    print(f"\nAll figures saved to: {output_dir}")
    print("Ready for modeling phase.")


if __name__ == "__main__":
    main()
