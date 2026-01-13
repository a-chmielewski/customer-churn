"""
Preprocessing pipeline for customer churn prediction.
Leakage-safe: all transformations fit only on training data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import sys

# Handle imports
try:
    from . import config
    from .data_load import load_and_validate_data
    from .eda import clean_data
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.churn import config
    from src.churn.data_load import load_and_validate_data
    from src.churn.eda import clean_data


# Feature groups based on EDA
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features based on EDA insights.
    
    Args:
        df: DataFrame with raw features
        
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()
    
    # Early customer flag (critical first year - from EDA)
    df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
    
    # Tenure bins to capture non-linear effects
    df['tenure_group'] = pd.cut(df['tenure'], 
                                bins=[0, 12, 24, 48, 72],
                                labels=['0-12mo', '13-24mo', '25-48mo', '49-72mo'])
    
    # Contract risk (month-to-month is high risk - from EDA)
    df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    
    # Price relative to tenure (loyalty indicator)
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero
    
    # High charges flag (median-based from EDA: $79.65 for churned)
    df['has_high_charges'] = (df['MonthlyCharges'] > 70).astype(int)
    
    # Count of services (internet + phone related)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['num_services'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    return df


def split_data(df: pd.DataFrame, 
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = config.RANDOM_SEED):
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Full dataset
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data)
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Separate features and target
    X = df.drop(columns=[config.TARGET_COLUMN, config.ID_COLUMN])
    y = (df[config.TARGET_COLUMN] == 'Yes').astype(int)  # Binary: 1=Churn, 0=No Churn
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust proportion
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"Data split complete:")
    print(f"  Train: {len(X_train):,} ({len(X_train)/len(df):.1%})")
    print(f"  Val:   {len(X_val):,} ({len(X_val)/len(df):.1%})")
    print(f"  Test:  {len(X_test):,} ({len(X_test)/len(df):.1%})")
    print(f"\nChurn rate:")
    print(f"  Train: {y_train.mean():.1%}")
    print(f"  Val:   {y_val.mean():.1%}")
    print(f"  Test:  {y_test.mean():.1%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessor():
    """
    Create preprocessing pipeline with ColumnTransformer.
    
    Numeric features: impute (median) + scale (StandardScaler)
    Categorical features: impute (most_frequent) + one-hot encode
    
    Returns:
        ColumnTransformer pipeline
    """
    # Update feature lists with engineered features
    numeric_features = NUMERIC_FEATURES + [
        'is_new_customer', 'is_month_to_month', 'charges_per_tenure',
        'has_high_charges', 'num_services'
    ]
    
    categorical_features = CATEGORICAL_FEATURES + ['tenure_group']
    
    # Numeric pipeline: impute median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute most_frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop ID and other unused columns
    )
    
    return preprocessor


def get_feature_names(preprocessor, X_sample):
    """
    Extract feature names after preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        X_sample: Sample dataframe to determine feature names
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    # Get numeric feature names
    numeric_features = preprocessor.transformers_[0][2]
    feature_names.extend(numeric_features)
    
    # Get categorical feature names (one-hot encoded)
    categorical_transformer = preprocessor.transformers_[1][1]
    categorical_features = preprocessor.transformers_[1][2]
    
    if hasattr(categorical_transformer.named_steps['onehot'], 'get_feature_names_out'):
        cat_names = categorical_transformer.named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_names)
    
    return feature_names


def prepare_data_for_modeling(save_artifacts: bool = True):
    """
    Complete preprocessing pipeline: load, engineer, split, transform.
    
    Args:
        save_artifacts: Whether to save preprocessor and splits
        
    Returns:
        Dictionary with train/val/test data and fitted preprocessor
    """
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load and clean data
    print("\n[1/5] Loading data...")
    df = load_and_validate_data()
    df = clean_data(df)
    
    # 2. Engineer features
    print("\n[2/5] Engineering features...")
    df = engineer_features(df)
    print(f"  Added engineered features: is_new_customer, tenure_group, etc.")
    
    # 3. Split data (BEFORE fitting preprocessor - critical for no leakage!)
    print("\n[3/5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # 4. Create and fit preprocessor (FIT ON TRAIN ONLY!)
    print("\n[4/5] Creating preprocessing pipeline...")
    preprocessor = create_preprocessor()
    
    print("  Fitting on training data only (no leakage)...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform val and test (using train statistics)
    print("  Transforming validation and test data...")
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"  Final feature count: {X_train_processed.shape[1]}")
    
    # 5. Save artifacts
    if save_artifacts:
        print("\n[5/5] Saving artifacts...")
        
        # Create models directory
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor
        preprocessor_path = config.MODELS_DIR / 'preprocessor.pkl'
        joblib.dump(preprocessor, preprocessor_path)
        print(f"  Saved preprocessor: {preprocessor_path.name}")
        
        # Save processed data
        np.save(config.PROCESSED_DATA_DIR / 'X_train.npy', X_train_processed)
        np.save(config.PROCESSED_DATA_DIR / 'X_val.npy', X_val_processed)
        np.save(config.PROCESSED_DATA_DIR / 'X_test.npy', X_test_processed)
        np.save(config.PROCESSED_DATA_DIR / 'y_train.npy', y_train.values)
        np.save(config.PROCESSED_DATA_DIR / 'y_val.npy', y_val.values)
        np.save(config.PROCESSED_DATA_DIR / 'y_test.npy', y_test.values)
        print(f"  Saved processed data to: {config.PROCESSED_DATA_DIR.name}/")
        
        # Save feature names
        feature_names = get_feature_names(preprocessor, X_train)
        joblib.dump(feature_names, config.MODELS_DIR / 'feature_names.pkl')
        print(f"  Saved feature names: feature_names.pkl")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    return {
        'X_train': X_train_processed,
        'X_val': X_val_processed,
        'X_test': X_test_processed,
        'y_train': y_train.values,
        'y_val': y_val.values,
        'y_test': y_test.values,
        'preprocessor': preprocessor,
        'feature_names': get_feature_names(preprocessor, X_train)
    }


def load_processed_data():
    """
    Load preprocessed data from disk.
    
    Returns:
        Dictionary with processed data
    """
    return {
        'X_train': np.load(config.PROCESSED_DATA_DIR / 'X_train.npy'),
        'X_val': np.load(config.PROCESSED_DATA_DIR / 'X_val.npy'),
        'X_test': np.load(config.PROCESSED_DATA_DIR / 'X_test.npy'),
        'y_train': np.load(config.PROCESSED_DATA_DIR / 'y_train.npy'),
        'y_val': np.load(config.PROCESSED_DATA_DIR / 'y_val.npy'),
        'y_test': np.load(config.PROCESSED_DATA_DIR / 'y_test.npy'),
    }


def load_preprocessor():
    """Load fitted preprocessor from disk."""
    return joblib.load(config.MODELS_DIR / 'preprocessor.pkl')


def load_feature_names():
    """Load feature names from disk."""
    return joblib.load(config.MODELS_DIR / 'feature_names.pkl')


if __name__ == "__main__":
    # Run preprocessing pipeline
    results = prepare_data_for_modeling(save_artifacts=True)
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"Training set shape: {results['X_train'].shape}")
    print(f"Feature names count: {len(results['feature_names'])}")
    print(f"\nSample feature names (first 10):")
    for i, name in enumerate(results['feature_names'][:10], 1):
        print(f"  {i}. {name}")
    
    print("\n[OK] Ready for model training")
