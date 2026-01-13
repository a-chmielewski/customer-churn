"""
Unit tests for churn prediction pipeline.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.churn import config
from src.churn.data_load import load_and_validate_data, validate_schema, DataValidationError
from src.churn.preprocess import engineer_features, create_preprocessor, split_data
from src.churn.eda import clean_data


class TestDataValidation(unittest.TestCase):
    """Test data loading and validation."""
    
    def test_load_data_returns_dataframe(self):
        """Test that data loading returns a DataFrame."""
        df = load_and_validate_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
    
    def test_schema_validation_required_columns(self):
        """Test schema validation catches missing columns."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        with self.assertRaises(DataValidationError):
            validate_schema(df)
    
    def test_target_values_are_valid(self):
        """Test target column contains only Yes/No."""
        df = load_and_validate_data()
        unique_values = set(df[config.TARGET_COLUMN].unique())
        self.assertEqual(unique_values, {'Yes', 'No'})
    
    def test_no_duplicate_customer_ids(self):
        """Test no duplicate customer IDs in data."""
        df = load_and_validate_data()
        duplicates = df[config.ID_COLUMN].duplicated().sum()
        self.assertEqual(duplicates, 0)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing pipeline."""
    
    def setUp(self):
        """Load data for tests."""
        self.df = load_and_validate_data()
        self.df = clean_data(self.df)
    
    def test_feature_engineering_adds_columns(self):
        """Test feature engineering creates new features."""
        original_cols = set(self.df.columns)
        df_engineered = engineer_features(self.df)
        new_cols = set(df_engineered.columns) - original_cols
        
        # Should add at least 5 new features
        self.assertGreaterEqual(len(new_cols), 5)
        self.assertIn('is_new_customer', df_engineered.columns)
        self.assertIn('is_month_to_month', df_engineered.columns)
    
    def test_data_split_preserves_size(self):
        """Test data splitting produces correct sizes."""
        df_engineered = engineer_features(self.df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_engineered)
        
        total_samples = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total_samples, len(self.df))
        
        # Check proportions (roughly 70/15/15)
        self.assertAlmostEqual(len(X_train) / len(self.df), 0.70, delta=0.01)
        self.assertAlmostEqual(len(X_val) / len(self.df), 0.15, delta=0.01)
        self.assertAlmostEqual(len(X_test) / len(self.df), 0.15, delta=0.01)
    
    def test_preprocessor_output_shape(self):
        """Test preprocessor produces expected output shape."""
        df_engineered = engineer_features(self.df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_engineered)
        
        preprocessor = create_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Should have 38 features after preprocessing
        self.assertEqual(X_train_processed.shape[1], 38)
        self.assertEqual(X_train_processed.shape[0], len(X_train))
    
    def test_preprocessor_handles_new_data(self):
        """Test fitted preprocessor can transform new data."""
        df_engineered = engineer_features(self.df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_engineered)
        
        preprocessor = create_preprocessor()
        preprocessor.fit(X_train, y_train)
        
        # Should transform validation data without errors
        X_val_processed = preprocessor.transform(X_val)
        self.assertEqual(X_val_processed.shape[0], len(X_val))
        self.assertEqual(X_val_processed.shape[1], 38)


class TestEndToEnd(unittest.TestCase):
    """Test end-to-end pipeline on small sample."""
    
    def test_full_pipeline_small_sample(self):
        """Test entire pipeline runs on small data sample."""
        # Load and sample data
        df = load_and_validate_data()
        df_sample = df.sample(n=100, random_state=42)
        df_sample = clean_data(df_sample)
        df_sample = engineer_features(df_sample)
        
        # Split
        X = df_sample.drop(columns=[config.TARGET_COLUMN, config.ID_COLUMN])
        y = (df_sample[config.TARGET_COLUMN] == 'Yes').astype(int)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Preprocess
        preprocessor = create_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train simple model
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_processed, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Verify predictions
        self.assertEqual(len(y_pred_proba), len(y_test))
        self.assertTrue(all(0 <= p <= 1 for p in y_pred_proba))


if __name__ == '__main__':
    unittest.main()
