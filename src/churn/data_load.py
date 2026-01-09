"""
Data loading and validation module for customer churn prediction.
"""
import pandas as pd
from pathlib import Path
from typing import List
import logging
import sys

# Handle imports for both direct execution and module import
try:
    from . import config
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.churn import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def load_raw_data(file_path: Path = config.TELCO_CHURN_CSV) -> pd.DataFrame:
    """
    Load raw customer churn data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the raw data
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def validate_required_columns(df: pd.DataFrame, required_columns: List[str] = None) -> None:
    """
    Validate that all required columns exist in the DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        DataValidationError: If any required columns are missing
    """
    if required_columns is None:
        required_columns = config.REQUIRED_COLUMNS
    
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    
    logger.info("All required columns are present")


def validate_target_values(df: pd.DataFrame, 
                          target_column: str = config.TARGET_COLUMN,
                          valid_values: set = config.VALID_TARGET_VALUES) -> None:
    """
    Validate that target column contains only expected values.
    
    Args:
        df: DataFrame to validate
        target_column: Name of the target column
        valid_values: Set of valid values for the target column
        
    Raises:
        DataValidationError: If target column contains invalid values
    """
    if target_column not in df.columns:
        raise DataValidationError(f"Target column '{target_column}' not found in DataFrame")
    
    unique_values = set(df[target_column].dropna().unique())
    invalid_values = unique_values - valid_values
    
    if invalid_values:
        raise DataValidationError(
            f"Invalid values in target column '{target_column}': {sorted(invalid_values)}. "
            f"Expected values: {sorted(valid_values)}"
        )
    
    logger.info(f"Target column '{target_column}' contains valid values: {sorted(unique_values)}")


def validate_no_duplicate_ids(df: pd.DataFrame, 
                              id_column: str = config.ID_COLUMN) -> None:
    """
    Validate that there are no duplicate customer IDs.
    
    Args:
        df: DataFrame to validate
        id_column: Name of the ID column
        
    Raises:
        DataValidationError: If duplicate IDs are found
    """
    if id_column not in df.columns:
        raise DataValidationError(f"ID column '{id_column}' not found in DataFrame")
    
    duplicate_ids = df[id_column].duplicated()
    num_duplicates = duplicate_ids.sum()
    
    if num_duplicates > 0:
        duplicate_values = df[id_column][duplicate_ids].unique()
        raise DataValidationError(
            f"Found {num_duplicates} duplicate entries for {len(duplicate_values)} unique IDs. "
            f"First few duplicates: {list(duplicate_values[:5])}"
        )
    
    logger.info(f"No duplicate IDs found in '{id_column}' column")


def validate_schema(df: pd.DataFrame) -> None:
    """
    Perform all schema validation checks on the DataFrame.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        DataValidationError: If any validation check fails
    """
    logger.info("Starting schema validation")
    
    # Check required columns
    validate_required_columns(df)
    
    # Check target values
    validate_target_values(df)
    
    # Check for duplicate IDs
    validate_no_duplicate_ids(df)
    
    logger.info("Schema validation completed successfully")


def load_and_validate_data(file_path: Path = config.TELCO_CHURN_CSV) -> pd.DataFrame:
    """
    Load data from CSV and perform schema validation.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Validated DataFrame
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
        DataValidationError: If any validation check fails
    """
    # Load data
    df = load_raw_data(file_path)
    
    # Validate schema
    validate_schema(df)
    
    return df


if __name__ == "__main__":
    # Test the data loading and validation
    try:
        df = load_and_validate_data()
        print(f"\nData successfully loaded and validated!")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nTarget distribution:")
        print(df[config.TARGET_COLUMN].value_counts())
    except (FileNotFoundError, DataValidationError) as e:
        print(f"Error: {e}")
