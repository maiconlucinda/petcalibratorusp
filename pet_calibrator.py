#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PET Thermal Comfort Calibrator

Sistema de calibração de conforto térmico baseado no índice PET (Physiological 
Equivalent Temperature) usando modelagem ordinal estatística.

Autora: Carol Freire do Santos
Instituição: Universidade de São Paulo (USP)
Programa: Doutorado em Climatologia

Este sistema processa dados de questionários com valores de PET pré-calculados,
aplica modelagem de regressão logística ordinal proporcional, e gera relatórios
com visualizações e métricas de conforto térmico calibradas localmente.

Metodologia:
- Regressão logística ordinal proporcional (proportional odds model)
- Cálculo de PET neutro (τ_0 / β)
- Determinação de faixas de conforto (80% e 90%)
- Análise opcional de aceitabilidade térmica

Referências:
- Höppe, P. (1999). The physiological equivalent temperature.
- McCullagh, P. (1980). Regression models for ordinal data.
- Agresti, A. (2010). Analysis of Ordinal Categorical Data.
"""

import sys
import os
import logging
import argparse
import traceback
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

# Sensation mapping (normalized - lowercase, no accents)
SENSATION_MAPPING = {
    'muito frio': -3,
    'frio': -2,
    'frio moderado': -1,
    'confortavel': 0,  # normalized without accent
    'calor moderado': 1,
    'quente': 2,
    'muito quente': 3
}

# Reverse mapping for display (with proper Portuguese)
SENSATION_LABELS = {
    -3: 'Muito Frio',
    -2: 'Frio',
    -1: 'Frio Moderado',
    0: 'Confortável',
    1: 'Calor Moderado',
    2: 'Quente',
    3: 'Muito Quente'
}

# Validation thresholds
MIN_SAMPLE_SIZE = 30
PET_MIN_PLAUSIBLE = -20.0
PET_MAX_PLAUSIBLE = 60.0

# Comfort calculation settings
PET_GRID_MIN = -5.0
PET_GRID_MAX = 55.0
PET_GRID_STEP = 0.05
COMFORT_THRESHOLDS = [0.8, 0.9]

# Visualization settings
PLOT_DPI = 300
PLOT_FIGSIZE = (10, 6)
COLOR_PALETTE = {
    -3: '#08519c',  # Dark blue
    -2: '#3182bd',  # Blue
    -1: '#6baed6',  # Light blue
    0: '#31a354',   # Green
    1: '#fd8d3c',   # Orange
    2: '#e6550d',   # Dark orange
    3: '#a63603'    # Red
}


# ============================================================================
# CUSTOM EXCEPTION CLASSES
# ============================================================================

class PETCalibratorError(Exception):
    """Base exception for PET calibrator."""
    pass


class DataLoadError(PETCalibratorError):
    """Error loading input data."""
    pass


class ValidationError(PETCalibratorError):
    """Data validation failed."""
    pass


class ModelConvergenceError(PETCalibratorError):
    """Ordinal model failed to converge."""
    pass


class InsufficientDataError(PETCalibratorError):
    """Not enough valid observations."""
    pass


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Configure logging with file and console handlers.
    
    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO
        log_file: Optional path to log file. If None, uses 'pet_calibrator.log'
    
    Side effects:
        Configures the root logger with appropriate handlers and formatters
    """
    # Determine log level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = 'pet_calibrator.log'
    
    try:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        logger.warning(f"Could not create log file {log_file}: {e}")


# ============================================================================
# DATA LOADING AND COLUMN MAPPING
# ============================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV or Excel file.
    
    Detects file format by extension and uses appropriate pandas loader.
    Supports .csv, .xlsx, and .xls formats.
    
    Args:
        file_path: Path to input file (.csv, .xlsx, .xls)
    
    Returns:
        DataFrame with raw data
    
    Raises:
        DataLoadError: If file doesn't exist or format is unsupported
        
    Requirements: 1.1, 11.1
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise DataLoadError(
            f"Input file not found: {file_path}\n"
            f"Please check the file path and try again."
        )
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    logging.info(f"Loading data from {file_path}")
    
    try:
        if ext == '.csv':
            # Load CSV file
            df = pd.read_csv(file_path, encoding='utf-8')
            logging.info(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
        elif ext in ['.xlsx', '.xls']:
            # Load Excel file
            df = pd.read_excel(file_path, engine='openpyxl' if ext == '.xlsx' else None)
            logging.info(f"Successfully loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
        else:
            raise DataLoadError(
                f"Unsupported file format: {ext}\n"
                f"Supported formats: .csv, .xlsx, .xls"
            )
        
        # Check if DataFrame is empty
        if df.empty:
            raise DataLoadError(f"File {file_path} is empty (no data rows)")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise DataLoadError(f"File {file_path} is empty or has no valid data")
    except pd.errors.ParserError as e:
        raise DataLoadError(f"Error parsing file {file_path}: {e}")
    except Exception as e:
        raise DataLoadError(f"Error loading file {file_path}: {e}")


def load_column_mapping(mapping_file: str) -> Dict[str, str]:
    """
    Load and parse JSON mapping file for column names.
    
    The JSON file should map expected column names to actual column names
    in the input data file.
    
    Expected format:
    {
        "PET_C": "actual_pet_column_name",
        "Sensation": "actual_sensation_column_name",
        "Acceptability": "actual_acceptability_column_name"  // optional
    }
    
    Args:
        mapping_file: Path to JSON mapping file
    
    Returns:
        Dictionary mapping expected names to actual column names
    
    Raises:
        DataLoadError: If file doesn't exist or JSON is invalid
        
    Requirements: 1.2, 11.2
    """
    import json
    
    # Check if file exists
    if not os.path.exists(mapping_file):
        raise DataLoadError(
            f"Column mapping file not found: {mapping_file}\n"
            f"Please check the file path and try again."
        )
    
    logging.info(f"Loading column mapping from {mapping_file}")
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Validate that mapping is a dictionary
        if not isinstance(mapping, dict):
            raise DataLoadError(
                f"Invalid JSON structure in {mapping_file}\n"
                f"Expected a dictionary/object, got {type(mapping).__name__}"
            )
        
        # Validate that all values are strings
        for key, value in mapping.items():
            if not isinstance(value, str):
                raise DataLoadError(
                    f"Invalid mapping value for '{key}' in {mapping_file}\n"
                    f"Expected string, got {type(value).__name__}"
                )
        
        logging.info(f"Successfully loaded column mapping with {len(mapping)} entries")
        logging.debug(f"Column mapping: {mapping}")
        
        return mapping
        
    except json.JSONDecodeError as e:
        raise DataLoadError(
            f"Invalid JSON in {mapping_file}:\n"
            f"Line {e.lineno}, Column {e.colno}: {e.msg}"
        )
    except Exception as e:
        raise DataLoadError(f"Error loading mapping file {mapping_file}: {e}")


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename DataFrame columns based on mapping dictionary.
    
    The mapping dictionary should map expected column names (keys) to actual
    column names in the DataFrame (values). Only columns present in both the
    mapping and the DataFrame will be renamed.
    
    Args:
        df: Input DataFrame
        mapping: Dictionary mapping expected names to actual column names
                 Format: {"expected_name": "actual_name"}
    
    Returns:
        DataFrame with renamed columns
        
    Requirements: 1.2
    """
    if not mapping:
        logging.info("No column mapping provided, using original column names")
        return df.copy()
    
    # Create reverse mapping (actual -> expected)
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    # Find columns that exist in the DataFrame and need to be renamed
    columns_to_rename = {}
    for actual_name, expected_name in reverse_mapping.items():
        if actual_name in df.columns:
            columns_to_rename[actual_name] = expected_name
    
    if columns_to_rename:
        df_mapped = df.rename(columns=columns_to_rename)
        logging.info(f"Applied column mapping: renamed {len(columns_to_rename)} columns")
        logging.debug(f"Renamed columns: {columns_to_rename}")
    else:
        df_mapped = df.copy()
        logging.warning("No columns matched the mapping - using original column names")
    
    return df_mapped


def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check for presence of required columns (PET_C and Sensation).
    
    These are the only two columns required for the calibration analysis.
    All other columns are optional.
    
    Args:
        df: Input DataFrame to validate
    
    Returns:
        Tuple of (is_valid, missing_columns)
        - is_valid: True if all required columns are present
        - missing_columns: List of missing required column names
        
    Requirements: 1.3, 1.4, 1.5, 11.3
    """
    required_columns = ['PET_C', 'Sensation']
    missing_columns = []
    
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    is_valid = len(missing_columns) == 0
    
    if is_valid:
        logging.info("All required columns are present (PET_C, Sensation)")
    else:
        logging.error(f"Missing required columns: {missing_columns}")
    
    return is_valid, missing_columns


# ============================================================================
# DATA CLEANING AND VALIDATION
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for consistent sensation mapping.
    
    Normalization steps:
    1. Convert to lowercase
    2. Remove extra whitespace (leading, trailing, and multiple spaces)
    3. Remove accents using unicodedata
    
    Args:
        text: Raw text string to normalize
    
    Returns:
        Normalized text string
        
    Examples:
        >>> normalize_text("  Confortável  ")
        'confortavel'
        >>> normalize_text("MUITO  FRIO")
        'muito frio'
        
    Requirements: 1.7, 3.8
    """
    import unicodedata
    
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove accents using NFD normalization
    # NFD decomposes characters into base + combining marks
    # Then filter out combining marks (category 'Mn')
    text_nfd = unicodedata.normalize('NFD', text)
    text_no_accents = ''.join(
        char for char in text_nfd 
        if unicodedata.category(char) != 'Mn'
    )
    
    return text_no_accents


def map_sensation_to_ordinal(df: pd.DataFrame, 
                             sensation_col: str = 'Sensation') -> pd.DataFrame:
    """
    Map sensation categories to ordinal scale (-3 to +3).
    
    Creates a new column 'TSV_ordinal' with ordinal values based on the
    SENSATION_MAPPING dictionary. Applies text normalization before mapping.
    
    Mapping:
        - "muito frio" -> -3
        - "frio" -> -2
        - "frio moderado" -> -1
        - "confortável" -> 0
        - "calor moderado" -> +1
        - "quente" -> +2
        - "muito quente" -> +3
    
    Args:
        df: Input DataFrame with sensation column
        sensation_col: Name of the sensation column (default: 'Sensation')
    
    Returns:
        DataFrame with new 'TSV_ordinal' column and 'valid_sensation' flag
    
    Side effects:
        - Logs warnings for unmapped values
        - Adds 'valid_sensation' boolean column (True if mapped successfully)
        
    Requirements: 1.8, 3.1-3.9
    """
    df = df.copy()
    
    # Check if sensation column exists
    if sensation_col not in df.columns:
        raise ValidationError(
            f"Sensation column '{sensation_col}' not found in DataFrame.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    logging.info(f"Mapping sensation values to ordinal scale")
    
    # Normalize sensation text
    df['sensation_normalized'] = df[sensation_col].apply(normalize_text)
    
    # Map to ordinal values
    df['TSV_ordinal'] = df['sensation_normalized'].map(SENSATION_MAPPING)
    
    # Flag valid sensations
    df['valid_sensation'] = df['TSV_ordinal'].notna()
    
    # Count and log unmapped values
    n_unmapped = (~df['valid_sensation']).sum()
    if n_unmapped > 0:
        unmapped_values = df.loc[~df['valid_sensation'], sensation_col].unique()
        logging.warning(
            f"Found {n_unmapped} rows with unmapped sensation values.\n"
            f"Unmapped values: {list(unmapped_values)}\n"
            f"Valid sensations: {list(SENSATION_MAPPING.keys())}"
        )
    
    # Log mapping statistics
    n_mapped = df['valid_sensation'].sum()
    logging.info(f"Successfully mapped {n_mapped} rows to ordinal scale")
    
    # Log distribution of sensations
    if n_mapped > 0:
        sensation_counts = df.loc[df['valid_sensation'], 'TSV_ordinal'].value_counts().sort_index()
        logging.info("Sensation distribution:")
        for ordinal_val, count in sensation_counts.items():
            ordinal_int = int(ordinal_val)
            label = SENSATION_LABELS.get(ordinal_int, f"Unknown ({ordinal_val})")
            logging.info(f"  {label} ({ordinal_int:+d}): {count} responses")
    
    # Drop temporary normalized column
    df = df.drop(columns=['sensation_normalized'])
    
    return df


def validate_pet_values(df: pd.DataFrame, 
                       pet_col: str = 'PET_C') -> pd.DataFrame:
    """
    Validate PET values and flag invalid entries.
    
    Validation rules:
    1. Must be numeric (not NaN)
    2. Plausible range: -20°C to 60°C (warning if outside, but kept)
    
    Args:
        df: DataFrame with PET column
        pet_col: Name of the PET column (default: 'PET_C')
    
    Returns:
        DataFrame with 'valid_pet' boolean column
    
    Side effects:
        - Logs warnings for NaN values
        - Logs warnings for out-of-range values
        
    Requirements: 1.9, 2.1-2.3, 11.7
    """
    df = df.copy()
    
    # Check if PET column exists
    if pet_col not in df.columns:
        raise ValidationError(
            f"PET column '{pet_col}' not found in DataFrame.\n"
            f"Available columns: {list(df.columns)}"
        )
    
    logging.info(f"Validating PET values in column '{pet_col}'")
    
    # Check for numeric type
    try:
        df[pet_col] = pd.to_numeric(df[pet_col], errors='coerce')
    except Exception as e:
        logging.warning(f"Error converting PET column to numeric: {e}")
    
    # Identify NaN values
    n_nan = df[pet_col].isna().sum()
    if n_nan > 0:
        logging.warning(f"Found {n_nan} rows with missing PET values (NaN or empty)")
    
    # Flag valid PET values (not NaN)
    df['valid_pet'] = df[pet_col].notna()
    
    # Check plausible range for non-NaN values
    valid_pet_mask = df['valid_pet']
    if valid_pet_mask.any():
        pet_values = df.loc[valid_pet_mask, pet_col]
        
        # Check minimum
        below_min = pet_values < PET_MIN_PLAUSIBLE
        n_below_min = below_min.sum()
        if n_below_min > 0:
            min_val = pet_values[below_min].min()
            logging.warning(
                f"Found {n_below_min} PET values below plausible minimum "
                f"({PET_MIN_PLAUSIBLE}°C). Minimum value: {min_val:.1f}°C. "
                f"These values will be kept for analysis but may indicate data issues."
            )
        
        # Check maximum
        above_max = pet_values > PET_MAX_PLAUSIBLE
        n_above_max = above_max.sum()
        if n_above_max > 0:
            max_val = pet_values[above_max].max()
            logging.warning(
                f"Found {n_above_max} PET values above plausible maximum "
                f"({PET_MAX_PLAUSIBLE}°C). Maximum value: {max_val:.1f}°C. "
                f"These values will be kept for analysis but may indicate data issues."
            )
        
        # Log PET statistics
        logging.info(f"PET statistics for valid values:")
        logging.info(f"  Mean: {pet_values.mean():.2f}°C")
        logging.info(f"  Median: {pet_values.median():.2f}°C")
        logging.info(f"  Std Dev: {pet_values.std():.2f}°C")
        logging.info(f"  Range: [{pet_values.min():.2f}, {pet_values.max():.2f}]°C")
    
    return df


def remove_invalid_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Remove rows with invalid data and return statistics.
    
    Removes rows where:
    - PET_C is NaN or invalid (valid_pet == False)
    - Sensation is unmapped or invalid (valid_sensation == False)
    
    Args:
        df: DataFrame with 'valid_pet' and 'valid_sensation' columns
    
    Returns:
        Tuple of (cleaned_df, removal_stats)
        
    removal_stats contains:
        - n_total: Total rows in input
        - n_missing_pet: Count of rows with missing/invalid PET
        - n_invalid_sensation: Count of rows with invalid sensation
        - n_both_invalid: Count of rows with both invalid
        - n_total_removed: Total rows removed
        - n_retained: Rows retained after cleaning
        - pct_retained: Percentage of data retained
        
    Requirements: 1.9, 2.4, 2.5
    """
    # Check for required validation columns
    if 'valid_pet' not in df.columns:
        raise ValidationError(
            "DataFrame missing 'valid_pet' column. "
            "Run validate_pet_values() first."
        )
    if 'valid_sensation' not in df.columns:
        raise ValidationError(
            "DataFrame missing 'valid_sensation' column. "
            "Run map_sensation_to_ordinal() first."
        )
    
    logging.info("Removing invalid rows")
    
    # Calculate statistics
    n_total = len(df)
    n_missing_pet = (~df['valid_pet']).sum()
    n_invalid_sensation = (~df['valid_sensation']).sum()
    
    # Count rows with both invalid
    n_both_invalid = ((~df['valid_pet']) & (~df['valid_sensation'])).sum()
    
    # Keep only rows with both valid PET and valid sensation
    df_clean = df[df['valid_pet'] & df['valid_sensation']].copy()
    
    # Calculate removal statistics
    n_retained = len(df_clean)
    n_total_removed = n_total - n_retained
    pct_retained = (n_retained / n_total * 100) if n_total > 0 else 0
    
    removal_stats = {
        'n_total': n_total,
        'n_missing_pet': n_missing_pet,
        'n_invalid_sensation': n_invalid_sensation,
        'n_both_invalid': n_both_invalid,
        'n_total_removed': n_total_removed,
        'n_retained': n_retained,
        'pct_retained': pct_retained
    }
    
    # Log removal statistics
    logging.info(f"Data cleaning summary:")
    logging.info(f"  Total rows: {n_total}")
    logging.info(f"  Rows with invalid PET: {n_missing_pet}")
    logging.info(f"  Rows with invalid sensation: {n_invalid_sensation}")
    logging.info(f"  Rows with both invalid: {n_both_invalid}")
    logging.info(f"  Total rows removed: {n_total_removed}")
    logging.info(f"  Rows retained: {n_retained} ({pct_retained:.1f}%)")
    
    # Drop validation flag columns (no longer needed)
    df_clean = df_clean.drop(columns=['valid_pet', 'valid_sensation'])
    
    # Check if we have enough data
    if n_retained < MIN_SAMPLE_SIZE:
        logging.warning(
            f"Only {n_retained} valid observations remaining. "
            f"Minimum recommended: {MIN_SAMPLE_SIZE}. "
            f"Model results may be unstable."
        )
    
    return df_clean, removal_stats


def save_cleaned_data(df: pd.DataFrame, 
                     output_path: str,
                     filename: str = 'respostas_com_PET.csv') -> str:
    """
    Save cleaned DataFrame to CSV with UTF-8 encoding.
    
    The output file includes the TSV_ordinal column along with all original
    columns from the cleaned dataset.
    
    Args:
        df: Cleaned DataFrame with TSV_ordinal column
        output_path: Directory path for output file
        filename: Name of output file (default: 'respostas_com_PET.csv')
    
    Returns:
        Full path to saved file
        
    Raises:
        IOError: If file cannot be written
        
    Requirements: 2.6
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(output_path, filename)
    
    logging.info(f"Saving cleaned data to {file_path}")
    
    try:
        # Save to CSV with UTF-8 encoding
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        logging.info(
            f"Successfully saved {len(df)} rows and {len(df.columns)} columns "
            f"to {file_path}"
        )
        
        return file_path
        
    except Exception as e:
        raise IOError(f"Error saving cleaned data to {file_path}: {e}")


# ============================================================================
# ORDINAL REGRESSION MODEL
# ============================================================================

def fit_ordered_model(df: pd.DataFrame) -> 'OrderedModel':
    """
    Fit proportional odds ordinal logistic regression model.
    
    Model specification:
        TSV_ordinal ~ PET_C
        Link function: logit
        
    The proportional odds model assumes:
        logit(P(Y ≤ k | PET)) = τ_k - β * PET
        
    Where:
        - Y is the ordinal sensation (-3 to +3)
        - τ_k are cutpoints (thresholds) for k ∈ {-3, -2, -1, 0, 1, 2}
        - β is the coefficient for PET_C
        - The negative sign ensures higher PET → higher sensation
    
    Args:
        df: Cleaned DataFrame with 'TSV_ordinal' and 'PET_C' columns
    
    Returns:
        Fitted OrderedModel results object from statsmodels
    
    Raises:
        InsufficientDataError: If fewer than 30 valid observations
        ModelConvergenceError: If model fails to converge
        ValidationError: If required columns are missing
        
    Requirements: 4.1-4.6, 11.4, 11.5
    """
    # Validate required columns
    required_cols = ['TSV_ordinal', 'PET_C']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Missing required columns for model fitting: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Check for minimum sample size
    n_obs = len(df)
    if n_obs < MIN_SAMPLE_SIZE:
        raise InsufficientDataError(
            f"Insufficient data for model fitting: {n_obs} observations.\n"
            f"Minimum required: {MIN_SAMPLE_SIZE} observations.\n"
            f"Please collect more data or check data cleaning steps."
        )
    
    logging.info(f"Fitting ordinal regression model with {n_obs} observations")
    
    # Check for variation in both variables
    if df['TSV_ordinal'].nunique() < 3:
        raise ValidationError(
            f"Insufficient variation in TSV_ordinal: only {df['TSV_ordinal'].nunique()} unique values.\n"
            f"Need at least 3 different sensation categories for ordinal regression."
        )
    
    if df['PET_C'].nunique() < 10:
        logging.warning(
            f"Limited variation in PET_C: only {df['PET_C'].nunique()} unique values.\n"
            f"Model may be unstable with limited predictor variation."
        )
    
    # Prepare data for modeling
    # Remove any remaining NaN values
    df_model = df[['TSV_ordinal', 'PET_C']].dropna()
    
    if len(df_model) < n_obs:
        n_dropped = n_obs - len(df_model)
        logging.warning(f"Dropped {n_dropped} rows with NaN values before modeling")
        n_obs = len(df_model)
        
        if n_obs < MIN_SAMPLE_SIZE:
            raise InsufficientDataError(
                f"After removing NaN values, only {n_obs} observations remain.\n"
                f"Minimum required: {MIN_SAMPLE_SIZE} observations."
            )
    
    try:
        # Fit OrderedModel using statsmodels
        # Formula: TSV_ordinal ~ PET_C
        # Link: logit (default for OrderedModel)
        model = OrderedModel(
            endog=df_model['TSV_ordinal'],
            exog=df_model[['PET_C']],
            distr='logit'
        )
        
        logging.info("Fitting model with logit link function...")
        
        # Fit the model
        result = model.fit(method='bfgs', disp=False)
        
        # Check convergence
        if not result.mle_retvals['converged']:
            raise ModelConvergenceError(
                "Ordinal model failed to converge.\n"
                "Possible causes:\n"
                "  - Insufficient sample size\n"
                "  - Limited variation in PET or sensation values\n"
                "  - Extreme outliers in the data\n"
                "  - Perfect or near-perfect separation\n"
                "Suggestions:\n"
                "  - Check data quality and distribution\n"
                "  - Ensure adequate sample size (recommended: 100+ observations)\n"
                "  - Remove extreme outliers if present"
            )
        
        logging.info("Model converged successfully")
        logging.info(f"Log-likelihood: {result.llf:.2f}")
        logging.info(f"AIC: {result.aic:.2f}")
        logging.info(f"BIC: {result.bic:.2f}")
        
        return result
        
    except ModelConvergenceError:
        # Re-raise our custom convergence error
        raise
    except Exception as e:
        # Catch any other errors during model fitting
        raise ModelConvergenceError(
            f"Error fitting ordinal model: {e}\n"
            f"This may indicate issues with data quality or model specification."
        )


def extract_model_parameters(model_results: 'OrderedModel') -> Dict:
    """
    Extract model parameters and standard errors from fitted model.
    
    Extracts:
        - Beta coefficient for PET_C and its standard error
        - Cutpoints (τ_k) for each threshold and their standard errors
        - Model fit statistics (log-likelihood, AIC, BIC)
        - Convergence status
        - Number of observations
    
    Args:
        model_results: Fitted OrderedModel results from statsmodels
    
    Returns:
        Dictionary containing:
            - 'beta': float - Coefficient for PET_C
            - 'beta_se': float - Standard error of beta
            - 'cutpoints': dict - {k: τ_k} for k in {-3, -2, -1, 0, 1, 2}
            - 'cutpoints_se': dict - {k: SE(τ_k)}
            - 'converged': bool - Convergence status
            - 'n_obs': int - Number of observations
            - 'log_likelihood': float - Model log-likelihood
            - 'aic': float - Akaike Information Criterion
            - 'bic': float - Bayesian Information Criterion
            
    Requirements: 4.3, 4.4
    """
    logging.info("Extracting model parameters")
    
    # Extract beta coefficient for PET_C
    # In OrderedModel, exogenous variables come first in params
    beta = model_results.params['PET_C']
    beta_se = model_results.bse['PET_C']
    
    logging.info(f"Beta coefficient (PET_C): {beta:.4f} (SE: {beta_se:.4f})")
    
    # Extract cutpoints (thresholds)
    # In OrderedModel, cutpoints are named like '0/-1', '1/2', etc.
    # They represent thresholds between adjacent categories
    cutpoints = {}
    cutpoints_se = {}
    
    # Get all parameter names that contain '/'
    cutpoint_names = [name for name in model_results.params.index if '/' in name]
    
    # Sort cutpoint names to ensure correct order
    cutpoint_names_sorted = sorted(cutpoint_names)
    
    logging.info("Cutpoints (thresholds):")
    for i, name in enumerate(cutpoint_names_sorted):
        # Extract the cutpoint value and SE
        tau_value = model_results.params[name]
        tau_se = model_results.bse[name]
        
        # Map to ordinal scale
        # The cutpoints separate categories, so we map them to the lower category
        # For 7 categories (-3 to +3), we have 6 cutpoints
        ordinal_k = -3 + i  # Maps to -3, -2, -1, 0, 1, 2
        
        cutpoints[ordinal_k] = tau_value
        cutpoints_se[ordinal_k] = tau_se
        
        logging.info(f"  τ_{ordinal_k:+d} = {tau_value:.4f} (SE: {tau_se:.4f})")
    
    # Extract model fit statistics
    log_likelihood = model_results.llf
    aic = model_results.aic
    bic = model_results.bic
    n_obs = int(model_results.nobs)
    converged = model_results.mle_retvals['converged']
    
    logging.info(f"Model fit statistics:")
    logging.info(f"  N observations: {n_obs}")
    logging.info(f"  Log-likelihood: {log_likelihood:.2f}")
    logging.info(f"  AIC: {aic:.2f}")
    logging.info(f"  BIC: {bic:.2f}")
    logging.info(f"  Converged: {converged}")
    
    # Compile all parameters into dictionary
    params = {
        'beta': beta,
        'beta_se': beta_se,
        'cutpoints': cutpoints,
        'cutpoints_se': cutpoints_se,
        'converged': converged,
        'n_obs': n_obs,
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic
    }
    
    return params


def calculate_confidence_intervals(params: Dict, 
                                   alpha: float = 0.05) -> Dict:
    """
    Calculate 95% confidence intervals for model parameters.
    
    Uses normal approximation:
        CI = estimate ± z_(α/2) * SE
        
    For 95% CI: z_(0.025) = 1.96
    
    Args:
        params: Dictionary from extract_model_parameters() containing:
                - 'beta': coefficient value
                - 'beta_se': standard error
                - 'cutpoints': dict of cutpoint values
                - 'cutpoints_se': dict of cutpoint standard errors
        alpha: Significance level (default: 0.05 for 95% CI)
    
    Returns:
        Dictionary containing:
            - 'beta_ci': tuple (lower, upper) - 95% CI for beta
            - 'cutpoints_ci': dict {k: (lower, upper)} - 95% CI for each cutpoint
            
    Requirements: 4.7
    """
    logging.info(f"Calculating {(1-alpha)*100:.0f}% confidence intervals")
    
    # Get critical value from standard normal distribution
    z_critical = norm.ppf(1 - alpha/2)
    
    # Calculate CI for beta
    beta = params['beta']
    beta_se = params['beta_se']
    beta_lower = beta - z_critical * beta_se
    beta_upper = beta + z_critical * beta_se
    beta_ci = (beta_lower, beta_upper)
    
    logging.info(
        f"Beta CI: [{beta_lower:.4f}, {beta_upper:.4f}] "
        f"(point estimate: {beta:.4f})"
    )
    
    # Calculate CIs for cutpoints
    cutpoints = params['cutpoints']
    cutpoints_se = params['cutpoints_se']
    cutpoints_ci = {}
    
    logging.info("Cutpoint confidence intervals:")
    for k in sorted(cutpoints.keys()):
        tau = cutpoints[k]
        tau_se = cutpoints_se[k]
        tau_lower = tau - z_critical * tau_se
        tau_upper = tau + z_critical * tau_se
        cutpoints_ci[k] = (tau_lower, tau_upper)
        
        logging.info(
            f"  τ_{k:+d} CI: [{tau_lower:.4f}, {tau_upper:.4f}] "
            f"(point estimate: {tau:.4f})"
        )
    
    # Compile confidence intervals
    cis = {
        'beta_ci': beta_ci,
        'cutpoints_ci': cutpoints_ci
    }
    
    return cis


# ============================================================================
# COMFORT CALCULATIONS
# ============================================================================

def calculate_pet_neutral(beta: float, 
                         tau_0: float,
                         beta_se: float,
                         tau_0_se: float,
                         alpha: float = 0.05) -> Dict:
    """
    Calculate PET neutral value and its 95% confidence interval.
    
    PET neutral is the temperature where the probability of "comfortable" (0)
    sensation is maximized. In the proportional odds model, this occurs at
    the cutpoint between categories 0 and +1:
    
        PET_neutral = τ_0 / β
    
    The confidence interval is calculated using the delta method for the
    ratio of two random variables:
    
        Var(τ/β) ≈ (1/β)² Var(τ) + (τ/β²)² Var(β)
        
    Assuming independence between τ and β (reasonable approximation).
    
    Args:
        beta: Model coefficient for PET_C
        tau_0: Cutpoint between comfortable (0) and warm (+1)
        beta_se: Standard error of beta
        tau_0_se: Standard error of tau_0
        alpha: Significance level (default: 0.05 for 95% CI)
    
    Returns:
        Dictionary containing:
            - 'pet_neutral': float - PET neutral value (°C)
            - 'pet_neutral_ci': tuple (lower, upper) - 95% CI
            - 'pet_neutral_se': float - Standard error (from delta method)
            
    Requirements: 5.1-5.4
    """
    logging.info("Calculating PET neutral")
    
    # Calculate PET neutral
    pet_neutral = tau_0 / beta
    
    logging.info(f"PET neutral = τ_0 / β = {tau_0:.4f} / {beta:.4f} = {pet_neutral:.2f}°C")
    
    # Calculate standard error using delta method
    # Var(τ/β) ≈ (1/β)² Var(τ) + (τ/β²)² Var(β)
    var_tau = tau_0_se ** 2
    var_beta = beta_se ** 2
    
    # Partial derivatives
    d_tau = 1 / beta  # ∂(τ/β)/∂τ = 1/β
    d_beta = -tau_0 / (beta ** 2)  # ∂(τ/β)/∂β = -τ/β²
    
    # Variance of PET neutral
    var_pet_neutral = (d_tau ** 2) * var_tau + (d_beta ** 2) * var_beta
    
    # Standard error
    pet_neutral_se = np.sqrt(var_pet_neutral)
    
    # Calculate confidence interval
    z_critical = norm.ppf(1 - alpha/2)
    pet_neutral_lower = pet_neutral - z_critical * pet_neutral_se
    pet_neutral_upper = pet_neutral + z_critical * pet_neutral_se
    pet_neutral_ci = (pet_neutral_lower, pet_neutral_upper)
    
    logging.info(
        f"PET neutral: {pet_neutral:.2f}°C "
        f"(95% CI: [{pet_neutral_lower:.2f}, {pet_neutral_upper:.2f}]°C, "
        f"SE: {pet_neutral_se:.2f})"
    )
    
    result = {
        'pet_neutral': pet_neutral,
        'pet_neutral_ci': pet_neutral_ci,
        'pet_neutral_se': pet_neutral_se
    }
    
    return result


def calculate_probabilities(beta: float, 
                           cutpoints: Dict[int, float]) -> Dict:
    """
    Calculate probabilities for all sensation categories across PET range.
    
    For each PET value in a fine grid, calculates:
    1. Cumulative probabilities: P(Y ≤ k | PET) for each threshold k
    2. Category probabilities: P(Y = k | PET) for each category k
    3. Comfort probability: P(-1 ≤ Y ≤ +1 | PET)
    
    Using the proportional odds model:
        P(Y ≤ k | PET) = expit(τ_k - β * PET)
        
    Where expit(x) = 1 / (1 + exp(-x)) is the logistic function.
    
    Category probabilities are derived from cumulative:
        P(Y = k | PET) = P(Y ≤ k | PET) - P(Y ≤ k-1 | PET)
    
    Comfort probability (union of categories -1, 0, +1):
        P_conf(PET) = P(Y ≤ +1 | PET) - P(Y ≤ -2 | PET)
    
    Args:
        beta: Model coefficient for PET_C
        cutpoints: Dictionary {k: τ_k} for k in {-3, -2, -1, 0, 1, 2}
    
    Returns:
        Dictionary containing:
            - 'pet_grid': np.ndarray - PET values from -5 to 55°C (step 0.05)
            - 'prob_cumulative': dict {k: array} - P(Y ≤ k) for each k
            - 'prob_category': dict {k: array} - P(Y = k) for each k ∈ {-3..+3}
            - 'prob_comfort': np.ndarray - P(-1 ≤ Y ≤ +1)
            
    Requirements: 6.2
    """
    logging.info("Calculating probability distributions across PET range")
    
    # Create PET grid
    pet_grid = np.arange(PET_GRID_MIN, PET_GRID_MAX + PET_GRID_STEP, PET_GRID_STEP)
    n_points = len(pet_grid)
    
    logging.info(
        f"PET grid: {PET_GRID_MIN}°C to {PET_GRID_MAX}°C, "
        f"step {PET_GRID_STEP}°C ({n_points} points)"
    )
    
    # Calculate cumulative probabilities P(Y ≤ k | PET) for each cutpoint
    prob_cumulative = {}
    
    # Sort cutpoints by ordinal value
    sorted_cutpoints = sorted(cutpoints.items())
    
    logging.info("Calculating cumulative probabilities for each threshold:")
    for k, tau_k in sorted_cutpoints:
        # P(Y ≤ k | PET) = expit(τ_k - β * PET)
        logit_values = tau_k - beta * pet_grid
        prob_cumulative[k] = expit(logit_values)
        
        # Log some sample probabilities
        mid_idx = len(pet_grid) // 2
        mid_pet = pet_grid[mid_idx]
        mid_prob = prob_cumulative[k][mid_idx]
        logging.debug(
            f"  P(Y ≤ {k:+d} | PET={mid_pet:.1f}°C) = {mid_prob:.3f}"
        )
    
    # Calculate category probabilities P(Y = k | PET)
    prob_category = {}
    
    # Get available cutpoints (may not have all if data is imbalanced)
    available_cutpoints = sorted(cutpoints.keys())
    min_cutpoint = min(available_cutpoints)
    max_cutpoint = max(available_cutpoints)
    
    # All possible ordinal values
    ordinal_values = list(range(-3, 4))  # -3, -2, -1, 0, 1, 2, 3
    
    logging.info("Calculating category probabilities:")
    for k in ordinal_values:
        if k < min_cutpoint:
            # Category below minimum cutpoint: P(Y = k) = P(Y ≤ min_cutpoint)
            if k == min_cutpoint:
                prob_category[k] = prob_cumulative[min_cutpoint]
            else:
                # For categories below the minimum, assign very small probability
                prob_category[k] = np.zeros_like(pet_grid)
                logging.debug(f"  P(Y = {k:+d}): not in data (below minimum cutpoint)")
        elif k > max_cutpoint + 1:
            # Category above maximum cutpoint: P(Y = k) = 1 - P(Y ≤ max_cutpoint)
            if k == max_cutpoint + 1:
                prob_category[k] = 1 - prob_cumulative[max_cutpoint]
            else:
                # For categories above the maximum, assign very small probability
                prob_category[k] = np.zeros_like(pet_grid)
                logging.debug(f"  P(Y = {k:+d}): not in data (above maximum cutpoint)")
        elif k == min_cutpoint:
            # First category with data
            prob_category[k] = prob_cumulative[k]
        elif k - 1 in prob_cumulative and k in prob_cumulative:
            # P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)
            prob_category[k] = prob_cumulative[k] - prob_cumulative[k-1]
        elif k - 1 not in prob_cumulative and k in prob_cumulative:
            # Missing previous cutpoint, use current as lower bound
            prob_category[k] = prob_cumulative[k]
        elif k not in prob_cumulative and k - 1 in prob_cumulative:
            # Missing current cutpoint, use 1 - previous
            prob_category[k] = 1 - prob_cumulative[k-1]
        else:
            # Both missing, assign zero probability
            prob_category[k] = np.zeros_like(pet_grid)
            logging.debug(f"  P(Y = {k:+d}): not in data (missing cutpoints)")
        
        # Log average probability for this category
        if k in prob_category:
            avg_prob = np.mean(prob_category[k])
            logging.debug(f"  P(Y = {k:+d}): mean = {avg_prob:.3f}")
    
    # Calculate comfort probability P(-1 ≤ Y ≤ +1)
    # This is the union of categories -1, 0, and +1
    # P(-1 ≤ Y ≤ +1) = P(Y ≤ +1) - P(Y ≤ -2)
    # Handle missing cutpoints gracefully
    if 1 in prob_cumulative and -2 in prob_cumulative:
        prob_comfort = prob_cumulative[1] - prob_cumulative[-2]
    elif 1 in prob_cumulative:
        # Missing lower bound, use P(Y ≤ +1) as approximation
        prob_comfort = prob_cumulative[1]
        logging.warning("Cutpoint -2 missing, using P(Y ≤ +1) as comfort probability")
    elif -2 in prob_cumulative:
        # Missing upper bound, use 1 - P(Y ≤ -2) as approximation
        prob_comfort = 1 - prob_cumulative[-2]
        logging.warning("Cutpoint +1 missing, using 1 - P(Y ≤ -2) as comfort probability")
    else:
        # Both missing, sum individual category probabilities
        prob_comfort = np.zeros_like(pet_grid)
        for k in [-1, 0, 1]:
            if k in prob_category:
                prob_comfort += prob_category[k]
        logging.warning("Cutpoints -2 and +1 missing, using sum of category probabilities")
    
    # Log comfort probability statistics
    max_comfort_prob = np.max(prob_comfort)
    max_comfort_idx = np.argmax(prob_comfort)
    max_comfort_pet = pet_grid[max_comfort_idx]
    
    logging.info(
        f"Comfort probability: max = {max_comfort_prob:.3f} "
        f"at PET = {max_comfort_pet:.2f}°C"
    )
    
    # Compile results
    result = {
        'pet_grid': pet_grid,
        'prob_cumulative': prob_cumulative,
        'prob_category': prob_category,
        'prob_comfort': prob_comfort
    }
    
    return result


def calculate_comfort_bands(pet_grid: np.ndarray,
                           prob_comfort: np.ndarray,
                           thresholds: List[float] = None) -> Dict:
    """
    Find PET ranges where comfort probability exceeds specified thresholds.
    
    For each threshold (e.g., 0.8 for 80%, 0.9 for 90%), finds the continuous
    range of PET values where P_conf(PET) ≥ threshold.
    
    Algorithm:
        1. For each threshold p:
        2. Find all indices where prob_comfort >= p
        3. L_p = min(pet_grid[indices]) - lower limit
        4. U_p = max(pet_grid[indices]) - upper limit
        5. Return band [L_p, U_p]
    
    Args:
        pet_grid: Array of PET values (e.g., -5 to 55°C with 0.05 step)
        prob_comfort: Comfort probability P(-1 ≤ Y ≤ +1) at each PET
        thresholds: List of probability thresholds (default: [0.8, 0.9])
    
    Returns:
        Dictionary with bands for each threshold:
            {
                0.8: {'lower': L_80, 'upper': U_80, 'width': U_80 - L_80},
                0.9: {'lower': L_90, 'upper': U_90, 'width': U_90 - L_90}
            }
            
        If no PET values meet a threshold, that threshold's entry will have
        None values for lower and upper.
        
    Requirements: 6.1, 6.3-6.7
    """
    if thresholds is None:
        thresholds = COMFORT_THRESHOLDS
    
    logging.info(f"Calculating comfort bands for thresholds: {thresholds}")
    
    # Validate inputs
    if len(pet_grid) != len(prob_comfort):
        raise ValueError(
            f"pet_grid and prob_comfort must have same length. "
            f"Got {len(pet_grid)} and {len(prob_comfort)}"
        )
    
    bands = {}
    
    for threshold in thresholds:
        logging.info(f"Finding comfort band for {threshold*100:.0f}% threshold")
        
        # Find indices where comfort probability meets or exceeds threshold
        indices = np.where(prob_comfort >= threshold)[0]
        
        if len(indices) == 0:
            # No PET values meet this threshold
            logging.warning(
                f"No PET values found where comfort probability >= {threshold:.2f}. "
                f"Maximum comfort probability: {np.max(prob_comfort):.3f}"
            )
            bands[threshold] = {
                'lower': None,
                'upper': None,
                'width': None
            }
        else:
            # Extract lower and upper limits
            lower_limit = pet_grid[indices[0]]
            upper_limit = pet_grid[indices[-1]]
            width = upper_limit - lower_limit
            
            bands[threshold] = {
                'lower': lower_limit,
                'upper': upper_limit,
                'width': width
            }
            
            logging.info(
                f"  {threshold*100:.0f}% comfort band: "
                f"[{lower_limit:.1f}, {upper_limit:.1f}]°C "
                f"(width: {width:.1f}°C)"
            )
            
            # Log additional statistics
            n_points = len(indices)
            pct_range = (n_points / len(pet_grid)) * 100
            logging.debug(
                f"  Band covers {n_points} grid points "
                f"({pct_range:.1f}% of PET range)"
            )
    
    return bands


# ============================================================================
# ACCEPTABILITY ANALYSIS (OPTIONAL)
# ============================================================================

def check_acceptability_column(df: pd.DataFrame, 
                               acceptability_col: str = 'Acceptability') -> Tuple[bool, Optional[pd.Series]]:
    """
    Check if acceptability column exists and convert to binary (0/1).
    
    Acceptability is typically recorded as text responses like:
    - "aceitável" / "acceptable" -> 1
    - "inaceitável" / "unacceptable" -> 0
    
    The function normalizes text (lowercase, no accents) before mapping.
    
    Args:
        df: Input DataFrame
        acceptability_col: Name of acceptability column (default: 'Acceptability')
    
    Returns:
        Tuple of (is_available, binary_series)
        - is_available: True if column exists and has valid data
        - binary_series: pandas Series with binary values (0/1), or None if unavailable
        
    Requirements: 7.1
    """
    logging.info(f"Checking for acceptability column: '{acceptability_col}'")
    
    # Check if column exists
    if acceptability_col not in df.columns:
        logging.info(
            f"Acceptability column '{acceptability_col}' not found in DataFrame. "
            f"Skipping acceptability analysis."
        )
        return False, None
    
    # Get the column
    acceptability_raw = df[acceptability_col].copy()
    
    # Check if column has any non-null values
    n_non_null = acceptability_raw.notna().sum()
    if n_non_null == 0:
        logging.warning(
            f"Acceptability column '{acceptability_col}' exists but contains no valid data. "
            f"Skipping acceptability analysis."
        )
        return False, None
    
    logging.info(f"Found acceptability column with {n_non_null} non-null values")
    
    # Normalize text values
    acceptability_normalized = acceptability_raw.apply(
        lambda x: normalize_text(x) if pd.notna(x) else x
    )
    
    # Define mapping for acceptability
    # Common Portuguese and English terms
    acceptability_mapping = {
        'aceitavel': 1,      # Portuguese without accent
        'acceptable': 1,      # English
        'sim': 1,            # Portuguese "yes"
        'yes': 1,            # English
        '1': 1,              # Numeric string
        'inaceitavel': 0,    # Portuguese without accent
        'unacceptable': 0,   # English
        'nao': 0,            # Portuguese "no" without tilde
        'no': 0,             # English
        '0': 0               # Numeric string
    }
    
    # Map to binary
    acceptability_binary = acceptability_normalized.map(acceptability_mapping)
    
    # Check how many values were successfully mapped
    n_mapped = acceptability_binary.notna().sum()
    n_unmapped = n_non_null - n_mapped
    
    if n_unmapped > 0:
        unmapped_values = acceptability_normalized[
            acceptability_normalized.notna() & acceptability_binary.isna()
        ].unique()
        logging.warning(
            f"Could not map {n_unmapped} acceptability values to binary. "
            f"Unmapped values: {list(unmapped_values)[:10]}"  # Show first 10
        )
    
    if n_mapped < 10:
        logging.warning(
            f"Only {n_mapped} acceptability values successfully mapped. "
            f"This may be insufficient for reliable analysis. "
            f"Skipping acceptability analysis."
        )
        return False, None
    
    # Log distribution
    value_counts = acceptability_binary.value_counts()
    logging.info("Acceptability distribution:")
    logging.info(f"  Acceptable (1): {value_counts.get(1, 0)} responses")
    logging.info(f"  Unacceptable (0): {value_counts.get(0, 0)} responses")
    
    # Check for sufficient variation
    if len(value_counts) < 2:
        logging.warning(
            "Acceptability has no variation (all same value). "
            "Cannot fit logistic regression. Skipping acceptability analysis."
        )
        return False, None
    
    logging.info(f"Acceptability column is valid and ready for analysis")
    
    return True, acceptability_binary


def fit_acceptability_model(df: pd.DataFrame,
                            acceptability_binary: pd.Series,
                            pet_col: str = 'PET_C') -> Optional[Dict]:
    """
    Fit binary logistic regression for acceptability.
    
    Model specification:
        Acceptable (0/1) ~ PET_C
        
    Uses statsmodels Logit model with binary response.
    
    Args:
        df: DataFrame with PET_C column
        acceptability_binary: Binary series (0/1) for acceptability
        pet_col: Name of PET column (default: 'PET_C')
    
    Returns:
        Dictionary containing model results:
            - 'converged': bool - Convergence status
            - 'n_obs': int - Number of observations
            - 'coef_intercept': float - Intercept coefficient
            - 'coef_pet': float - PET coefficient
            - 'se_intercept': float - Standard error of intercept
            - 'se_pet': float - Standard error of PET coefficient
            - 'log_likelihood': float - Model log-likelihood
            - 'aic': float - AIC
            - 'bic': float - BIC
            - 'model_result': Logit result object (for further analysis)
            
        Returns None if model fails to fit or converge.
        
    Requirements: 7.2
    """
    logging.info("Fitting binary logistic regression for acceptability")
    
    # Prepare data - combine PET and acceptability, remove NaN
    df_model = pd.DataFrame({
        'PET_C': df[pet_col],
        'Acceptable': acceptability_binary
    }).dropna()
    
    n_obs = len(df_model)
    
    if n_obs < 20:
        logging.warning(
            f"Insufficient data for acceptability model: {n_obs} observations. "
            f"Minimum recommended: 20. Skipping acceptability analysis."
        )
        return None
    
    logging.info(f"Fitting acceptability model with {n_obs} observations")
    
    # Check for variation in both variables
    if df_model['Acceptable'].nunique() < 2:
        logging.warning(
            "No variation in acceptability (all same value). "
            "Cannot fit logistic regression."
        )
        return None
    
    if df_model['PET_C'].nunique() < 5:
        logging.warning(
            f"Limited variation in PET_C: only {df_model['PET_C'].nunique()} unique values. "
            f"Acceptability model may be unstable."
        )
    
    try:
        # Prepare exogenous variables (add constant for intercept)
        X = sm.add_constant(df_model['PET_C'])
        y = df_model['Acceptable']
        
        # Fit logistic regression
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=False)
        
        # Check convergence
        if not result.mle_retvals['converged']:
            logging.warning(
                "Acceptability model failed to converge. "
                "Results may be unreliable. Skipping acceptability analysis."
            )
            return None
        
        logging.info("Acceptability model converged successfully")
        
        # Extract parameters
        coef_intercept = result.params['const']
        coef_pet = result.params['PET_C']
        se_intercept = result.bse['const']
        se_pet = result.bse['PET_C']
        
        logging.info(f"Model coefficients:")
        logging.info(f"  Intercept: {coef_intercept:.4f} (SE: {se_intercept:.4f})")
        logging.info(f"  PET_C: {coef_pet:.4f} (SE: {se_pet:.4f})")
        logging.info(f"  Log-likelihood: {result.llf:.2f}")
        logging.info(f"  AIC: {result.aic:.2f}")
        logging.info(f"  BIC: {result.bic:.2f}")
        
        # Compile results
        model_results = {
            'converged': True,
            'n_obs': n_obs,
            'coef_intercept': coef_intercept,
            'coef_pet': coef_pet,
            'se_intercept': se_intercept,
            'se_pet': se_pet,
            'log_likelihood': result.llf,
            'aic': result.aic,
            'bic': result.bic,
            'model_result': result
        }
        
        return model_results
        
    except Exception as e:
        logging.warning(
            f"Error fitting acceptability model: {e}. "
            f"Skipping acceptability analysis."
        )
        return None




def calculate_observed_pet_ranges(df, pet_col='PET_C'):
    """
    Calculate observed PET ranges for each sensation category using descriptive statistics.
    
    This method is more robust than model-based ranges when data has high variability
    or when the ordinal model fails to converge properly.
    
    Args:
        df: DataFrame with TSV_ordinal and PET columns
        pet_col: Name of PET column (default: 'PET_C')
    
    Returns:
        Dictionary with ranges for each sensation ordinal value
    """
    logging.info("Calculating observed PET ranges (descriptive statistics)")
    
    # Validate columns exist
    if 'TSV_ordinal' not in df.columns:
        raise ValidationError("Column 'TSV_ordinal' not found in DataFrame")
    if pet_col not in df.columns:
        raise ValidationError(f"Column '{pet_col}' not found in DataFrame")
    
    ranges = {}
    
    # Get unique ordinal values from data
    ordinal_values = sorted(df['TSV_ordinal'].dropna().unique())
    
    for ordinal in ordinal_values:
        # Get label for this ordinal value
        sens_label = SENSATION_LABELS.get(int(ordinal), f'{int(ordinal):+d}')
        
        # Get data for this sensation
        data = df[df['TSV_ordinal'] == ordinal][pet_col]
        
        if len(data) == 0:
            logging.warning(f"No data for sensation '{sens_label}' ({int(ordinal):+d})")
            continue
        
        # Calculate statistics
        n = len(data)
        mean = float(data.mean())
        median = float(data.median())
        std = float(data.std())
        
        # Calculate percentile ranges
        p10 = float(data.quantile(0.10))
        p25 = float(data.quantile(0.25))
        p75 = float(data.quantile(0.75))
        p90 = float(data.quantile(0.90))
        min_val = float(data.min())
        max_val = float(data.max())
        
        # Compile ranges
        range_50 = {
            'lower': p25,
            'upper': p75,
            'width': p75 - p25
        }
        
        range_80 = {
            'lower': p10,
            'upper': p90,
            'width': p90 - p10
        }
        
        range_full = {
            'lower': min_val,
            'upper': max_val,
            'width': max_val - min_val
        }
        
        ranges[int(ordinal)] = {
            'label': sens_label,
            'ordinal': int(ordinal),
            'n': n,
            'mean': mean,
            'median': median,
            'std': std,
            'range_50': range_50,
            'range_80': range_80,
            'range_full': range_full
        }
        
        logging.info(
            f"  {sens_label}: n={n}, mean={mean:.1f}°C, "
            f"50%=[{p25:.1f}, {p75:.1f}]°C, "
            f"80%=[{p10:.1f}, {p90:.1f}]°C"
        )
    
    logging.info("✓ Observed PET ranges calculated")
    
    return ranges


def calculate_acceptability_bands(model_results: Dict,
                                  thresholds: List[float] = None,
                                  pet_grid: np.ndarray = None) -> Optional[Dict]:
    """
    Calculate PET ranges for specified acceptability thresholds.
    
    For each threshold (e.g., 0.8 for 80%, 0.9 for 90%), finds the continuous
    range of PET values where P(Acceptable | PET) ≥ threshold.
    
    Uses the fitted logistic regression model:
        P(Acceptable | PET) = expit(β_0 + β_1 * PET)
        
    Where expit(x) = 1 / (1 + exp(-x)) is the logistic function.
    
    Args:
        model_results: Dictionary from fit_acceptability_model() containing:
                      - 'coef_intercept': intercept coefficient
                      - 'coef_pet': PET coefficient
        thresholds: List of probability thresholds (default: [0.8, 0.9])
        pet_grid: Array of PET values to evaluate (default: -5 to 55°C, step 0.05)
    
    Returns:
        Dictionary with bands for each threshold:
            {
                0.8: {'lower': L_80, 'upper': U_80, 'width': U_80 - L_80},
                0.9: {'lower': L_90, 'upper': U_90, 'width': U_90 - L_90},
                'pet_grid': array of PET values,
                'prob_acceptable': array of acceptability probabilities
            }
            
        Returns None if model_results is None or invalid.
        
    Requirements: 7.3, 7.4
    """
    if model_results is None:
        logging.info("No acceptability model results provided. Skipping band calculation.")
        return None
    
    if thresholds is None:
        thresholds = COMFORT_THRESHOLDS
    
    if pet_grid is None:
        pet_grid = np.arange(PET_GRID_MIN, PET_GRID_MAX + PET_GRID_STEP, PET_GRID_STEP)
    
    logging.info(f"Calculating acceptability bands for thresholds: {thresholds}")
    
    # Extract coefficients
    beta_0 = model_results['coef_intercept']
    beta_1 = model_results['coef_pet']
    
    # Calculate acceptability probability across PET range
    # P(Acceptable | PET) = expit(β_0 + β_1 * PET)
    logit_values = beta_0 + beta_1 * pet_grid
    prob_acceptable = expit(logit_values)
    
    # Log some statistics
    max_prob = np.max(prob_acceptable)
    max_prob_idx = np.argmax(prob_acceptable)
    max_prob_pet = pet_grid[max_prob_idx]
    
    min_prob = np.min(prob_acceptable)
    min_prob_idx = np.argmin(prob_acceptable)
    min_prob_pet = pet_grid[min_prob_idx]
    
    logging.info(
        f"Acceptability probability range: "
        f"[{min_prob:.3f} at {min_prob_pet:.1f}°C, "
        f"{max_prob:.3f} at {max_prob_pet:.1f}°C]"
    )
    
    # Calculate bands for each threshold
    bands = {}
    
    for threshold in thresholds:
        logging.info(f"Finding acceptability band for {threshold*100:.0f}% threshold")
        
        # Find indices where acceptability probability meets or exceeds threshold
        indices = np.where(prob_acceptable >= threshold)[0]
        
        if len(indices) == 0:
            # No PET values meet this threshold
            logging.warning(
                f"No PET values found where acceptability >= {threshold:.2f}. "
                f"Maximum acceptability: {max_prob:.3f}"
            )
            bands[threshold] = {
                'lower': None,
                'upper': None,
                'width': None
            }
        else:
            # Extract lower and upper limits
            lower_limit = pet_grid[indices[0]]
            upper_limit = pet_grid[indices[-1]]
            width = upper_limit - lower_limit
            
            bands[threshold] = {
                'lower': lower_limit,
                'upper': upper_limit,
                'width': width
            }
            
            logging.info(
                f"  {threshold*100:.0f}% acceptability band: "
                f"[{lower_limit:.1f}, {upper_limit:.1f}]°C "
                f"(width: {width:.1f}°C)"
            )
    
    # Add probability data for potential visualization
    bands['pet_grid'] = pet_grid
    bands['prob_acceptable'] = prob_acceptable
    
    return bands


def calculate_category_pet_ranges(pet_grid: np.ndarray,
                                  prob_category: Dict[int, np.ndarray],
                                  df: pd.DataFrame = None,
                                  threshold: float = 0.3) -> Dict:
    """
    Calculate PET ranges for each thermal sensation category.
    
    For each sensation category (e.g., "Calor", "Frio"), determines:
    1. The PET range where that category has the highest probability (modal range)
    2. The PET range where that category has probability >= threshold
    3. Observed PET statistics from actual data (if df provided)
    
    This helps answer questions like:
    - "When people feel 'Calor' (+2), what is the typical PET range?"
    - "At what PET values is 'Frio' (-2) the most likely sensation?"
    
    Args:
        pet_grid: Array of PET values (e.g., -5 to 55°C with 0.05 step)
        prob_category: Dictionary {k: prob_array} with probability of each category
        df: Optional DataFrame with 'PET_C' and 'TSV_ordinal' columns for observed stats
        threshold: Minimum probability threshold for range (default: 0.3 = 30%)
    
    Returns:
        Dictionary with ranges for each category:
        {
            -3: {
                'label': 'Muito Frio',
                'modal_range': {'lower': L, 'upper': U, 'peak_pet': P},
                'threshold_range': {'lower': L, 'upper': U, 'threshold': 0.3},
                'observed': {'mean': M, 'median': Med, 'std': S, 'min': Min, 'max': Max, 'n': N}
            },
            ...
        }
        
    Requirements: New feature for category-specific PET calibration
    """
    logging.info("Calculating PET ranges for each thermal sensation category")
    
    ranges = {}
    
    for k in sorted(prob_category.keys()):
        label = SENSATION_LABELS.get(k, f'{k:+d}')
        logging.info(f"Analyzing category {k:+d} ({label})")
        
        prob = prob_category[k]
        
        # 1. Find modal range (where this category has highest probability)
        # First, find where this category is the most probable
        all_probs = np.array([prob_category[cat] for cat in sorted(prob_category.keys())])
        is_modal = np.argmax(all_probs, axis=0) == list(sorted(prob_category.keys())).index(k)
        modal_indices = np.where(is_modal)[0]
        
        if len(modal_indices) > 0:
            modal_lower = pet_grid[modal_indices[0]]
            modal_upper = pet_grid[modal_indices[-1]]
            # Find peak probability within modal range
            peak_idx = modal_indices[np.argmax(prob[modal_indices])]
            peak_pet = pet_grid[peak_idx]
            peak_prob = prob[peak_idx]
            
            modal_range = {
                'lower': modal_lower,
                'upper': modal_upper,
                'peak_pet': peak_pet,
                'peak_prob': peak_prob,
                'width': modal_upper - modal_lower
            }
            
            logging.info(
                f"  Modal range: [{modal_lower:.1f}, {modal_upper:.1f}]°C "
                f"(peak at {peak_pet:.1f}°C, prob={peak_prob:.3f})"
            )
        else:
            modal_range = {
                'lower': None,
                'upper': None,
                'peak_pet': None,
                'peak_prob': 0.0,
                'width': None
            }
            logging.info(f"  No modal range found (category never most probable)")
        
        # 2. Find threshold range (where probability >= threshold)
        threshold_indices = np.where(prob >= threshold)[0]
        
        if len(threshold_indices) > 0:
            threshold_lower = pet_grid[threshold_indices[0]]
            threshold_upper = pet_grid[threshold_indices[-1]]
            
            threshold_range = {
                'lower': threshold_lower,
                'upper': threshold_upper,
                'width': threshold_upper - threshold_lower,
                'threshold': threshold
            }
            
            logging.info(
                f"  Threshold range (≥{threshold*100:.0f}%): "
                f"[{threshold_lower:.1f}, {threshold_upper:.1f}]°C"
            )
        else:
            threshold_range = {
                'lower': None,
                'upper': None,
                'width': None,
                'threshold': threshold
            }
            logging.info(f"  No range found with probability ≥ {threshold*100:.0f}%")
        
        # 3. Calculate observed statistics from data (if provided)
        observed = None
        if df is not None and 'PET_C' in df.columns and 'TSV_ordinal' in df.columns:
            category_data = df[df['TSV_ordinal'] == k]['PET_C']
            
            if len(category_data) > 0:
                observed = {
                    'mean': float(category_data.mean()),
                    'median': float(category_data.median()),
                    'std': float(category_data.std()),
                    'min': float(category_data.min()),
                    'max': float(category_data.max()),
                    'n': len(category_data),
                    'percentile_25': float(category_data.quantile(0.25)),
                    'percentile_75': float(category_data.quantile(0.75))
                }
                
                logging.info(
                    f"  Observed PET: mean={observed['mean']:.1f}°C, "
                    f"median={observed['median']:.1f}°C, "
                    f"range=[{observed['min']:.1f}, {observed['max']:.1f}]°C (n={observed['n']})"
                )
            else:
                logging.info(f"  No observed data for this category")
        
        # Compile results for this category
        ranges[k] = {
            'label': label,
            'ordinal': k,
            'modal_range': modal_range,
            'threshold_range': threshold_range,
            'observed': observed
        }
    
    logging.info("✓ Category PET ranges calculated")
    
    return ranges


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_scatter_tsv_pet(df: pd.DataFrame,
                        pet_neutral: float,
                        output_path: str,
                        filename: str = 'scatter_TSV_PET.png') -> str:
    """
    Create scatter plot of PET vs TSV ordinal with jitter.
    
    Features:
    - PET on X-axis, TSV_ordinal on Y-axis
    - Jitter on Y-axis (random noise ±0.1) to show overlapping points
    - Color-coded by sensation category using COLOR_PALETTE
    - Vertical line at PET neutral
    - Grid for readability
    - Legend with sensation labels
    
    Args:
        df: DataFrame with 'PET_C' and 'TSV_ordinal' columns
        pet_neutral: PET neutral value for vertical line marker
        output_path: Directory path for output file
        filename: Name of output file (default: 'scatter_TSV_PET.png')
    
    Returns:
        Full path to saved plot file
        
    Raises:
        IOError: If plot cannot be saved
        
    Requirements: 8.1, 8.2, 8.8, 8.9
    """
    logging.info(f"Creating scatter plot: TSV vs PET")
    
    # Validate required columns
    required_cols = ['PET_C', 'TSV_ordinal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Missing required columns for scatter plot: {missing_cols}"
        )
    
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(output_path, filename)
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    
    # Add jitter to Y-axis (±0.1 random noise)
    np.random.seed(42)  # For reproducibility
    jitter = np.random.uniform(-0.1, 0.1, size=len(df))
    y_jittered = df['TSV_ordinal'] + jitter
    
    # Plot points for each sensation category with distinct colors
    for ordinal_val in sorted(df['TSV_ordinal'].unique()):
        mask = df['TSV_ordinal'] == ordinal_val
        ordinal_int = int(ordinal_val)
        
        ax.scatter(
            df.loc[mask, 'PET_C'],
            y_jittered[mask],
            c=COLOR_PALETTE.get(ordinal_int, '#666666'),
            label=SENSATION_LABELS.get(ordinal_int, f'{ordinal_int:+d}'),
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidths=0.5
        )
    
    # Add vertical line at PET neutral
    ax.axvline(
        pet_neutral,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'PET Neutro ({pet_neutral:.1f}°C)',
        zorder=10
    )
    
    # Configure axes
    ax.set_xlabel('PET (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sensação Térmica (Ordinal)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Relação entre PET e Sensação Térmica',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Set Y-axis ticks to ordinal values
    ax.set_yticks(range(-3, 4))
    ax.set_yticklabels([f'{i:+d}' for i in range(-3, 4)])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add legend
    ax.legend(
        loc='best',
        framealpha=0.9,
        fontsize=9,
        ncol=2
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    try:
        plt.savefig(file_path, dpi=PLOT_DPI, bbox_inches='tight')
        logging.info(f"Saved scatter plot to {file_path}")
    except Exception as e:
        plt.close(fig)
        raise IOError(f"Error saving scatter plot to {file_path}: {e}")
    
    # Close figure to free memory
    plt.close(fig)
    
    return file_path


def plot_probability_curves(pet_grid: np.ndarray,
                           prob_category: Dict[int, np.ndarray],
                           pet_neutral: float,
                           comfort_bands: Dict,
                           output_path: str,
                           filename: str = 'probs_ordinais_PET.png') -> str:
    """
    Plot probability curves for all 7 sensation categories.
    
    Features:
    - P(Y=k|PET) curves for all categories k ∈ {-3, -2, -1, 0, 1, 2, 3}
    - Distinct colors for each category from COLOR_PALETTE
    - Vertical line at PET neutral
    - Shaded regions for 80% and 90% comfort bands
    - Legend with category names in Portuguese
    - Grid and axis labels
    
    Args:
        pet_grid: Array of PET values
        prob_category: Dictionary {k: prob_array} for each category k
        pet_neutral: PET neutral value for vertical line marker
        comfort_bands: Dictionary with 80% and 90% band limits
        output_path: Directory path for output file
        filename: Name of output file (default: 'probs_ordinais_PET.png')
    
    Returns:
        Full path to saved plot file
        
    Raises:
        IOError: If plot cannot be saved
        
    Requirements: 8.3, 8.4, 8.8, 8.9
    """
    logging.info(f"Creating probability curves plot")
    
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(output_path, filename)
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    
    # Plot probability curves for each category
    for ordinal_val in sorted(prob_category.keys()):
        ordinal_int = int(ordinal_val)
        
        ax.plot(
            pet_grid,
            prob_category[ordinal_val],
            color=COLOR_PALETTE.get(ordinal_int, '#666666'),
            label=SENSATION_LABELS.get(ordinal_int, f'{ordinal_int:+d}'),
            linewidth=2.5,
            alpha=0.8
        )
    
    # Shade comfort bands if available
    # 80% band (lighter shade)
    if 0.8 in comfort_bands and comfort_bands[0.8]['lower'] is not None:
        L_80 = comfort_bands[0.8]['lower']
        U_80 = comfort_bands[0.8]['upper']
        ax.axvspan(
            L_80, U_80,
            alpha=0.15,
            color='green',
            label=f'Faixa 80% [{L_80:.1f}, {U_80:.1f}]°C'
        )
    
    # 90% band (darker shade)
    if 0.9 in comfort_bands and comfort_bands[0.9]['lower'] is not None:
        L_90 = comfort_bands[0.9]['lower']
        U_90 = comfort_bands[0.9]['upper']
        ax.axvspan(
            L_90, U_90,
            alpha=0.25,
            color='darkgreen',
            label=f'Faixa 90% [{L_90:.1f}, {U_90:.1f}]°C'
        )
    
    # Add vertical line at PET neutral
    ax.axvline(
        pet_neutral,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'PET Neutro ({pet_neutral:.1f}°C)',
        zorder=10
    )
    
    # Configure axes
    ax.set_xlabel('PET (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probabilidade P(Y=k|PET)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Curvas de Probabilidade por Categoria de Sensação',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Set Y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add legend
    ax.legend(
        loc='upper left',
        framealpha=0.9,
        fontsize=8,
        ncol=2
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    try:
        plt.savefig(file_path, dpi=PLOT_DPI, bbox_inches='tight')
        logging.info(f"Saved probability curves plot to {file_path}")
    except Exception as e:
        plt.close(fig)
        raise IOError(f"Error saving probability curves plot to {file_path}: {e}")
    
    # Close figure to free memory
    plt.close(fig)
    
    return file_path


def plot_comfort_zone(pet_grid: np.ndarray,
                     prob_comfort: np.ndarray,
                     comfort_bands: Dict,
                     output_path: str,
                     filename: str = 'zona_conforto_logit.png') -> str:
    """
    Plot comfort zone probability with threshold lines and band annotations.
    
    Features:
    - Curve of P_conf(PET) = P(-1 ≤ Y ≤ +1 | PET)
    - Horizontal lines at 0.8 and 0.9 thresholds
    - Vertical lines at L_80, U_80, L_90, U_90
    - Shaded region between L_80 and U_80
    - Annotations for band limits
    - Grid and axis labels
    
    Args:
        pet_grid: Array of PET values
        prob_comfort: Comfort probability P(-1 ≤ Y ≤ +1) at each PET
        comfort_bands: Dictionary with band limits for 0.8 and 0.9 thresholds
        output_path: Directory path for output file
        filename: Name of output file (default: 'zona_conforto_logit.png')
    
    Returns:
        Full path to saved plot file
        
    Raises:
        IOError: If plot cannot be saved
        
    Requirements: 8.5, 8.6, 8.7, 8.8, 8.9
    """
    logging.info(f"Creating comfort zone plot")
    
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(output_path, filename)
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    
    # Plot comfort probability curve
    ax.plot(
        pet_grid,
        prob_comfort,
        color='#2ca02c',  # Green
        linewidth=3,
        label='P(Conforto) = P(-1 ≤ Y ≤ +1 | PET)'
    )
    
    # Add horizontal threshold lines
    ax.axhline(
        0.8,
        color='orange',
        linestyle='-',
        linewidth=1.5,
        alpha=0.7,
        label='Limiar 80%'
    )
    ax.axhline(
        0.9,
        color='red',
        linestyle='-',
        linewidth=1.5,
        alpha=0.7,
        label='Limiar 90%'
    )
    
    # Add vertical lines and annotations for 80% band
    if 0.8 in comfort_bands and comfort_bands[0.8]['lower'] is not None:
        L_80 = comfort_bands[0.8]['lower']
        U_80 = comfort_bands[0.8]['upper']
        
        # Vertical lines
        ax.axvline(L_80, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(U_80, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Shade region
        ax.axvspan(
            L_80, U_80,
            alpha=0.2,
            color='orange',
            label=f'Faixa 80%'
        )
        
        # Annotations
        ax.annotate(
            f'L₈₀ = {L_80:.1f}°C',
            xy=(L_80, 0.8),
            xytext=(L_80 - 3, 0.75),
            fontsize=9,
            ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        ax.annotate(
            f'U₈₀ = {U_80:.1f}°C',
            xy=(U_80, 0.8),
            xytext=(U_80 + 3, 0.75),
            fontsize=9,
            ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    # Add vertical lines and annotations for 90% band
    if 0.9 in comfort_bands and comfort_bands[0.9]['lower'] is not None:
        L_90 = comfort_bands[0.9]['lower']
        U_90 = comfort_bands[0.9]['upper']
        
        # Vertical lines
        ax.axvline(L_90, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(U_90, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Annotations
        ax.annotate(
            f'L₉₀ = {L_90:.1f}°C',
            xy=(L_90, 0.9),
            xytext=(L_90 - 3, 0.95),
            fontsize=9,
            ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        ax.annotate(
            f'U₉₀ = {U_90:.1f}°C',
            xy=(U_90, 0.9),
            xytext=(U_90 + 3, 0.95),
            fontsize=9,
            ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    # Configure axes
    ax.set_xlabel('PET (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probabilidade de Conforto', fontsize=12, fontweight='bold')
    ax.set_title(
        'Zona de Conforto Térmico',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Set Y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add legend
    ax.legend(
        loc='lower right',
        framealpha=0.9,
        fontsize=9
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    try:
        plt.savefig(file_path, dpi=PLOT_DPI, bbox_inches='tight')
        logging.info(f"Saved comfort zone plot to {file_path}")
    except Exception as e:
        plt.close(fig)
        raise IOError(f"Error saving comfort zone plot to {file_path}: {e}")
    
    # Close figure to free memory
    plt.close(fig)
    
    return file_path


# ============================================================================
# REPORT GENERATION
# ============================================================================

def format_descriptive_statistics(df: pd.DataFrame, 
                                  removal_stats: Dict) -> Dict:
    """
    Calculate and format descriptive statistics for the report.
    
    Calculates:
    - N total (before cleaning)
    - N valid (after cleaning)
    - % valid (percentage retained)
    - PET statistics (mean, median, std, min, max)
    - Count of observations per sensation category
    
    Args:
        df: Cleaned DataFrame with 'PET_C' and 'TSV_ordinal' columns
        removal_stats: Dictionary from remove_invalid_rows() containing:
                      - n_total: total rows before cleaning
                      - n_retained: rows after cleaning
                      - pct_retained: percentage retained
    
    Returns:
        Dictionary containing formatted statistics:
            - 'n_total': int - Total responses before cleaning
            - 'n_valid': int - Valid responses after cleaning
            - 'pct_valid': float - Percentage of valid responses
            - 'pet_mean': float - Mean PET value
            - 'pet_median': float - Median PET value
            - 'pet_std': float - Standard deviation of PET
            - 'pet_min': float - Minimum PET value
            - 'pet_max': float - Maximum PET value
            - 'sensation_counts': dict - {ordinal_value: count}
            - 'sensation_distribution': list - [(label, ordinal, count)]
            
    Requirements: 9.2, 9.3, 9.7
    """
    logging.info("Calculating descriptive statistics for report")
    
    # Validate required columns
    required_cols = ['PET_C', 'TSV_ordinal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Missing required columns for statistics: {missing_cols}"
        )
    
    # Extract sample size information
    n_total = removal_stats.get('n_total', len(df))
    n_valid = removal_stats.get('n_retained', len(df))
    pct_valid = removal_stats.get('pct_retained', 100.0)
    
    # Calculate PET statistics
    pet_values = df['PET_C']
    pet_mean = float(pet_values.mean())
    pet_median = float(pet_values.median())
    pet_std = float(pet_values.std())
    pet_min = float(pet_values.min())
    pet_max = float(pet_values.max())
    
    logging.info(f"PET statistics: mean={pet_mean:.2f}, median={pet_median:.2f}, "
                f"std={pet_std:.2f}, range=[{pet_min:.2f}, {pet_max:.2f}]")
    
    # Count observations per sensation category
    sensation_counts = df['TSV_ordinal'].value_counts().to_dict()
    
    # Create formatted distribution list with labels
    sensation_distribution = []
    for ordinal_val in sorted(sensation_counts.keys()):
        ordinal_int = int(ordinal_val)
        label = SENSATION_LABELS.get(ordinal_int, f'Unknown ({ordinal_int})')
        count = sensation_counts[ordinal_val]
        sensation_distribution.append((label, ordinal_int, count))
    
    logging.info(f"Sensation distribution: {len(sensation_counts)} categories, "
                f"{n_valid} total observations")
    
    # Compile statistics dictionary
    stats = {
        'n_total': n_total,
        'n_valid': n_valid,
        'pct_valid': pct_valid,
        'pet_mean': pet_mean,
        'pet_median': pet_median,
        'pet_std': pet_std,
        'pet_min': pet_min,
        'pet_max': pet_max,
        'sensation_counts': sensation_counts,
        'sensation_distribution': sensation_distribution
    }
    
    return stats


def format_model_results_section(params: Dict, cis: Dict) -> str:
    """
    Format model parameters and results as Markdown section.
    
    Creates a formatted Markdown string containing:
    - Model parameters (β, τ_k) with standard errors and confidence intervals
    - Convergence status
    - Model fit statistics (log-likelihood, AIC, BIC)
    
    Args:
        params: Dictionary from extract_model_parameters() containing:
               - 'beta': coefficient value
               - 'beta_se': standard error
               - 'cutpoints': dict of cutpoint values
               - 'cutpoints_se': dict of cutpoint standard errors
               - 'converged': convergence status
               - 'n_obs': number of observations
               - 'log_likelihood': model log-likelihood
               - 'aic': AIC value
               - 'bic': BIC value
        cis: Dictionary from calculate_confidence_intervals() containing:
            - 'beta_ci': tuple (lower, upper)
            - 'cutpoints_ci': dict {k: (lower, upper)}
    
    Returns:
        Markdown formatted string for the model results section
        
    Requirements: 9.6
    """
    logging.info("Formatting model results section")
    
    # Extract values
    beta = params['beta']
    beta_se = params['beta_se']
    beta_ci = cis['beta_ci']
    cutpoints = params['cutpoints']
    cutpoints_se = params['cutpoints_se']
    cutpoints_ci = cis['cutpoints_ci']
    converged = params['converged']
    n_obs = params['n_obs']
    log_likelihood = params['log_likelihood']
    aic = params['aic']
    bic = params['bic']
    
    # Build Markdown section
    md = []
    md.append("## 2. Modelo Ordinal\n")
    md.append("### Parâmetros Estimados\n")
    md.append("**Coeficiente β (PET_C)**:\n")
    md.append(f"- Estimativa: **{beta:.4f}**")
    md.append(f"- Erro padrão: {beta_se:.4f}")
    md.append(f"- IC 95%: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]\n")
    
    md.append("**Limiares (Cutpoints) τ_k**:\n")
    md.append("| Limiar | Estimativa | Erro Padrão | IC 95% |")
    md.append("|--------|------------|-------------|---------|")
    
    for k in sorted(cutpoints.keys()):
        tau = cutpoints[k]
        tau_se = cutpoints_se[k]
        tau_ci = cutpoints_ci[k]
        md.append(f"| τ_{k:+d} | {tau:.4f} | {tau_se:.4f} | [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}] |")
    
    md.append("\n### Qualidade do Ajuste\n")
    md.append(f"- **Convergência**: {'✓ Sim' if converged else '✗ Não'}")
    md.append(f"- **N observações**: {n_obs}")
    md.append(f"- **Log-verossimilhança**: {log_likelihood:.2f}")
    md.append(f"- **AIC**: {aic:.2f}")
    md.append(f"- **BIC**: {bic:.2f}\n")
    
    md.append("### Interpretação do Modelo\n")
    md.append("O modelo de regressão logística ordinal proporcional relaciona o PET ")
    md.append("com a sensação térmica usando a função de ligação logit:\n")
    md.append("```")
    md.append("logit(P(Y ≤ k | PET)) = τ_k - β × PET")
    md.append("```\n")
    md.append(f"O coeficiente β = {beta:.4f} indica que cada aumento de 1°C no PET ")
    
    if beta > 0:
        md.append("está associado a um aumento na probabilidade de sensações mais quentes.")
    else:
        md.append("está associado a uma diminuição na probabilidade de sensações mais quentes.")
    
    md.append("\n")
    
    return "\n".join(md)


def format_comfort_metrics_section(pet_neutral_result: Dict,
                                   comfort_bands: Dict,
                                   category_ranges: Optional[Dict] = None,
                                   observed_ranges: Optional[Dict] = None,
                                   acceptability_bands: Optional[Dict] = None) -> str:
    """
    Format comfort metrics (PET neutral and comfort bands) as Markdown section.
    
    Creates a formatted Markdown string containing:
    - PET neutral value with confidence interval
    - Comfort bands (80% and 90%) with lower and upper limits
    - Optional: Acceptability bands for comparison
    
    Args:
        pet_neutral_result: Dictionary from calculate_pet_neutral() containing:
                           - 'pet_neutral': PET neutral value
                           - 'pet_neutral_ci': tuple (lower, upper)
                           - 'pet_neutral_se': standard error
        comfort_bands: Dictionary from calculate_comfort_bands() containing:
                      - 0.8: {'lower': L_80, 'upper': U_80, 'width': width}
                      - 0.9: {'lower': L_90, 'upper': U_90, 'width': width}
        acceptability_bands: Optional dictionary from calculate_acceptability_bands()
    
    Returns:
        Markdown formatted string for the comfort metrics section
        
    Requirements: 9.4, 9.5
    """
    logging.info("Formatting comfort metrics section")
    
    # Extract PET neutral values
    pet_neutral = pet_neutral_result['pet_neutral']
    pet_neutral_ci = pet_neutral_result['pet_neutral_ci']
    pet_neutral_se = pet_neutral_result['pet_neutral_se']
    
    # Build Markdown section
    md = []
    md.append("## 3. PET Neutro\n")
    md.append(f"**PET Neutro = {pet_neutral:.1f}°C**\n")
    md.append(f"- **Intervalo de Confiança 95%**: [{pet_neutral_ci[0]:.1f}, {pet_neutral_ci[1]:.1f}]°C")
    md.append(f"- **Erro Padrão**: {pet_neutral_se:.2f}°C\n")
    md.append("O PET neutro representa a temperatura equivalente onde a sensação ")
    md.append('"confortável" (categoria 0) é mais provável. Este valor é calculado ')
    md.append("como o ponto médio entre as categorias de conforto no modelo ordinal ")
    md.append("(τ₀ / β).\n")
    
    md.append("## 4. Faixas de Conforto\n")
    md.append("As faixas de conforto representam os intervalos de PET onde a probabilidade ")
    md.append("combinada das categorias centrais (-1: Frio Moderado, 0: Confortável, ")
    md.append("+1: Calor Moderado) atinge os limiares especificados.\n")
    
    # 80% comfort band
    md.append("### Faixa de Conforto 80%\n")
    if 0.8 in comfort_bands and comfort_bands[0.8]['lower'] is not None:
        L_80 = comfort_bands[0.8]['lower']
        U_80 = comfort_bands[0.8]['upper']
        width_80 = comfort_bands[0.8]['width']
        md.append(f"- **Limite Inferior (L₈₀)**: {L_80:.1f}°C")
        md.append(f"- **Limite Superior (U₈₀)**: {U_80:.1f}°C")
        md.append(f"- **Amplitude**: {width_80:.1f}°C")
        md.append(f"- **Intervalo**: [{L_80:.1f}, {U_80:.1f}]°C\n")
        md.append(f"Nesta faixa, pelo menos 80% dos respondentes reportam sensação ")
        md.append("térmica confortável (categorias -1, 0 ou +1).\n")
    else:
        md.append("⚠️ Não foi possível determinar a faixa de 80% com os dados disponíveis.\n")
    
    # 90% comfort band
    md.append("### Faixa de Conforto 90%\n")
    if 0.9 in comfort_bands and comfort_bands[0.9]['lower'] is not None:
        L_90 = comfort_bands[0.9]['lower']
        U_90 = comfort_bands[0.9]['upper']
        width_90 = comfort_bands[0.9]['width']
        md.append(f"- **Limite Inferior (L₉₀)**: {L_90:.1f}°C")
        md.append(f"- **Limite Superior (U₉₀)**: {U_90:.1f}°C")
        md.append(f"- **Amplitude**: {width_90:.1f}°C")
        md.append(f"- **Intervalo**: [{L_90:.1f}, {U_90:.1f}]°C\n")
        md.append(f"Nesta faixa, pelo menos 90% dos respondentes reportam sensação ")
        md.append("térmica confortável (categorias -1, 0 ou +1).\n")
    else:
        md.append("⚠️ Não foi possível determinar a faixa de 90% com os dados disponíveis.\n")
    
    # Category-specific PET ranges
    if category_ranges is not None:
        section_num = 5
        md.append(f"## {section_num}. Faixas de PET por Categoria de Sensação\n")
        md.append("Esta seção apresenta as faixas de PET características para cada categoria ")
        md.append("de sensação térmica, baseadas no modelo probabilístico calibrado.\n\n")
        md.append("Para cada categoria, são apresentadas:\n")
        md.append("- **Faixa Modal**: Intervalo de PET onde esta sensação é a mais provável\n")
        md.append("- **Faixa de Probabilidade ≥30%**: Intervalo onde a probabilidade desta sensação é ≥30%\n")
        md.append("- **Dados Observados**: Estatísticas descritivas do PET quando esta sensação foi reportada\n\n")
        
        # Create table for all categories
        md.append("### Resumo das Faixas de PET\n\n")
        md.append("| Sensação | Faixa Modal (°C) | Pico PET (°C) | PET Observado Médio (°C) | N Obs. |\n")
        md.append("|----------|------------------|---------------|--------------------------|--------|\n")
        
        for k in sorted(category_ranges.keys()):
            cat = category_ranges[k]
            label = cat['label']
            
            # Modal range
            if cat['modal_range']['lower'] is not None:
                modal_str = f"[{cat['modal_range']['lower']:.1f}, {cat['modal_range']['upper']:.1f}]"
                peak_str = f"{cat['modal_range']['peak_pet']:.1f}"
            else:
                modal_str = "—"
                peak_str = "—"
            
            # Observed mean
            if cat['observed'] is not None:
                obs_mean_str = f"{cat['observed']['mean']:.1f}"
                n_obs_str = str(cat['observed']['n'])
            else:
                obs_mean_str = "—"
                n_obs_str = "0"
            
            md.append(f"| {label} ({k:+d}) | {modal_str} | {peak_str} | {obs_mean_str} | {n_obs_str} |\n")
        
        md.append("\n")
        
        # Detailed information for each category
        md.append("### Detalhamento por Categoria\n\n")
        
        for k in sorted(category_ranges.keys()):
            cat = category_ranges[k]
            label = cat['label']
            
            md.append(f"#### {label} ({k:+d})\n\n")
            
            # Modal range
            if cat['modal_range']['lower'] is not None:
                md.append(f"**Faixa Modal**: [{cat['modal_range']['lower']:.1f}, {cat['modal_range']['upper']:.1f}]°C  \n")
                md.append(f"- Pico de probabilidade em {cat['modal_range']['peak_pet']:.1f}°C ")
                md.append(f"(P = {cat['modal_range']['peak_prob']:.1%})  \n")
                md.append(f"- Amplitude: {cat['modal_range']['width']:.1f}°C\n\n")
            else:
                md.append("**Faixa Modal**: Não identificada (sensação nunca é a mais provável)\n\n")
            
            # Threshold range
            if cat['threshold_range']['lower'] is not None:
                md.append(f"**Faixa com P ≥ 30%**: [{cat['threshold_range']['lower']:.1f}, {cat['threshold_range']['upper']:.1f}]°C  \n")
                md.append(f"- Amplitude: {cat['threshold_range']['width']:.1f}°C\n\n")
            else:
                md.append("**Faixa com P ≥ 30%**: Não identificada\n\n")
            
            # Observed statistics
            if cat['observed'] is not None:
                obs = cat['observed']
                md.append(f"**Dados Observados** (n = {obs['n']}):  \n")
                md.append(f"- Média: {obs['mean']:.1f}°C (DP: {obs['std']:.1f}°C)  \n")
                md.append(f"- Mediana: {obs['median']:.1f}°C  \n")
                md.append(f"- Intervalo: [{obs['min']:.1f}, {obs['max']:.1f}]°C  \n")
                md.append(f"- Percentis 25-75: [{obs['percentile_25']:.1f}, {obs['percentile_75']:.1f}]°C\n\n")
            else:
                md.append("**Dados Observados**: Nenhuma observação nesta categoria\n\n")
        
        section_num += 1
    else:
        section_num = 5
    
    # Observed PET ranges section
    if observed_ranges is not None:
        md.append(f"## {section_num}. Faixas de PET Observadas (Análise Descritiva)\n\n")
        md.append("Esta análise apresenta as faixas de PET baseadas diretamente nos dados ")
        md.append("coletados, sem depender de modelagem probabilística.\n\n")
        
        md.append("### Resumo das Faixas Observadas\n\n")
        md.append("| Sensação | N | Média (°C) | Faixa 50% (°C) | Faixa 80% (°C) | Amplitude Total (°C) |\n")
        md.append("|----------|---|------------|----------------|----------------|----------------------|\n")
        
        for k in sorted(observed_ranges.keys()):
            obs = observed_ranges[k]
            label = obs['label']
            n = obs['n']
            mean_val = obs['mean']
            range_50_str = f"[{obs['range_50']['lower']:.1f}, {obs['range_50']['upper']:.1f}]"
            range_80_str = f"[{obs['range_80']['lower']:.1f}, {obs['range_80']['upper']:.1f}]"
            range_full_str = f"[{obs['range_full']['lower']:.1f}, {obs['range_full']['upper']:.1f}]"
            md.append(f"| {label} ({k:+d}) | {n} | {mean_val:.1f} | {range_50_str} | {range_80_str} | {range_full_str} |\n")
        
        md.append("\n")
        
        # Practical recommendations
        if 0 in observed_ranges:
            comfort_obs = observed_ranges[0]
            md.append("### Zona de Conforto Observada\n\n")
            md.append(f"- Faixa Central (50%): [{comfort_obs['range_50']['lower']:.1f}, {comfort_obs['range_50']['upper']:.1f}]°C\n")
            md.append(f"- Faixa Ampla (80%): [{comfort_obs['range_80']['lower']:.1f}, {comfort_obs['range_80']['upper']:.1f}]°C\n")
            md.append(f"- PET médio: {comfort_obs['mean']:.1f}°C\n\n")
        
        # Detailed explanation of ranges with statistical foundation
        md.append("### Interpretação Detalhada das Faixas\n\n")
        
        md.append("As três faixas apresentadas representam diferentes níveis de confiança e abrangência, ")
        md.append("cada uma adequada para aplicações específicas. Todas são baseadas em **estatísticas descritivas robustas** ")
        md.append("calculadas diretamente dos dados observados, sem depender de suposições de distribuição probabilística.\n\n")
        
        # Faixa 50%
        md.append("#### 1. Faixa 50% (Intervalo Interquartil: P25-P75)\n\n")
        md.append("**Definição**: Intervalo entre o percentil 25 (P25) e o percentil 75 (P75), também conhecido como ")
        md.append("Intervalo Interquartil (IQR). Contém os 50% centrais das observações para cada categoria de sensação.\n\n")
        
        md.append("**Fundamentação Estatística**:\n")
        md.append("- Remove automaticamente os 25% mais baixos e 25% mais altos dos dados\n")
        md.append("- Altamente resistente a valores extremos e outliers\n")
        md.append("- Medida robusta de dispersão, amplamente utilizada em análise exploratória de dados\n")
        md.append("- Base para identificação de outliers pela regra de Tukey (IQR × 1.5)\n\n")
        
        md.append("**Por que é confiável?**\n")
        md.append("- **Robustez**: Não é afetada por valores extremos que podem ser erros de medição ou condições atípicas\n")
        md.append("- **Representatividade**: Captura o comportamento típico da maioria das pessoas\n")
        md.append("- **Estabilidade**: Menos sensível a variações amostrais que a média ou desvio padrão\n")
        md.append("- **Validação**: Método padrão em climatologia e estudos de conforto térmico\n\n")
        
        md.append("**Quando usar**:\n")
        md.append("- ✅ **Design urbano e arquitetônico**: Para garantir conforto para a maioria das pessoas\n")
        md.append("- ✅ **Normas e diretrizes**: Quando é necessário estabelecer faixas conservadoras\n")
        md.append("- ✅ **Projetos com alta exigência de conforto**: Espaços públicos, áreas de permanência\n")
        md.append("- ✅ **Comparação entre locais**: Faixa mais estável para comparações científicas\n\n")
        
        # Faixa 80%
        md.append("#### 2. Faixa 80% (P10-P90)\n\n")
        md.append("**Definição**: Intervalo entre o percentil 10 (P10) e o percentil 90 (P90). ")
        md.append("Contém 80% das observações centrais, excluindo apenas os 10% mais extremos de cada lado.\n\n")
        
        md.append("**Fundamentação Estatística**:\n")
        md.append("- Equilibra abrangência e robustez, incluindo variabilidade natural sem extremos\n")
        md.append("- Percentis P10 e P90 são pontos de corte comuns em análises climáticas\n")
        md.append("- Mantém resistência razoável a outliers enquanto captura maior variabilidade\n")
        md.append("- Aproxima-se de ±1.28 desvios padrão em distribuições normais\n\n")
        
        md.append("**Por que é confiável?**\n")
        md.append("- **Realismo**: Reflete a variabilidade natural do conforto térmico em condições reais\n")
        md.append("- **Abrangência**: Cobre a grande maioria dos casos sem incluir extremos raros\n")
        md.append("- **Aplicabilidade**: Útil para entender a amplitude esperada do fenômeno\n")
        md.append("- **Contexto climático**: Alinha-se com análises de variabilidade climática (decis)\n\n")
        
        md.append("**Quando usar**:\n")
        md.append("- ✅ **Análise de variabilidade**: Para entender a amplitude real do conforto térmico\n")
        md.append("- ✅ **Planejamento adaptativo**: Quando é necessário considerar maior diversidade de condições\n")
        md.append("- ✅ **Estudos de adaptação**: Para avaliar a capacidade de adaptação da população\n")
        md.append("- ✅ **Contexto de pesquisa**: Apresentar a variabilidade completa sem extremos\n\n")
        
        # Amplitude Total
        md.append("#### 3. Amplitude Total (Min-Max)\n\n")
        md.append("**Definição**: Intervalo completo dos dados observados, do valor mínimo absoluto ao valor máximo absoluto. ")
        md.append("Representa 100% das observações coletadas na pesquisa.\n\n")
        
        md.append("**Fundamentação Estatística**:\n")
        md.append("- Medida de dispersão mais simples e direta: Range = Max - Min\n")
        md.append("- Não faz suposições sobre a distribuição dos dados\n")
        md.append("- Sensível a todos os valores, incluindo outliers e casos extremos\n")
        md.append("- Aumenta com o tamanho da amostra (mais dados = maior chance de extremos)\n\n")
        
        md.append("**Por que é confiável?**\n")
        md.append("- **Completude**: Mostra os limites absolutos observados na pesquisa\n")
        md.append("- **Transparência**: Não oculta nenhum dado, apresenta a realidade completa\n")
        md.append("- **Contexto**: Essencial para identificar condições extremas que realmente ocorreram\n")
        md.append("- **Validação**: Permite verificar se há valores implausíveis ou erros de medição\n\n")
        
        md.append("**Quando usar**:\n")
        md.append("- ✅ **Identificação de extremos**: Para conhecer os limites absolutos observados\n")
        md.append("- ✅ **Análise de casos especiais**: Quando extremos são relevantes (ondas de calor/frio)\n")
        md.append("- ✅ **Contexto completo**: Para apresentar toda a amplitude de condições encontradas\n")
        md.append("- ✅ **Validação de dados**: Verificar se há valores fora do esperado\n\n")
        
        md.append("**⚠️ Atenção**: A amplitude total é sensível a outliers e aumenta com o tamanho da amostra. ")
        md.append("Valores extremos podem representar condições raras ou erros de medição. Use com cautela para design.\n\n")
        
        # Comparison and recommendations
        md.append("### Comparação e Recomendações de Uso\n\n")
        md.append("| Faixa | Abrangência | Robustez | Melhor Aplicação |\n")
        md.append("|-------|-------------|----------|------------------|\n")
        md.append("| **50% (IQR)** | 50% central | ⭐⭐⭐⭐⭐ Muito alta | Design urbano, normas |\n")
        md.append("| **80% (P10-P90)** | 80% central | ⭐⭐⭐⭐ Alta | Análise de variabilidade |\n")
        md.append("| **Total (Min-Max)** | 100% completo | ⭐⭐ Moderada | Contexto, extremos |\n\n")
        
        md.append("**Recomendação Geral**: Para a maioria dos projetos de design urbano e arquitetônico, ")
        md.append("recomenda-se usar a **Faixa 50%** como referência principal, consultando a **Faixa 80%** ")
        md.append("para entender a variabilidade esperada e a **Amplitude Total** para contexto completo.\n\n")
        
        
        # New section: Single Recommended Range
        md.append("### Faixa Única Recomendada para Cada Sensação\n\n")
        
        md.append("Para facilitar a aplicação prática dos resultados, apresentamos abaixo uma **faixa única** ")
        md.append("para cada categoria de sensação térmica, baseada no **Intervalo Interquartil (IQR)**, ")
        md.append("que corresponde à Faixa 50% (P25-P75) apresentada anteriormente.\n\n")
        # Methodology explanation
        md.append("#### Metodologia: Por que usar o Intervalo Interquartil (IQR)?\n\n")
        
        md.append("**Contexto**: Em pesquisas de percepção térmica com entrevistas, os dados apresentam ")
        md.append("características específicas que exigem métodos estatísticos robustos:\n\n")
        
        md.append("1. **Alta Variabilidade Individual**: Pessoas têm metabolismos, vestimentas e níveis de ")
        md.append("aclimatação diferentes, resultando em percepções térmicas variadas para o mesmo PET.\n\n")
        
        md.append("2. **Presença de Outliers**: Sempre existem respostas atípicas em pesquisas (erros de ")
        md.append("resposta, condições de saúde específicas, aclimatação extrema).\n\n")
        
        md.append("3. **Distribuição Não-Normal**: A percepção térmica humana raramente segue uma distribuição ")
        md.append("normal, tornando inadequados métodos baseados em média e desvio padrão.\n\n")
        
        md.append("**Solução: Intervalo Interquartil (IQR)**\n\n")
        
        md.append("O IQR é definido como o intervalo entre o percentil 25 (P25) e o percentil 75 (P75), ")
        md.append("contendo os **50% centrais** das observações. Esta é a escolha ideal porque:\n\n")
        
        md.append("✅ **Robustez**: Remove automaticamente os 25% mais extremos de cada lado, eliminando ")
        md.append("outliers sem perder informação relevante\n\n")
        
        md.append("✅ **Não-paramétrico**: Não assume distribuição normal, adequado para dados de percepção humana\n\n")
        
        md.append("✅ **Representatividade**: Captura o comportamento típico da maioria das pessoas, ")
        md.append("não casos extremos\n\n")
        
        md.append("✅ **Validação Científica**: Método padrão em normas internacionais (ISO 7730, ASHRAE 55) ")
        md.append("e amplamente usado em estudos de conforto térmico\n\n")
        
        md.append("✅ **Estabilidade**: Menos sensível a variações amostrais que média ou amplitude total\n\n")
        
        md.append("✅ **Aplicabilidade**: Ideal para design urbano e arquitetônico, onde se busca garantir ")
        md.append("conforto para a maioria das pessoas\n\n")
        # Table with single recommended range
        md.append("#### Tabela de Faixas Únicas Recomendadas\n\n")
        md.append("| Sensação | N | Faixa Recomendada (°C) | Amplitude (°C) | PET Médio (°C) |\n")
        md.append("|----------|---|------------------------|----------------|----------------|\n")
        
        for k in sorted(observed_ranges.keys()):
            obs = observed_ranges[k]
            label = obs["label"]
            n = obs["n"]
            lower = obs["range_50"]["lower"]
            upper = obs["range_50"]["upper"]
            width = obs["range_50"]["width"]
            mean_val = obs["mean"]
            md.append(f"| {label} ({k:+d}) | {n} | [{lower:.1f}, {upper:.1f}] | {width:.1f} | {mean_val:.1f} |\n")
        
        md.append("\n")
        
        # Interpretation
        md.append("#### Interpretação da Tabela\n\n")
        
        md.append("**Faixa Recomendada**: Intervalo de PET onde 50% das pessoas reportaram aquela sensação térmica. ")
        md.append("Esta é a faixa mais confiável para uso em projetos de design urbano e arquitetônico.\n\n")
        
        md.append("**Amplitude**: Largura da faixa em graus Celsius. Amplitudes menores indicam maior consenso ")
        md.append("entre as pessoas sobre aquela sensação térmica.\n\n")
        
        md.append("**PET Médio**: Valor central de PET para aquela sensação. Útil como referência rápida.\n\n")
        
        # Practical recommendations
        md.append("#### Como Usar Estas Faixas\n\n")
        
        md.append("**Para Design Urbano e Arquitetônico**:\n\n")
        
        if 0 in observed_ranges:
            comfort_obs = observed_ranges[0]
            comfort_lower = comfort_obs["range_50"]["lower"]
            comfort_upper = comfort_obs["range_50"]["upper"]
            comfort_mean = comfort_obs["mean"]
            
            md.append(f"1. **Zona de Conforto Térmico**: Mantenha o PET entre **{comfort_lower:.1f}°C e {comfort_upper:.1f}°C** ")
            md.append(f"para garantir que a maioria das pessoas se sinta confortável.\n\n")
            
            md.append(f"2. **Valor de Referência**: Use **{comfort_mean:.1f}°C** como PET ideal para conforto térmico.\n\n")
        
        md.append("3. **Evitar Desconforto**: Identifique as faixas de sensações indesejadas (muito frio/quente) ")
        md.append("e projete para evitar que o PET atinja esses valores.\n\n")
        
        md.append("4. **Estratégias de Mitigação**: Para cada faixa de desconforto identificada, desenvolva ")
        md.append("estratégias específicas (sombreamento, ventilação, aquecimento).\n\n")
        # Analysis of overlaps
        md.append("#### Análise de Sobreposição entre Categorias\n\n")
        
        md.append("É importante notar que as faixas de diferentes sensações podem se sobrepor. Isso é **esperado e natural** ")
        md.append("em dados de percepção humana, pois:\n\n")
        
        md.append("- Pessoas têm diferentes níveis de sensibilidade térmica\n")
        md.append("- A aclimatação local influencia a percepção\n")
        md.append("- Fatores individuais (idade, metabolismo, vestimenta) afetam o conforto\n\n")
        
        md.append("**Sobreposições observadas**:\n\n")
        
        # Calculate overlaps
        sorted_keys = sorted(observed_ranges.keys())
        for i in range(len(sorted_keys) - 1):
            k1 = sorted_keys[i]
            k2 = sorted_keys[i + 1]
            obs1 = observed_ranges[k1]
            obs2 = observed_ranges[k2]
            upper1 = obs1["range_50"]["upper"]
            lower2 = obs2["range_50"]["lower"]
            if upper1 > lower2:
                overlap = upper1 - lower2
                label1 = obs1["label"]
                label2 = obs2["label"]
                md.append(f"- **{label1}** e **{label2}**: Sobreposição de {overlap:.1f}°C (entre {lower2:.1f}°C e {upper1:.1f}°C)\n")
            else:
                gap = lower2 - upper1
                label1 = obs1["label"]
                label2 = obs2["label"]
                md.append(f"- **{label1}** e **{label2}**: Sem sobreposição (gap de {gap:.1f}°C)\n")
        
        md.append("\n")
        md.append("**Implicação Prática**: Em zonas de sobreposição, diferentes pessoas podem ter percepções diferentes. ")
        md.append("Para design, priorize manter o PET dentro da faixa de conforto.\n\n")
        
        # Scientific validation
        md.append("#### Validação Científica\n\n")
        md.append("O método do Intervalo Interquartil (IQR) é:\n\n")
        md.append("✅ **ISO 7730**: Norma internacional para ambientes térmicos\n\n")
        md.append("✅ **ASHRAE 55**: Padrão americano para conforto térmico\n\n")
        md.append("✅ **Literatura**: Nikolopoulou & Lykoudis (2006), Matzarakis et al. (1999)\n\n")
        
        if 0 in observed_ranges:
            comfort_obs = observed_ranges[0]
            comfort_lower = comfort_obs["range_50"]["lower"]
            comfort_upper = comfort_obs["range_50"]["upper"]
            comfort_mean = comfort_obs["mean"]
            md.append(f"**Seus dados**: Conforto em [{comfort_lower:.1f}, {comfort_upper:.1f}]°C (média: {comfort_mean:.1f}°C)\n\n")
            md.append("💡 **Dica**: Diferenças em relação à literatura indicam adaptação climática local!\n\n")
        
        
        
        
        md.append("💡 **Nota**: Faixas baseadas exclusivamente nos dados observados.\n\n")
        section_num += 1
    
        # Optional: Acceptability bands
    if acceptability_bands is not None:
        md.append(f"## {section_num}. Faixas de Aceitabilidade (Análise Complementar)\n")
        md.append("As faixas de aceitabilidade são baseadas em um modelo logístico binário ")
        md.append("separado e fornecem uma perspectiva complementar sobre o conforto térmico.\n")
        
        # 80% acceptability band
        if 0.8 in acceptability_bands and acceptability_bands[0.8]['lower'] is not None:
            L_80_acc = acceptability_bands[0.8]['lower']
            U_80_acc = acceptability_bands[0.8]['upper']
            width_80_acc = acceptability_bands[0.8]['width']
            md.append(f"### Faixa de Aceitabilidade 80%\n")
            md.append(f"- **Intervalo**: [{L_80_acc:.1f}, {U_80_acc:.1f}]°C")
            md.append(f"- **Amplitude**: {width_80_acc:.1f}°C\n")
        
        # 90% acceptability band
        if 0.9 in acceptability_bands and acceptability_bands[0.9]['lower'] is not None:
            L_90_acc = acceptability_bands[0.9]['lower']
            U_90_acc = acceptability_bands[0.9]['upper']
            width_90_acc = acceptability_bands[0.9]['width']
            md.append(f"### Faixa de Aceitabilidade 90%\n")
            md.append(f"- **Intervalo**: [{L_90_acc:.1f}, {U_90_acc:.1f}]°C")
            md.append(f"- **Amplitude**: {width_90_acc:.1f}°C\n")
    
    return "\n".join(md)


def generate_markdown_report(stats: Dict,
                            params: Dict,
                            cis: Dict,
                            pet_neutral_result: Dict,
                            comfort_bands: Dict,
                            output_path: str,
                            category_ranges: Optional[Dict] = None,
                            observed_ranges: Optional[Dict] = None,
                            acceptability_bands: Optional[Dict] = None,
                            plot_files: Optional[Dict] = None,
                            filename: str = 'RELATORIO_PET.md') -> str:
    """
    Generate complete analysis report in Markdown format.
    
    Creates a comprehensive report with:
    - Header with author credits (Carol Freire do Santos, USP)
    - Data summary section (N, %, descriptive statistics)
    - Model results section (parameters, fit statistics)
    - PET neutral section (value and CI)
    - Comfort bands section (80% and 90% limits)
    - Optional: Acceptability section (if data available)
    - Visualizations section (embedded image references)
    - Interpretation guidelines
    - References section
    
    Args:
        stats: Dictionary from format_descriptive_statistics()
        params: Dictionary from extract_model_parameters()
        cis: Dictionary from calculate_confidence_intervals()
        pet_neutral_result: Dictionary from calculate_pet_neutral()
        comfort_bands: Dictionary from calculate_comfort_bands()
        output_path: Directory path for output file
        acceptability_bands: Optional dictionary from calculate_acceptability_bands()
        plot_files: Optional dictionary with paths to plot files:
                   - 'scatter': path to scatter plot
                   - 'probability': path to probability curves plot
                   - 'comfort_zone': path to comfort zone plot
        filename: Name of output file (default: 'RELATORIO_PET.md')
    
    Returns:
        Full path to saved report file
        
    Raises:
        IOError: If report cannot be written
        
    Requirements: 9.1, 9.8, 9.9
    """
    logging.info(f"Generating Markdown report")
    
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(output_path, filename)
    
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build report content
    report = []
    
    # ========== HEADER ==========
    report.append("# Relatório de Calibração PET - Conforto Térmico\n")
    report.append("**Autora**: Carol Freire do Santos  ")
    report.append("**Instituição**: Universidade de São Paulo (USP)  ")
    report.append("**Programa**: Doutorado em Climatologia  ")
    report.append(f"**Data de Geração**: {timestamp}\n")
    report.append("---\n")
    
    # ========== DATA SUMMARY ==========
    report.append("## 1. Resumo dos Dados\n")
    report.append(f"- **Total de respostas**: {stats['n_total']}")
    report.append(f"- **Respostas válidas**: {stats['n_valid']} ({stats['pct_valid']:.1f}%)")
    report.append(f"- **PET médio**: {stats['pet_mean']:.1f}°C (DP: {stats['pet_std']:.1f}°C)")
    report.append(f"- **PET mediana**: {stats['pet_median']:.1f}°C")
    report.append(f"- **Intervalo PET**: [{stats['pet_min']:.1f}, {stats['pet_max']:.1f}]°C\n")
    
    report.append("### Distribuição de Sensação Térmica\n")
    report.append("| Categoria | Valor Ordinal | N Respostas | % |")
    report.append("|-----------|---------------|-------------|---|")
    
    for label, ordinal, count in stats['sensation_distribution']:
        pct = (count / stats['n_valid'] * 100) if stats['n_valid'] > 0 else 0
        report.append(f"| {label} | {ordinal:+d} | {count} | {pct:.1f}% |")
    
    report.append("\n")
    
    # ========== MODEL RESULTS ==========
    model_section = format_model_results_section(params, cis)
    report.append(model_section)
    
    # ========== COMFORT METRICS ==========
    comfort_section = format_comfort_metrics_section(
        pet_neutral_result, 
        comfort_bands,
        category_ranges,
        observed_ranges,
        acceptability_bands
    )
    report.append(comfort_section)
    
    # ========== VISUALIZATIONS ==========
    if plot_files:
        report.append("## 6. Visualizações\n")
        
        if 'scatter' in plot_files:
            scatter_filename = os.path.basename(plot_files['scatter'])
            report.append("### Relação PET vs Sensação Térmica\n")
            report.append(f"![Scatter TSV vs PET]({scatter_filename})\n")
            report.append("Gráfico de dispersão mostrando a relação entre PET e sensação ")
            report.append("térmica ordinal. A linha vertical vermelha indica o PET neutro.\n")
        
        if 'probability' in plot_files:
            prob_filename = os.path.basename(plot_files['probability'])
            report.append("### Curvas de Probabilidade por Categoria\n")
            report.append(f"![Curvas de Probabilidade]({prob_filename})\n")
            report.append("Probabilidades de cada categoria de sensação térmica em função ")
            report.append("do PET. As regiões sombreadas indicam as faixas de conforto ")
            report.append("(80% e 90%).\n")
        
        if 'comfort_zone' in plot_files:
            comfort_filename = os.path.basename(plot_files['comfort_zone'])
            report.append("### Zona de Conforto Térmico\n")
            report.append(f"![Zona de Conforto]({comfort_filename})\n")
            report.append("Probabilidade de conforto (P(-1 ≤ Y ≤ +1)) em função do PET. ")
            report.append("As linhas horizontais indicam os limiares de 80% e 90%, e as ")
            report.append("linhas verticais marcam os limites das faixas de conforto.\n")
    
    # ========== INTERPRETATION GUIDELINES ==========
    report.append("## 7. Interpretação dos Resultados\n")
    report.append("### Como usar as faixas de conforto\n")
    report.append("1. **Faixa 80%**: Recomendada para aplicações gerais de planejamento ")
    report.append("   urbano e design de espaços externos. Garante que a maioria das ")
    report.append("   pessoas (80%) se sentirá confortável.\n")
    report.append("2. **Faixa 90%**: Recomendada para espaços que requerem maior rigor ")
    report.append("   de conforto, como áreas de permanência prolongada ou populações ")
    report.append("   sensíveis.\n")
    report.append("3. **PET Neutro**: Representa a temperatura ideal de conforto térmico ")
    report.append("   para a população estudada. Pode ser usado como referência para ")
    report.append("   estratégias de mitigação térmica.\n")
    
    report.append("### Limitações e considerações\n")
    report.append("- Os resultados são específicos para a população e contexto climático ")
    report.append("  estudados. Extrapolações para outras regiões devem ser feitas com cautela.\n")
    report.append("- O modelo assume proporcionalidade dos odds (proportional odds assumption). ")
    report.append("  Violações desta suposição podem afetar a precisão das estimativas.\n")
    report.append(f"- O tamanho amostral (N = {stats['n_valid']}) influencia a precisão ")
    report.append("  dos intervalos de confiança. Amostras maiores produzem estimativas ")
    report.append("  mais precisas.\n")
    
    # ========== REFERENCES ==========
    report.append("## 8. Referências\n")
    report.append("### Metodologia Estatística\n")
    report.append("- **McCullagh, P.** (1980). Regression Models for Ordinal Data. ")
    report.append("  *Journal of the Royal Statistical Society: Series B*, 42(2), 109-127.\n")
    report.append("- **Agresti, A.** (2010). *Analysis of Ordinal Categorical Data* ")
    report.append("  (2nd ed.). Wiley.\n")
    
    report.append("### Índice PET\n")
    report.append("- **Höppe, P.** (1999). The physiological equivalent temperature - ")
    report.append("  a universal index for the biometeorological assessment of the thermal ")
    report.append("  environment. *International Journal of Biometeorology*, 43(2), 71-75.\n")
    report.append("- **Matzarakis, A., Mayer, H., & Iziomon, M. G.** (1999). Applications ")
    report.append("  of a universal thermal index: physiological equivalent temperature. ")
    report.append("  *International Journal of Biometeorology*, 43(2), 76-84.\n")
    
    report.append("### Conforto Térmico\n")
    report.append("- **ASHRAE** (2020). *ASHRAE Standard 55: Thermal Environmental ")
    report.append("  Conditions for Human Occupancy*. American Society of Heating, ")
    report.append("  Refrigerating and Air-Conditioning Engineers.\n")
    report.append("- **ISO 7730** (2005). *Ergonomics of the thermal environment - ")
    report.append("  Analytical determination and interpretation of thermal comfort using ")
    report.append("  calculation of the PMV and PPD indices and local thermal comfort criteria*. ")
    report.append("  International Organization for Standardization.\n")
    
    report.append("---\n")
    report.append("*Relatório gerado automaticamente pelo PET Thermal Comfort Calibrator*  ")
    report.append("*Desenvolvido por Carol Freire do Santos - Doutorado em Climatologia, USP*\n")
    
    # Write report to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logging.info(f"Successfully saved report to {file_path}")
        logging.info(f"Report contains {len(report)} lines")
        
        return file_path
        
    except Exception as e:
        raise IOError(f"Error saving report to {file_path}: {e}")


# ============================================================================
# CLI AND MAIN EXECUTION FLOW
# ============================================================================

def convert_markdown_to_pdf(md_file: str, pdf_file: str = None) -> Optional[str]:
    """
    Convert Markdown report to PDF.
    
    Tries multiple methods in order:
    1. weasyprint (if installed) - simpler, no LaTeX needed
    2. pandoc with various engines (if installed)
    
    Args:
        md_file: Path to input Markdown file
        pdf_file: Path to output PDF file (optional, defaults to same name with .pdf)
    
    Returns:
        Path to generated PDF file if successful, None otherwise
        
    Requirements: New feature for PDF export
    """
    import subprocess
    import shutil
    
    # Determine output PDF path
    if pdf_file is None:
        pdf_file = md_file.replace('.md', '.pdf')
    
    logging.info(f"Converting Markdown to PDF: {md_file} → {pdf_file}")
    
    # Method 1: Try reportlab (simple, no external dependencies)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        logging.info("Generating PDF with reportlab...")
        
        # Read markdown file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Create PDF
        doc = SimpleDocTemplate(
            pdf_file,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for PDF elements
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10
        )
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=8
        )
        normal_style = styles['Normal']
        
        # Parse markdown and convert to PDF elements
        lines = md_content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                story.append(Spacer(1, 0.1*inch))
            elif line.startswith('# '):
                story.append(Paragraph(line[2:], title_style))
            elif line.startswith('## '):
                story.append(Spacer(1, 0.2*inch))
                story.append(Paragraph(line[3:], heading1_style))
            elif line.startswith('### '):
                story.append(Paragraph(line[4:], heading2_style))
            elif line.startswith('| ') and '|' in line:
                # Table - collect all table rows
                table_data = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    row = [cell.strip() for cell in lines[i].strip().split('|')[1:-1]]
                    if not all(cell.replace('-', '').strip() == '' for cell in row):  # Skip separator rows
                        table_data.append(row)
                    i += 1
                i -= 1
                
                if table_data:
                    t = Table(table_data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 0.2*inch))
            elif line.startswith('**') and line.endswith('**'):
                story.append(Paragraph(f"<b>{line[2:-2]}</b>", normal_style))
            elif line.startswith('- '):
                story.append(Paragraph(f"• {line[2:]}", normal_style))
            else:
                if line:
                    story.append(Paragraph(line, normal_style))
            
            i += 1
        
        # Build PDF
        doc.build(story)
        
        logging.info(f"✓ PDF gerado com sucesso: {pdf_file}")
        return pdf_file
        
    except ImportError as e:
        logging.warning(f"reportlab não disponível: {e}")
    except Exception as e:
        logging.error(f"Erro ao gerar PDF com reportlab: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    # Method 2: Try pandoc with LaTeX (if available)
    if not shutil.which('pandoc'):
        logging.info("  pandoc não instalado - PDF direto não disponível")
        return html_file if 'html_file' in locals() else None
    
    logging.debug("Trying pandoc method...")
    
    # Try different PDF engines in order of preference
    pdf_engines = ['xelatex', 'pdflatex']
    
    for engine in pdf_engines:
        if not shutil.which(engine):
            continue
            
        cmd = ['pandoc', md_file, '-o', pdf_file, '--pdf-engine', engine]
        
        # Add formatting options
        cmd.extend([
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '--toc',
            '--toc-depth=2',
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=blue',
            '-V', 'urlcolor=blue'
        ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logging.info(f"✓ PDF gerado com sucesso: {pdf_file}")
                logging.debug(f"PDF engine usado: {engine}")
                return pdf_file
                
        except subprocess.TimeoutExpired:
            logging.warning("Timeout ao gerar PDF (>60s)")
            continue
        except Exception as e:
            logging.debug(f"Erro com {engine}: {e}")
            continue
    
    # If we got here, PDF generation failed but HTML might be available
    logging.info("  LaTeX não instalado - use o arquivo HTML para gerar PDF no navegador")
    return html_file if 'html_file' in locals() else None


def parse_arguments():
    """
    Parse command-line arguments for the PET calibrator.
    
    Defines CLI interface with required and optional arguments:
    - --input: Path to input CSV or Excel file (required)
    - --out: Output directory for results (required)
    - --map: Path to JSON column mapping file (optional)
    - --verbose: Enable verbose logging (optional flag)
    
    Returns:
        argparse.Namespace: Parsed arguments object with attributes:
            - input: str - Path to input file
            - out: str - Path to output directory
            - map: str or None - Path to mapping file
            - verbose: bool - Verbose flag
    
    Examples:
        Basic usage:
            python pet_calibrator.py --input data.csv --out results
        
        With column mapping:
            python pet_calibrator.py --input data.xlsx --out results --map mapping.json
        
        With verbose logging:
            python pet_calibrator.py --input data.csv --out results --verbose
    
    Requirements: 10.1-10.3, 10.5, 10.8
    """
    parser = argparse.ArgumentParser(
        description='PET Thermal Comfort Calibrator - Calibração de conforto térmico usando modelagem ordinal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de Uso:
  # Uso básico com arquivo CSV
  python pet_calibrator.py --input dados.csv --out resultados
  
  # Com arquivo Excel e mapeamento de colunas
  python pet_calibrator.py --input dados.xlsx --out resultados --map mapeamento.json
  
  # Com logging detalhado
  python pet_calibrator.py --input dados.csv --out resultados --verbose

Autora: Carol Freire do Santos
Instituição: Universidade de São Paulo (USP)
Programa: Doutorado em Climatologia

Para mais informações, consulte o README.md ou a documentação completa.
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input',
        required=True,
        type=str,
        metavar='FILE',
        help='Caminho para o arquivo de entrada (CSV ou Excel). '
             'Deve conter colunas de PET e sensação térmica.'
    )
    
    parser.add_argument(
        '--out',
        required=True,
        type=str,
        metavar='DIR',
        help='Diretório de saída para os resultados. '
             'Será criado se não existir.'
    )
    
    # Optional arguments
    parser.add_argument(
        '--map',
        required=False,
        type=str,
        metavar='FILE',
        default=None,
        help='Caminho para arquivo JSON de mapeamento de colunas (opcional). '
             'Use quando os nomes das colunas no arquivo de entrada '
             'diferem dos nomes esperados (PET_C, Sensation).'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Ativar logging detalhado (nível DEBUG). '
             'Por padrão, usa nível INFO.'
    )
    
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Gerar relatório em PDF além do Markdown. '
             'Requer pandoc instalado no sistema.'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    return args


def main():
    """
    Main execution function for PET Thermal Comfort Calibrator.
    
    Orchestrates the complete analysis pipeline:
    1. Parse CLI arguments
    2. Setup logging
    3. Create output directory
    4. Load data with error handling
    5. Apply column mapping if provided
    6. Validate required columns
    7. Clean and prepare data
    8. Check minimum sample size
    9. Save cleaned data
    10. Fit ordinal model
    11. Calculate comfort metrics
    12. Analyze acceptability (if available)
    13. Generate all visualizations
    14. Generate report
    15. Display success message with key results
    
    Handles all exceptions with appropriate error messages and exit codes.
    
    Exit codes:
        0: Success
        1: Error (data loading, validation, model fitting, etc.)
    
    Requirements: 10.4, 10.6, 10.7, 11.1-11.7
    """
    # Initialize variables for cleanup
    output_path = None
    
    try:
        # ========== STEP 1: Parse CLI arguments ==========
        args = parse_arguments()
        
        # ========== STEP 2: Setup logging ==========
        output_path = args.out
        log_file = os.path.join(output_path, 'pet_calibrator.log') if output_path else 'pet_calibrator.log'
        
        # Create output directory first (needed for log file)
        if output_path:
            os.makedirs(output_path, exist_ok=True)
        
        setup_logging(verbose=args.verbose, log_file=log_file)
        
        # Log startup information
        logging.info("=" * 70)
        logging.info("PET THERMAL COMFORT CALIBRATOR")
        logging.info("Autora: Carol Freire do Santos")
        logging.info("Instituição: Universidade de São Paulo (USP)")
        logging.info("Programa: Doutorado em Climatologia")
        logging.info("=" * 70)
        logging.info(f"Input file: {args.input}")
        logging.info(f"Output directory: {args.out}")
        if args.map:
            logging.info(f"Column mapping file: {args.map}")
        logging.info(f"Verbose mode: {args.verbose}")
        logging.info("=" * 70)
        
        # ========== STEP 3: Load data ==========
        logging.info("\n[STEP 1/10] Loading input data...")
        df = load_data(args.input)
        logging.info(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # ========== STEP 4: Apply column mapping if provided ==========
        if args.map:
            logging.info("\n[STEP 2/10] Applying column mapping...")
            mapping = load_column_mapping(args.map)
            df = apply_column_mapping(df, mapping)
            logging.info("✓ Column mapping applied")
        else:
            logging.info("\n[STEP 2/10] No column mapping provided, using original column names")
        
        # ========== STEP 5: Validate required columns ==========
        logging.info("\n[STEP 3/10] Validating required columns...")
        is_valid, missing_cols = validate_required_columns(df)
        
        if not is_valid:
            raise ValidationError(
                f"Missing required columns: {missing_cols}\n"
                f"Required columns: PET_C (PET values), Sensation (thermal sensation)\n"
                f"Available columns: {list(df.columns)}\n"
                f"Tip: Use --map to specify a column mapping file if your columns have different names."
            )
        
        logging.info("✓ All required columns present")
        
        # ========== STEP 6: Clean and prepare data ==========
        logging.info("\n[STEP 4/10] Cleaning and preparing data...")
        
        # Map sensation to ordinal
        df = map_sensation_to_ordinal(df)
        
        # Validate PET values
        df = validate_pet_values(df)
        
        # Remove invalid rows
        df_clean, removal_stats = remove_invalid_rows(df)
        
        logging.info(f"✓ Data cleaning complete: {removal_stats['n_retained']} valid rows retained")
        
        # ========== STEP 7: Check minimum sample size ==========
        n_valid = removal_stats['n_retained']
        
        if n_valid < MIN_SAMPLE_SIZE:
            raise InsufficientDataError(
                f"Insufficient valid data: {n_valid} observations.\n"
                f"Minimum required: {MIN_SAMPLE_SIZE} observations.\n"
                f"Please check your data quality or collect more responses."
            )
        
        if n_valid < 50:
            logging.warning(
                f"Sample size ({n_valid}) is below recommended minimum (50). "
                f"Model results may be less stable."
            )
        
        # ========== STEP 8: Save cleaned data ==========
        logging.info("\n[STEP 5/10] Saving cleaned data...")
        cleaned_file = save_cleaned_data(df_clean, output_path)
        logging.info(f"✓ Cleaned data saved to {cleaned_file}")
        
        # ========== STEP 9: Fit ordinal model ==========
        logging.info("\n[STEP 6/10] Fitting ordinal regression model...")
        model_result = fit_ordered_model(df_clean)
        logging.info("✓ Model fitted successfully")
        
        # Extract parameters
        params = extract_model_parameters(model_result)
        cis = calculate_confidence_intervals(params)
        
        # ========== STEP 10: Calculate comfort metrics ==========
        logging.info("\n[STEP 7/10] Calculating comfort metrics...")
        
        # Calculate PET neutral
        # Check if cutpoint 0 exists (may not exist with highly imbalanced data)
        if 0 not in params['cutpoints']:
            logging.warning(
                "Cutpoint 0 (between 'confortável' and 'calor moderado') not found in model. "
                "This typically occurs with highly imbalanced data where some categories are missing. "
                "PET neutral calculation will be skipped."
            )
            # Use a fallback: find the closest available cutpoint
            available_cutpoints = sorted(params['cutpoints'].keys())
            if available_cutpoints:
                closest_cutpoint = min(available_cutpoints, key=lambda x: abs(x - 0))
                logging.info(f"Using closest available cutpoint: {closest_cutpoint}")
                pet_neutral_result = calculate_pet_neutral(
                    beta=params['beta'],
                    tau_0=params['cutpoints'][closest_cutpoint],
                    beta_se=params['beta_se'],
                    tau_0_se=params['cutpoints_se'][closest_cutpoint]
                )
                pet_neutral_result['note'] = f"Calculated using cutpoint {closest_cutpoint} (cutpoint 0 not available)"
            else:
                raise ModelError("No cutpoints available in model. Cannot calculate PET neutral.")
        else:
            pet_neutral_result = calculate_pet_neutral(
                beta=params['beta'],
                tau_0=params['cutpoints'][0],
                beta_se=params['beta_se'],
                tau_0_se=params['cutpoints_se'][0]
            )
        
        # Calculate probabilities
        prob_results = calculate_probabilities(
            beta=params['beta'],
            cutpoints=params['cutpoints']
        )
        
        # Calculate comfort bands
        comfort_bands = calculate_comfort_bands(
            pet_grid=prob_results['pet_grid'],
            prob_comfort=prob_results['prob_comfort']
        )
        
        # Calculate category-specific PET ranges
        category_ranges = calculate_category_pet_ranges(
            pet_grid=prob_results['pet_grid'],
            prob_category=prob_results['prob_category'],
            df=df_clean,
            threshold=0.3  # 30% probability threshold
        )
        
        # Calculate observed PET ranges (data-based, more reliable)
        logging.info(">>> CALCULATING OBSERVED PET RANGES <<<")
        observed_ranges = calculate_observed_pet_ranges(df=df_clean)
        logging.info(f">>> OBSERVED RANGES: {len(observed_ranges)} categories <<<")
        
        logging.info("✓ Comfort metrics calculated")
        
        # ========== STEP 11: Analyze acceptability (optional) ==========
        logging.info("\n[STEP 8/10] Analyzing acceptability (optional)...")
        acceptability_bands = None
        
        is_available, acceptability_binary = check_acceptability_column(df_clean)
        
        if is_available:
            logging.info("Acceptability data found, fitting model...")
            acceptability_model = fit_acceptability_model(df_clean, acceptability_binary)
            
            if acceptability_model is not None:
                acceptability_bands = calculate_acceptability_bands(
                    model_results=acceptability_model,
                    pet_grid=prob_results['pet_grid']
                )
                logging.info("✓ Acceptability analysis complete")
            else:
                logging.info("⚠ Acceptability model could not be fitted, skipping")
        else:
            logging.info("⚠ No acceptability data available, skipping this analysis")
        
        # ========== STEP 12: Generate visualizations ==========
        logging.info("\n[STEP 9/10] Generating visualizations...")
        
        plot_files = {}
        
        # Scatter plot
        logging.info("Creating scatter plot...")
        plot_files['scatter'] = plot_scatter_tsv_pet(
            df=df_clean,
            pet_neutral=pet_neutral_result['pet_neutral'],
            output_path=output_path
        )
        
        # Probability curves
        logging.info("Creating probability curves plot...")
        plot_files['probability'] = plot_probability_curves(
            pet_grid=prob_results['pet_grid'],
            prob_category=prob_results['prob_category'],
            pet_neutral=pet_neutral_result['pet_neutral'],
            comfort_bands=comfort_bands,
            output_path=output_path
        )
        
        # Comfort zone
        logging.info("Creating comfort zone plot...")
        plot_files['comfort_zone'] = plot_comfort_zone(
            pet_grid=prob_results['pet_grid'],
            prob_comfort=prob_results['prob_comfort'],
            comfort_bands=comfort_bands,
            output_path=output_path
        )
        
        logging.info("✓ All visualizations generated")
        
        # ========== STEP 13: Generate report ==========
        logging.info("\n[STEP 10/10] Generating report...")
        
        # Format statistics
        stats = format_descriptive_statistics(df_clean, removal_stats)
        
        # Generate report
        report_file = generate_markdown_report(
            stats=stats,
            params=params,
            cis=cis,
            pet_neutral_result=pet_neutral_result,
            comfort_bands=comfort_bands,
            output_path=output_path,
            category_ranges=category_ranges,
            observed_ranges=observed_ranges,
            acceptability_bands=acceptability_bands,
            plot_files=plot_files
        )
        
        logging.info(f"✓ Report generated: {report_file}")
        
        # ========== OPTIONAL: Generate PDF ==========
        pdf_file = None
        if args.pdf:
            logging.info("\n[STEP 11/10] Generating PDF report...")
            pdf_file = convert_markdown_to_pdf(report_file)
            if pdf_file:
                logging.info(f"✓ PDF report generated: {pdf_file}")
        
        # ========== SUCCESS MESSAGE ==========
        logging.info("\n" + "=" * 70)
        logging.info("ANÁLISE CONCLUÍDA COM SUCESSO!")
        logging.info("=" * 70)
        logging.info("\nResultados Principais:")
        logging.info(f"  • PET Neutro: {pet_neutral_result['pet_neutral']:.1f}°C "
                    f"(IC 95%: [{pet_neutral_result['pet_neutral_ci'][0]:.1f}, "
                    f"{pet_neutral_result['pet_neutral_ci'][1]:.1f}]°C)")
        
        if 0.8 in comfort_bands and comfort_bands[0.8]['lower'] is not None:
            logging.info(f"  • Faixa de Conforto 80%: "
                        f"[{comfort_bands[0.8]['lower']:.1f}, {comfort_bands[0.8]['upper']:.1f}]°C")
        
        if 0.9 in comfort_bands and comfort_bands[0.9]['lower'] is not None:
            logging.info(f"  • Faixa de Conforto 90%: "
                        f"[{comfort_bands[0.9]['lower']:.1f}, {comfort_bands[0.9]['upper']:.1f}]°C")
        
        logging.info(f"\nArquivos Gerados:")
        logging.info(f"  • Dados limpos: {cleaned_file}")
        logging.info(f"  • Relatório: {report_file}")
        if pdf_file:
            logging.info(f"  • Relatório PDF: {pdf_file}")
        logging.info(f"  • Gráficos:")
        for plot_type, plot_path in plot_files.items():
            logging.info(f"    - {os.path.basename(plot_path)}")
        logging.info(f"  • Log: {log_file}")
        
        logging.info("\n" + "=" * 70)
        logging.info("Consulte o relatório RELATORIO_PET.md para análise completa.")
        logging.info("=" * 70)
        
        # Success exit
        return 0
        
    except KeyboardInterrupt:
        logging.error("\n\nExecução interrompida pelo usuário (Ctrl+C)")
        return 1
        
    except (DataLoadError, ValidationError, InsufficientDataError, ModelConvergenceError) as e:
        # Expected errors with user-friendly messages
        logging.error("\n" + "=" * 70)
        logging.error("ERRO NA ANÁLISE")
        logging.error("=" * 70)
        logging.error(f"\n{type(e).__name__}: {e}\n")
        logging.error("=" * 70)
        return 1
        
    except Exception as e:
        # Unexpected errors - show full traceback
        logging.error("\n" + "=" * 70)
        logging.error("ERRO INESPERADO")
        logging.error("=" * 70)
        logging.error(f"\n{type(e).__name__}: {e}\n")
        logging.error("Traceback completo:")
        logging.error(traceback.format_exc())
        logging.error("=" * 70)
        logging.error("\nPor favor, verifique os dados de entrada e tente novamente.")
        logging.error("Se o problema persistir, contate o desenvolvedor.")
        logging.error("=" * 70)
        return 1


if __name__ == '__main__':
    # Execute main function and exit with appropriate code
    exit_code = main()
    sys.exit(exit_code)
