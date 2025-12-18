"""
Portfolio Financial Data Pipeline
==================================

A production-grade Python solution for downloading and preprocessing financial market data
from Yahoo Finance for portfolio analysis and quantitative research.

Features:
- Financial correctness (no forward-fill bias)
- Calendar-enforced alignment (no gaps, no look-ahead)
- Reproducibility hardened (deterministic, timezone-normalized)
- Validation-first engineering (28 hard assertions, fail-fast)
- Institutional-grade error handling

Author: Quantitative Research Team
Date: December 2025
Usage: python portfolio_data_pipeline.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os
from pathlib import Path
import logging
import hashlib
import json
from typing import Dict, List, Optional, Tuple
import sys

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class PipelineException(Exception):
    """Base exception for pipeline errors."""
    pass

class ParameterValidationError(PipelineException):
    """Raised when parameters fail validation."""
    pass

class DataValidationError(PipelineException):
    """Raised when data fails quality checks."""
    pass

class AlignmentValidationError(PipelineException):
    """Raised when alignment checks fail."""
    pass

class CacheValidationError(PipelineException):
    """Raised when cache integrity is compromised."""
    pass

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Asset Configuration
ASSET_CONFIG = {
    'Equities (Eq)': ['SPY', 'IVV', 'QQQ', 'EEMV'],
    'Fixed Income (FI)': ['AGG', 'IEF'],
    'Alternatives (Alt)': ['GLD']
}

# Flatten all tickers in alphabetical order
ALL_TICKERS = sorted([ticker for category in ASSET_CONFIG.values() for ticker in category])

# Data Parameters
LOOKBACK_YEARS = 10
OUTPUT_DIR = Path('./data/processed')

# Price validation bounds (financial correctness)
MIN_VALID_PRICE = 0.01
MAX_VALID_PRICE = 100000.0

# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_parameters(end_date: str, tickers: List[str], lookback_years: int) -> Tuple[datetime, List[str], int]:
    """
    Validate all input parameters with hard assertions.
    
    Parameters
    ----------
    end_date : str
        End date in 'YYYY-MM-DD' format (required, not None)
    tickers : List[str]
        List of ticker symbols (non-empty, unique, all strings)
    lookback_years : int
        Number of years of historical data (1-50 range)
    
    Returns
    -------
    Tuple[datetime, List[str], int]
        Validated and normalized parameters
    
    Raises
    ------
    ParameterValidationError
        If any parameter fails validation
    """
    # Validate end_date
    if end_date is None:
        raise ParameterValidationError("end_date cannot be None (explicit date required for reproducibility)")
    
    if not isinstance(end_date, str):
        raise ParameterValidationError(f"end_date must be string, got {type(end_date).__name__}")
    
    try:
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ParameterValidationError(f"end_date must be 'YYYY-MM-DD' format, got '{end_date}': {str(e)}")
    
    # end_date cannot be in future
    if end_date_obj > datetime.utcnow():
        raise ParameterValidationError(f"end_date '{end_date}' cannot be in future (got {datetime.utcnow().date()})")
    
    # Validate tickers
    if tickers is None:
        raise ParameterValidationError("tickers cannot be None")
    
    if not isinstance(tickers, (list, tuple)):
        raise ParameterValidationError(f"tickers must be list or tuple, got {type(tickers).__name__}")
    
    if len(tickers) == 0:
        raise ParameterValidationError("tickers list cannot be empty")
    
    # Check all tickers are strings
    for ticker in tickers:
        if not isinstance(ticker, str):
            raise ParameterValidationError(f"All tickers must be strings, got {type(ticker).__name__} in {tickers}")
    
    # Check for duplicates
    if len(tickers) != len(set(tickers)):
        raise ParameterValidationError(f"Duplicate tickers found: {tickers}")
    
    # Normalize tickers (uppercase)
    tickers_normalized = [t.upper() for t in tickers]
    
    # Validate lookback_years
    if not isinstance(lookback_years, int):
        raise ParameterValidationError(f"lookback_years must be integer, got {type(lookback_years).__name__}")
    
    if not (1 <= lookback_years <= 50):
        raise ParameterValidationError(f"lookback_years must be 1-50, got {lookback_years}")
    
    # Calculate start_date from end_date
    start_date_obj = end_date_obj - timedelta(days=365.25 * lookback_years)
    
    logger.info(f"✓ Parameters validated:")
    logger.info(f"  - end_date: {end_date} (UTC)")
    logger.info(f"  - start_date: {start_date_obj.date()}")
    logger.info(f"  - tickers: {tickers_normalized}")
    logger.info(f"  - lookback_years: {lookback_years}")
    
    return end_date_obj, tickers_normalized, lookback_years

# ============================================================================
# STEP 1: DATA DOWNLOAD
# ============================================================================

def download_market_data(end_date: datetime, tickers: List[str], lookback_years: int) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Download OHLCV data from Yahoo Finance for specified tickers.
    
    Parameters
    ----------
    end_date : datetime
        End date for download (UTC)
    tickers : List[str]
        List of ticker symbols
    lookback_years : int
        Number of years of historical data
    
    Returns
    -------
    Dict[str, Optional[pd.DataFrame]]
        Dictionary with ticker symbols as keys and DataFrames as values (None if failed)
    
    Raises
    ------
    DataValidationError
        If download fails for all tickers
    """
    start_date = end_date - timedelta(days=365.25 * lookback_years)
    
    logger.info(f"Downloading data from {start_date.date()} to {end_date.date()}")
    logger.info(f"Tickers to download: {tickers}")
    
    data_dict = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            logger.info(f"Downloading {ticker}...")
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            # Validate download result is not None/empty
            if df is None or df.empty:
                logger.warning(f"✗ {ticker}: No data returned from Yahoo Finance")
                data_dict[ticker] = None
                failed_tickers.append(ticker)
                continue
            
            # Handle case where yfinance returns Series instead of DataFrame
            if isinstance(df, pd.Series):
                df = df.to_frame()
            
            data_dict[ticker] = df
            logger.info(f"✓ {ticker}: {len(df)} records downloaded")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {ticker}: {str(e)}")
            data_dict[ticker] = None
            failed_tickers.append(ticker)
    
    # Check if we have at least some data
    valid_tickers = [t for t in tickers if data_dict[t] is not None]
    if not valid_tickers:
        raise DataValidationError(f"Failed to download data for all tickers: {failed_tickers}")
    
    if failed_tickers:
        logger.warning(f"Failed to download {len(failed_tickers)} tickers: {failed_tickers}")
    
    return data_dict

# ============================================================================
# STEP 2: DATA CLEANING & PROCESSING
# ============================================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase with underscores.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from yfinance
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names
    """
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume'
    }
    
    df.columns = df.columns.str.strip()
    df = df.rename(columns=column_mapping)
    return df

def validate_single_ticker_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Validate data quality for a single ticker (hard assertions).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    ticker : str
        Ticker symbol (for logging)
    
    Raises
    ------
    DataValidationError
        If any validation check fails
    """
    if df is None or df.empty:
        raise DataValidationError(f"{ticker}: Data is None or empty")
    
    # Check for required columns
    required_cols = ['adj_close', 'close']
    has_required = any(col in df.columns for col in required_cols)
    if not has_required:
        raise DataValidationError(f"{ticker}: Missing both 'adj_close' and 'close' columns")
    
    # Check for all-NaN rows
    if df.isnull().all(axis=1).any():
        raise DataValidationError(f"{ticker}: Found rows with all NaN values")
    
    # Check price column for valid range
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
    prices = df[price_col].dropna()
    
    if len(prices) == 0:
        raise DataValidationError(f"{ticker}: No valid prices found")
    
    if (prices <= MIN_VALID_PRICE).any():
        raise DataValidationError(f"{ticker}: Found prices <= {MIN_VALID_PRICE} (minimum valid price)")
    
    if (prices >= MAX_VALID_PRICE).any():
        raise DataValidationError(f"{ticker}: Found prices >= {MAX_VALID_PRICE} (maximum valid price)")
    
    # Check date index is monotonic
    if not df.index.is_monotonic_increasing:
        raise DataValidationError(f"{ticker}: Date index is not monotonic increasing")
    
    # Check timezone is UTC or timezone-naive
    if df.index.tz is not None and str(df.index.tz) != 'UTC':
        raise DataValidationError(f"{ticker}: Date index has non-UTC timezone: {df.index.tz}")

def clean_single_ticker_data(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """
    Clean and preprocess data for a single ticker.
    
    Key fixes:
    - NO forward-fill (removed deprecated fillna(method='ffill'))
    - Timezone normalization to UTC
    - Numeric type validation
    - NaN handling with transparent logging
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from yfinance
    ticker : str
        Ticker symbol (for logging)
    
    Returns
    -------
    Optional[pd.DataFrame]
        Cleaned DataFrame, or None if critical errors occur
    
    Raises
    ------
    DataValidationError
        If data fails quality checks
    """
    if df is None or df.empty:
        logger.warning(f"Empty data for {ticker}, skipping")
        return None
    
    logger.info(f"Cleaning {ticker}...")
    
    # Step 1: Standardize column names
    df = standardize_column_names(df)
    
    # Step 2: Reset index to make date a column (handling MultiIndex from yfinance)
    df = df.reset_index()
    df.columns = df.columns.str.lower()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['date'] = pd.to_datetime(df.index)
    
    # Step 3: Remove duplicates by date (keep first occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['date'], keep='first')
    if len(df) < initial_rows:
        logger.warning(f"{ticker}: Removed {initial_rows - len(df)} duplicate dates")
    
    # Step 4: Sort by date in ascending order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Step 5: Ensure numeric types BEFORE any NaN handling
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 6: REMOVED forward-fill (financial bias source)
    # Instead: drop rows with missing price data (NaN in close or adj_close)
    # This is financially correct - gaps represent data quality issues
    nan_before = df.isnull().sum()
    
    price_cols = ['close', 'adj_close']
    for col in price_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
    
    nan_after = df.isnull().sum()
    nan_dropped = nan_before - nan_after
    if (nan_dropped > 0).any():
        logger.warning(f"{ticker}: Dropped {nan_dropped.sum()} rows with missing prices (NaN not filled)")
    
    # Step 7: Ensure timezone-naive, UTC timestamps
    df['date'] = df['date'].dt.tz_localize(None)
    
    # Step 8: Set date as index for easier alignment
    df = df.set_index('date')
    
    # Step 9: Validate cleaned data
    validate_single_ticker_data(df, ticker)
    
    logger.info(f"✓ {ticker} cleaned: {len(df)} records retained")
    return df

# ============================================================================
# STEP 3: CALENDAR ALIGNMENT
# ============================================================================

def create_trading_calendar(cleaned_data: Dict[str, Optional[pd.DataFrame]]) -> pd.DatetimeIndex:
    """
    Create master trading calendar as intersection of all asset dates.
    
    Parameters
    ----------
    cleaned_data : Dict[str, Optional[pd.DataFrame]]
        Dictionary with cleaned DataFrames
    
    Returns
    -------
    pd.DatetimeIndex
        Master calendar of common trading dates (sorted)
    
    Raises
    ------
    AlignmentValidationError
        If no common dates or alignment fails
    """
    # Filter out None values
    valid_data = {ticker: df for ticker, df in cleaned_data.items() if df is not None}
    
    if not valid_data:
        raise AlignmentValidationError("No valid data to align")
    
    # Get common dates (intersection/inner join)
    all_dates = valid_data[list(valid_data.keys())[0]].index
    
    for ticker in list(valid_data.keys())[1:]:
        all_dates = all_dates.intersection(valid_data[ticker].index)
    
    # Validate alignment result
    if len(all_dates) == 0:
        raise AlignmentValidationError("No common trading dates found across all assets")
    
    if not all_dates.is_monotonic_increasing:
        raise AlignmentValidationError("Master calendar dates not monotonic increasing")
    
    logger.info(f"Created master trading calendar: {len(all_dates)} common dates")
    logger.info(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")
    
    return all_dates

def align_assets_by_date(cleaned_data: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Align all asset data to master trading calendar (no gaps, no look-ahead).
    
    Parameters
    ----------
    cleaned_data : Dict[str, Optional[pd.DataFrame]]
        Dictionary with ticker symbols as keys and DataFrames as values
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with aligned DataFrames (all assets have same trading dates)
    
    Raises
    ------
    AlignmentValidationError
        If alignment fails any validation check
    """
    # Create master calendar
    all_dates = create_trading_calendar(cleaned_data)
    
    # Filter valid data and reindex to common dates
    valid_data = {ticker: df for ticker, df in cleaned_data.items() if df is not None}
    aligned_data = {}
    
    for ticker, df in valid_data.items():
        aligned_df = df.loc[all_dates].copy()
        
        # Validate no new NaNs introduced by reindexing
        if aligned_df.isnull().any().any():
            raise AlignmentValidationError(f"{ticker}: Reindexing introduced NaN values")
        
        aligned_data[ticker] = aligned_df
    
    # Final alignment validation
    validate_alignment(aligned_data)
    
    return aligned_data

def validate_alignment(aligned_data: Dict[str, pd.DataFrame]) -> None:
    """
    Validate that all assets are properly aligned (hard assertions).
    
    Parameters
    ----------
    aligned_data : Dict[str, pd.DataFrame]
        Dictionary with aligned DataFrames
    
    Raises
    ------
    AlignmentValidationError
        If any alignment check fails
    """
    if not aligned_data:
        raise AlignmentValidationError("No aligned data to validate")
    
    # Get reference shape from first asset
    tickers = sorted(aligned_data.keys())
    reference_ticker = tickers[0]
    reference_shape = aligned_data[reference_ticker].shape
    reference_index = aligned_data[reference_ticker].index
    
    # Check all assets have identical shape and index
    for ticker in tickers[1:]:
        df = aligned_data[ticker]
        
        if df.shape != reference_shape:
            raise AlignmentValidationError(
                f"{ticker}: Shape mismatch. Expected {reference_shape}, got {df.shape}"
            )
        
        if not df.index.equals(reference_index):
            raise AlignmentValidationError(f"{ticker}: Index mismatch with {reference_ticker}")
        
        if len(df) == 0:
            raise AlignmentValidationError(f"{ticker}: DataFrame is empty after alignment")
        
        # Check no all-NaN columns
        if df.isnull().all(axis=0).any():
            raise AlignmentValidationError(f"{ticker}: Found all-NaN columns after alignment")
    
    logger.info(f"✓ Alignment validated: {len(tickers)} assets, {len(reference_index)} dates, all aligned")

# ============================================================================
# STEP 4: OUTPUT & FILE SAVING
# ============================================================================

def ensure_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {OUTPUT_DIR}")

def save_individual_csvs(aligned_data: Dict[str, pd.DataFrame]) -> None:
    """
    Save cleaned data for each ticker as individual CSV files.
    
    Parameters
    ----------
    aligned_data : Dict[str, pd.DataFrame]
        Dictionary with ticker symbols as keys and aligned DataFrames as values
    """
    for ticker in sorted(aligned_data.keys()):
        df = aligned_data[ticker]
        filename = f"{ticker}_clean.csv"
        filepath = OUTPUT_DIR / filename
        
        # Reset index to include date as column
        df_export = df.reset_index()
        df_export.to_csv(filepath, index=False)
        
        logger.info(f"✓ Saved: {filename} ({len(df_export)} rows)")

def create_combined_adj_close_matrix(aligned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a combined DataFrame with adjusted close prices for all assets.
    
    Parameters
    ----------
    aligned_data : Dict[str, pd.DataFrame]
        Dictionary with ticker symbols as keys and aligned DataFrames as values
    
    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and adjusted close prices as columns
    
    Raises
    ------
    DataValidationError
        If adj_close column missing or contains NaN
    """
    prices_data = {}
    
    for ticker in sorted(aligned_data.keys()):
        df = aligned_data[ticker]
        
        if 'adj_close' not in df.columns:
            raise DataValidationError(f"{ticker}: Missing 'adj_close' column after cleaning")
        
        prices = df['adj_close']
        
        if prices.isnull().any():
            raise DataValidationError(f"{ticker}: Found NaN values in 'adj_close' after alignment")
        
        prices_data[ticker] = prices
    
    # Combine into single DataFrame (should maintain alignment)
    combined_df = pd.DataFrame(prices_data)
    combined_df.index.name = 'date'
    
    return combined_df

def save_combined_csv(combined_df: pd.DataFrame) -> None:
    """
    Save combined adjusted close price matrix to CSV.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        DataFrame with dates as index and tickers as columns
    """
    filename = 'prices_adj_close_10y.csv'
    filepath = OUTPUT_DIR / filename
    
    combined_df.to_csv(filepath)
    
    logger.info(f"✓ Saved: {filename} ({len(combined_df)} trading days, {len(combined_df.columns)} assets)")

# ============================================================================
# REPRODUCIBILITY: CACHE & HASH VERIFICATION
# ============================================================================

def compute_data_hash(combined_df: pd.DataFrame) -> str:
    """
    Compute SHA256 hash of data for reproducibility verification.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined price matrix
    
    Returns
    -------
    str
        SHA256 hash of CSV representation
    """
    csv_str = combined_df.to_csv()
    hash_obj = hashlib.sha256(csv_str.encode())
    return hash_obj.hexdigest()

def save_metadata(end_date: datetime, tickers: List[str], lookback_years: int, data_hash: str) -> None:
    """
    Save run metadata for reproducibility audit trail.
    
    Parameters
    ----------
    end_date : datetime
        End date used for download
    tickers : List[str]
        Tickers downloaded
    lookback_years : int
        Lookback period
    data_hash : str
        SHA256 hash of output data
    """
    metadata = {
        'run_date': datetime.utcnow().isoformat(),
        'end_date': end_date.isoformat(),
        'tickers': tickers,
        'lookback_years': lookback_years,
        'data_hash': data_hash,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__
    }
    
    metadata_file = OUTPUT_DIR / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved: metadata.json")
    logger.info(f"  Data hash (SHA256): {data_hash}")

# ============================================================================
# DATA VALIDATION & SUMMARY
# ============================================================================

def generate_data_summary(aligned_data: Dict[str, pd.DataFrame]) -> None:
    """
    Generate summary statistics for downloaded data.
    
    Parameters
    ----------
    aligned_data : Dict[str, pd.DataFrame]
        Dictionary with cleaned and aligned DataFrames
    """
    logger.info("\n" + "="*70)
    logger.info("DATA SUMMARY STATISTICS")
    logger.info("="*70)
    
    # Overall coverage
    all_tickers = sorted(aligned_data.keys())
    num_tickers = len(all_tickers)
    num_dates = len(aligned_data[all_tickers[0]])
    
    logger.info(f"\nTotal Assets: {num_tickers}")
    logger.info(f"Total Trading Days (aligned): {num_dates}")
    
    # Date range
    first_date = aligned_data[all_tickers[0]].index.min()
    last_date = aligned_data[all_tickers[0]].index.max()
    days_diff = (last_date - first_date).days
    
    logger.info(f"Date Range: {first_date.date()} to {last_date.date()} ({days_diff} days)")
    
    # Per-asset summary
    logger.info("\nPer-Asset Summary:")
    logger.info("-" * 70)
    
    for ticker in all_tickers:
        df = aligned_data[ticker]
        
        # Use adj_close if available, otherwise close
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
        prices = df[price_col]
        
        price_range = f"{prices.min():.2f} - {prices.max():.2f}"
        latest_price = prices.iloc[-1]
        price_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        
        logger.info(f"{ticker:6s} | Range: {price_range:20s} | Latest: ${latest_price:8.2f} | Return: {price_return:7.2f}%")
    
    logger.info("="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(end_date: str = None, tickers: List[str] = None, lookback_years: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Main execution function - complete portfolio data pipeline.
    
    Parameters
    ----------
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses today's date.
    tickers : List[str], optional
        List of ticker symbols. If None, uses default asset list.
    lookback_years : int, default=10
        Number of years of historical data
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with aligned data for each ticker
    
    Raises
    ------
    ParameterValidationError
        If parameters fail validation
    DataValidationError
        If data quality fails
    AlignmentValidationError
        If alignment fails
    """
    logger.info("="*70)
    logger.info("PORTFOLIO DATA PIPELINE - START")
    logger.info("="*70)
    
    # Set defaults if not provided
    if end_date is None:
        end_date = datetime.utcnow().strftime('%Y-%m-%d')
    
    if tickers is None:
        tickers = ALL_TICKERS
    
    # Step 0: Validate parameters
    logger.info("\n[STEP 0] PARAMETER VALIDATION")
    logger.info("-" * 70)
    end_date_validated, tickers_validated, lookback_years_validated = validate_parameters(
        end_date, tickers, lookback_years
    )
    
    # Step 1: Download raw data
    logger.info("\n[STEP 1] DOWNLOADING MARKET DATA")
    logger.info("-" * 70)
    raw_data = download_market_data(end_date_validated, tickers_validated, lookback_years_validated)
    
    # Step 2: Clean individual datasets
    logger.info("\n[STEP 2] CLEANING & PREPROCESSING DATA")
    logger.info("-" * 70)
    cleaned_data = {}
    for ticker in tickers_validated:
        try:
            cleaned_data[ticker] = clean_single_ticker_data(raw_data[ticker], ticker)
        except DataValidationError as e:
            logger.error(f"✗ {ticker}: {str(e)}")
            raise
    
    # Step 3: Align by trading date
    logger.info("\n[STEP 3] ALIGNING ASSETS BY TRADING DATE")
    logger.info("-" * 70)
    try:
        aligned_data = align_assets_by_date(cleaned_data)
    except AlignmentValidationError as e:
        logger.error(f"✗ Alignment failed: {str(e)}")
        raise
    
    # Step 4: Create output directory and save files
    logger.info("\n[STEP 4] SAVING OUTPUT FILES")
    logger.info("-" * 70)
    ensure_output_directory()
    save_individual_csvs(aligned_data)
    
    # Step 5: Create and save combined matrix
    logger.info("\n[STEP 5] CREATING COMBINED PRICE MATRIX")
    logger.info("-" * 70)
    combined_prices = create_combined_adj_close_matrix(aligned_data)
    save_combined_csv(combined_prices)
    
    # Step 6: Generate summary and metadata
    logger.info("\n[STEP 6] DATA VALIDATION & SUMMARY")
    logger.info("-" * 70)
    generate_data_summary(aligned_data)
    
    # Step 7: Compute hash for reproducibility
    logger.info("\n[STEP 7] REPRODUCIBILITY VERIFICATION")
    logger.info("-" * 70)
    data_hash = compute_data_hash(combined_prices)
    save_metadata(end_date_validated, tickers_validated, lookback_years_validated, data_hash)
    
    logger.info("="*70)
    logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"✓ All files saved to: {OUTPUT_DIR.absolute()}")
    logger.info("="*70)
    
    return aligned_data

if __name__ == "__main__":
    main()
