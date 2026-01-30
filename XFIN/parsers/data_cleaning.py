"""
Portfolio Data Cleaning Utilities
===================================

Handles missing data in portfolio CSV files with intelligent calculation
of missing values from available data.

Extracted from stress_app.py for reusability.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


def clean_portfolio_data(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
    show_info: bool = False
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Handle missing data in portfolio CSV files.
    
    Operations:
    1. Remove rows where stock name itself is missing
    2. Calculate missing buy/sell prices from quantity and P&L
    3. Calculate missing values from available data
    4. Track and report all changes made
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio data to clean
    column_mapping : dict
        Mapping of standard names to actual column names
    show_info : bool
        If True, print info messages (for Streamlit compatibility)
        
    Returns
    -------
    tuple
        (cleaned DataFrame, dict of changes made)
    """
    changes = {
        "Removed Rows (Missing Stock Name)": [],
        "Calculated Missing Buy Prices": [],
        "Calculated Missing Sell Prices": [],
        "Calculated Missing Current Values": [],
        "Calculated Missing Invested Values": [],
        "Rows Removed (Insufficient Data)": []
    }
    
    original_count = len(df)
    
    # Find actual column names in DataFrame
    stock_col_actual = _find_column(df, column_mapping.get('stock_name'), 
                                     ['stock', 'security', 'name'])
    qty_col_actual = _find_column(df, column_mapping.get('quantity'),
                                   ['quantity', 'qty', 'units', 'shares', 'holdings'])
    pnl_col_actual = _find_column(df, column_mapping.get('pnl'),
                                   ['p&l', 'pnl', 'profit', 'loss', 'gain'])
    current_value_col_actual = _find_column(df, column_mapping.get('current_value'),
                                             ['current value', 'market value', 'closing value'])
    invested_value_col_actual = _find_column(df, column_mapping.get('invested_value'),
                                              ['invested value', 'buy value', 'cost value'])
    current_price_col_actual = _find_column(df, column_mapping.get('current_price'),
                                             ['closing price', 'current price', 'market price', 'ltp'])
    avg_price_col_actual = _find_column(df, column_mapping.get('avg_price'),
                                         ['average buy', 'avg buy', 'buy price', 'purchase price'])
    prev_closing_price_col = _find_column(df, None,
                                           ['previous closing', 'prev closing', 'last closing'])
    
    # Step 1: Remove rows with missing stock names
    if stock_col_actual and stock_col_actual in df.columns:
        rows_to_remove = (df[stock_col_actual].isna() | 
                         (df[stock_col_actual] == '') | 
                         (df[stock_col_actual].astype(str).str.strip() == ''))
        removed_count = rows_to_remove.sum()
        
        if removed_count > 0:
            df = df[~rows_to_remove].copy()
            changes["Removed Rows (Missing Stock Name)"].append(
                f"Removed {removed_count} rows with missing stock names"
            )
    
    # Step 2: Calculate missing prices from quantity and P&L
    for idx in df.index:
        row = df.loc[idx]
        stock_name = str(row.get(stock_col_actual, 'Unknown')) if stock_col_actual else 'Unknown'
        
        # Get values (convert to numeric, handle NaN)
        qty = pd.to_numeric(row.get(qty_col_actual, 0), errors='coerce') if qty_col_actual else 0
        pnl = pd.to_numeric(row.get(pnl_col_actual, 0), errors='coerce') if pnl_col_actual else 0
        current_val = pd.to_numeric(row.get(current_value_col_actual, None), errors='coerce') if current_value_col_actual else None
        invested_val = pd.to_numeric(row.get(invested_value_col_actual, None), errors='coerce') if invested_value_col_actual else None
        prev_closing_price = pd.to_numeric(row.get(prev_closing_price_col, None), errors='coerce') if prev_closing_price_col else None
        
        # Skip if quantity is 0 or NaN
        if pd.isna(qty) or qty == 0:
            continue
        
        # Use previous closing price as current price if needed
        if prev_closing_price_col and not pd.isna(prev_closing_price) and prev_closing_price > 0:
            current_price_value = pd.to_numeric(row.get(current_price_col_actual, None), errors='coerce') if current_price_col_actual else None
            
            if pd.isna(current_price_value):
                if current_price_col_actual:
                    df.at[idx, current_price_col_actual] = prev_closing_price
                else:
                    if 'Current Price' not in df.columns:
                        df['Current Price'] = None
                    df.at[idx, 'Current Price'] = prev_closing_price
                    current_price_col_actual = 'Current Price'
                
                changes["Calculated Missing Sell Prices"].append(
                    f"{stock_name}: Used Previous Closing Price = ₹{prev_closing_price:,.2f}"
                )
                
                # Calculate current value from previous closing price
                if pd.isna(current_val):
                    calculated_current_val = prev_closing_price * qty
                    if current_value_col_actual:
                        df.at[idx, current_value_col_actual] = calculated_current_val
                    else:
                        if 'Current Value' not in df.columns:
                            df['Current Value'] = None
                        df.at[idx, 'Current Value'] = calculated_current_val
                        current_value_col_actual = 'Current Value'
                    
                    changes["Calculated Missing Current Values"].append(
                        f"{stock_name}: Current Value = ₹{calculated_current_val:,.2f}"
                    )
                    current_val = calculated_current_val
        
        # Calculate missing invested value
        if pd.isna(invested_val) and not pd.isna(current_val) and not pd.isna(pnl):
            calculated_invested = current_val - pnl
            if calculated_invested > 0:
                col_name = invested_value_col_actual or 'Invested Value'
                if col_name not in df.columns:
                    df[col_name] = None
                df.at[idx, col_name] = calculated_invested
                changes["Calculated Missing Invested Values"].append(
                    f"{stock_name}: Invested Value = ₹{calculated_invested:,.2f}"
                )
                invested_val = calculated_invested
        
        # Calculate missing current value
        if pd.isna(current_val) and not pd.isna(invested_val) and not pd.isna(pnl):
            calculated_current = invested_val + pnl
            if calculated_current > 0:
                col_name = current_value_col_actual or 'Current Value'
                if col_name not in df.columns:
                    df[col_name] = None
                df.at[idx, col_name] = calculated_current
                changes["Calculated Missing Current Values"].append(
                    f"{stock_name}: Current Value = ₹{calculated_current:,.2f}"
                )
                current_val = calculated_current
        
        # Calculate buy price if missing
        if not pd.isna(invested_val) and invested_val > 0:
            buy_price = invested_val / qty
            for col in df.columns:
                if any(term in col.lower() for term in ['average buy', 'avg buy', 'buy price']):
                    if pd.isna(row.get(col, None)):
                        df.at[idx, col] = buy_price
                        changes["Calculated Missing Buy Prices"].append(
                            f"{stock_name}: Buy Price = ₹{buy_price:,.2f}"
                        )
                    break
        
        # Calculate current price if missing
        if not pd.isna(current_val) and current_val > 0:
            current_price = current_val / qty
            for col in df.columns:
                if any(term in col.lower() for term in ['closing price', 'current price', 'market price']):
                    if pd.isna(row.get(col, None)):
                        df.at[idx, col] = current_price
                        changes["Calculated Missing Sell Prices"].append(
                            f"{stock_name}: Current Price = ₹{current_price:,.2f}"
                        )
                    break
    
    # Step 3: Remove rows without essential data
    essential_cols = [current_value_col_actual, invested_value_col_actual]
    rows_to_remove_final = []
    
    for idx in df.index:
        has_value = False
        for col in essential_cols:
            if col and col in df.columns:
                val = pd.to_numeric(df.at[idx, col], errors='coerce')
                if not pd.isna(val) and val > 0:
                    has_value = True
                    break
        
        if not has_value:
            rows_to_remove_final.append(idx)
    
    if rows_to_remove_final:
        stock_names = []
        for idx in rows_to_remove_final:
            stock = df.at[idx, stock_col_actual] if stock_col_actual else f"Row {idx}"
            stock_names.append(str(stock))
        
        df = df.drop(rows_to_remove_final)
        changes["Rows Removed (Insufficient Data)"].append(
            f"Removed {len(rows_to_remove_final)} rows: {', '.join(stock_names[:5])}" +
            (f" and {len(stock_names) - 5} more" if len(stock_names) > 5 else "")
        )
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Remove empty change categories
    changes = {k: v for k, v in changes.items() if v}
    
    return df, changes


def _find_column(df: pd.DataFrame, mapped_col: Optional[str], 
                 search_terms: List[str]) -> Optional[str]:
    """Find column in DataFrame by mapped name or search terms."""
    # First try the mapped column
    if mapped_col:
        for col in df.columns:
            if mapped_col in str(col):
                return col
    
    # Fall back to search terms
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in search_terms):
            return col
    
    return None


def get_value_column(df: pd.DataFrame) -> Optional[str]:
    """Find the appropriate value column from the dataframe."""
    if df.empty:
        return None
    
    possible_columns = [
        'Invested Value', 'Current Value',
        'Closing value', 'Closing Value', 'closing value', 'CLOSING VALUE',
        'Buy value', 'Buy Value', 'buy value', 'BUY VALUE',
        'Market Value as of last trading day', 'Overall Gain/Loss',
        'Sell Value', 'Market Value', 'Market value', 'market value',
        'Investment Value', 'Cur. val', 'Investment', 'Holdings Value',
        'Total Value', 'Present Value', 'Value', 'value', 'VALUE',
        'Amount', 'Current Market Value'
    ]
    
    for col in possible_columns:
        if col in df.columns and not df[col].isna().all():
            try:
                pd.to_numeric(df[col], errors='coerce')
                return col
            except:
                continue
    
    # Look for any numeric column with value-related keywords
    for col in df.columns:
        col_lower = col.lower()
        if (('val' in col_lower or 'investment' in col_lower or 
             'amount' in col_lower or 'price' in col_lower) and
            pd.api.types.is_numeric_dtype(df[col]) and 
            not df[col].isna().all()):
            return col
    
    return None


def get_stock_name_column(df: pd.DataFrame) -> Optional[str]:
    """Find the appropriate stock name column."""
    if df.empty:
        return None
    
    possible_columns = [
        'Stock Name', 'stock name', 'Stock name', 'STOCK NAME',
        'Scrip/Contract', 'Company Name', 'Scrip', 'Contract',
        'Symbol', 'symbol', 'SYMBOL',
        'Security Name', 'security name', 'Security name', 'SECURITY NAME',
        'Name', 'name', 'NAME', 'ISIN',
        'Name of Security', 'Script Name', 'CompanyName', 'StockName',
        'SecurityName', 'Instrument', 'Ticker'
    ]
    
    for col in possible_columns:
        if col in df.columns and not df[col].isna().all():
            sample_values = df[col].dropna().head(5).astype(str)
            if any(len(val) > 1 and not val.replace('.', '').isdigit() 
                   for val in sample_values):
                return col
    
    # Fall back to first non-numeric column with name-like values
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
            sample_values = df[col].dropna().head(3).astype(str)
            if any(len(val) > 2 and not val.isdigit() and 'date' not in val.lower() 
                   for val in sample_values):
                return col
    
    return None
