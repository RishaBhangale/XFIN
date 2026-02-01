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


# =============================================================================
# Ticker Normalization for ESG/Market Data
# =============================================================================

# Common Indian company names to NSE ticker mappings
INDIAN_COMPANY_TICKER_MAP = {
    # Top 50 Indian stocks
    'RELIANCE': 'RELIANCE.NS',
    'RELIANCE INDUSTRIES': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'TATA CONSULTANCY': 'TCS.NS',
    'TATA CONSULTANCY SERVICES': 'TCS.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'INFOSYS': 'INFY.NS',
    'INFY': 'INFY.NS',
    'ICICI BANK': 'ICICIBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'HINDUSTAN UNILEVER': 'HINDUNILVR.NS',
    'HUL': 'HINDUNILVR.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'STATE BANK': 'SBIN.NS',
    'SBI': 'SBIN.NS',
    'SBIN': 'SBIN.NS',
    'BHARTI AIRTEL': 'BHARTIARTL.NS',
    'AIRTEL': 'BHARTIARTL.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'KOTAK MAHINDRA': 'KOTAKBANK.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'KOTAK BANK': 'KOTAKBANK.NS',
    'BAJAJ FINANCE': 'BAJFINANCE.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'LARSEN': 'LT.NS',
    'L&T': 'LT.NS',
    'LT': 'LT.NS',
    'ASIAN PAINTS': 'ASIANPAINT.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'AXIS BANK': 'AXISBANK.NS',
    'AXISBANK': 'AXISBANK.NS',
    'MARUTI': 'MARUTI.NS',
    'MARUTI SUZUKI': 'MARUTI.NS',
    'TITAN': 'TITAN.NS',
    'WIPRO': 'WIPRO.NS',
    'HCLTECH': 'HCLTECH.NS',
    'HCL TECHNOLOGIES': 'HCLTECH.NS',
    'SUNPHARMA': 'SUNPHARMA.NS',
    'SUN PHARMA': 'SUNPHARMA.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'ULTRATECH': 'ULTRACEMCO.NS',
    'ULTRATECH CEMENT': 'ULTRACEMCO.NS',
    'POWERGRID': 'POWERGRID.NS',
    'POWER GRID': 'POWERGRID.NS',
    'NTPC': 'NTPC.NS',
    'TECHM': 'TECHM.NS',
    'TECH MAHINDRA': 'TECHM.NS',
    'ONGC': 'ONGC.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'TATA MOTORS': 'TATAMOTORS.NS',
    'TATASTEEL': 'TATASTEEL.NS',
    'TATA STEEL': 'TATASTEEL.NS',
    'JSWSTEEL': 'JSWSTEEL.NS',
    'JSW STEEL': 'JSWSTEEL.NS',
    'ADANIENT': 'ADANIENT.NS',
    'ADANI ENT': 'ADANIENT.NS',
    'ADANI ENTERPRISES': 'ADANIENT.NS',
    'ADANIPORTS': 'ADANIPORTS.NS',
    'ADANI PORTS': 'ADANIPORTS.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'BAJAJ FINSERV': 'BAJAJFINSV.NS',
    'DIVISLAB': 'DIVISLAB.NS',
    "DIVI'S LAB": 'DIVISLAB.NS',
    'DRREDDY': 'DRREDDY.NS',
    "DR REDDY'S": 'DRREDDY.NS',
    'CIPLA': 'CIPLA.NS',
    'EICHERMOT': 'EICHERMOT.NS',
    'EICHER MOTORS': 'EICHERMOT.NS',
    'GRASIM': 'GRASIM.NS',
    'BRITANNIA': 'BRITANNIA.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'NESTLE': 'NESTLEIND.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS',
    'HERO MOTOCORP': 'HEROMOTOCO.NS',
    'COALINDIA': 'COALINDIA.NS',
    'COAL INDIA': 'COALINDIA.NS',
    'INDUSINDBK': 'INDUSINDBK.NS',
    'INDUSIND BANK': 'INDUSINDBK.NS',
    'SHREECEM': 'SHREECEM.NS',
    'SHREE CEMENT': 'SHREECEM.NS',
    'HINDALCO': 'HINDALCO.NS',
    'M&M': 'M&M.NS',
    'MAHINDRA': 'M&M.NS',
    'TATACONSUM': 'TATACONSUM.NS',
    'TATA CONSUMER': 'TATACONSUM.NS',
    'BPCL': 'BPCL.NS',
    'IOC': 'IOC.NS',
    'INDIAN OIL': 'IOC.NS',
    'SBILIFE': 'SBILIFE.NS',
    'SBI LIFE': 'SBILIFE.NS',
    'HDFCLIFE': 'HDFCLIFE.NS',
    'HDFC LIFE': 'HDFCLIFE.NS',
    'ICICIGI': 'ICICIGI.NS',
    'ICICI GENERAL': 'ICICIGI.NS',
    'VEDL': 'VEDL.NS',
    'VEDANTA': 'VEDL.NS',
    'BAJAJ AUTO': 'BAJAJ-AUTO.NS',
    'BAJAJAUTO': 'BAJAJ-AUTO.NS',
    'UPL': 'UPL.NS',
    'APOLLOHOSP': 'APOLLOHOSP.NS',
    'APOLLO HOSPITALS': 'APOLLOHOSP.NS',
    
    # Additional stocks from Zerodha P&L reports
    'NLC INDIA': 'NLCINDIA.NS',
    'NLC INDIA LIMITED': 'NLCINDIA.NS',
    'NLCINDIA': 'NLCINDIA.NS',
    'NTPC LTD': 'NTPC.NS',
    'HINDUSTAN PETROLEUM CORP': 'HINDPETRO.NS',
    'HINDUSTAN PETROLEUM': 'HINDPETRO.NS',
    'HINDPETRO': 'HINDPETRO.NS',
    'HPCL': 'HINDPETRO.NS',
    'OIL AND NATURAL GAS CORP': 'ONGC.NS',
    'OIL AND NATURAL GAS CORP.': 'ONGC.NS',
    'BANK OF MAHARASHTRA': 'MAHABANK.NS',
    'MAHABANK': 'MAHABANK.NS',
    'CENTRAL DEPO SER (I) LTD': 'CDSL.NS',
    'CDSL': 'CDSL.NS',
    'COAL INDIA LTD': 'COALINDIA.NS',
    'GAIL': 'GAIL.NS',
    'GAIL (INDIA) LTD': 'GAIL.NS',
    'GAIL INDIA': 'GAIL.NS',
    'GRAPHITE INDIA': 'GRAPHITE.NS',
    'GRAPHITE INDIA LTD': 'GRAPHITE.NS',
    'GRAPHITE': 'GRAPHITE.NS',
    'IDFC FIRST BANK': 'IDFCFIRSTB.NS',
    'IDFC FIRST BANK LIMITED': 'IDFCFIRSTB.NS',
    'IDFCFIRSTB': 'IDFCFIRSTB.NS',
    'INDIAN OIL CORP': 'IOC.NS',
    'INDIAN OIL CORP LTD': 'IOC.NS',
    'ITC LTD': 'ITC.NS',
    'JIO FIN SERVICES': 'JIOFIN.NS',
    'JIO FIN SERVICES LTD': 'JIOFIN.NS',
    'JIOFIN': 'JIOFIN.NS',
    'NMDC': 'NMDC.NS',
    'NMDC LTD': 'NMDC.NS',
    'NMDC LTD.': 'NMDC.NS',
    'OIL INDIA': 'OIL.NS',
    'OIL INDIA LTD': 'OIL.NS',
    'POWER FIN CORP': 'PFC.NS',
    'POWER FIN CORP LTD': 'PFC.NS',
    'POWER FIN CORP LTD.': 'PFC.NS',
    'PFC': 'PFC.NS',
    'REC': 'RECLTD.NS',
    'REC LIMITED': 'RECLTD.NS',
    'RECLTD': 'RECLTD.NS',
    'RITES': 'RITES.NS',
    'RITES LIMITED': 'RITES.NS',
    'SUZLON': 'SUZLON.NS',
    'SUZLON ENERGY': 'SUZLON.NS',
    'SUZLON ENERGY LIMITED': 'SUZLON.NS',
    'TATA CAPITAL': 'TATACAPITAL.NS',
    'TATA CAPITAL LIMITED': 'TATACAPITAL.NS',
    'TATA MOTORS LIMITED': 'TATAMOTORS.NS',
    'SJVN': 'SJVN.NS',
    'SJVN LTD': 'SJVN.NS',
    'NCC': 'NCC.NS',
    'NCC LIMITED': 'NCC.NS',
    'IRCON': 'IRCON.NS',
    'IRCON INTERNATIONAL': 'IRCON.NS',
    'IRCON INTERNATIONAL LTD': 'IRCON.NS',
    'BHARAT PETROLEUM': 'BPCL.NS',
    'BHARAT PETROLEUM CORP': 'BPCL.NS',
    'BHARAT PETROLEUM CORP LT': 'BPCL.NS',
    'BHARAT PETROLEUM CORP  LT': 'BPCL.NS',
    'IRB INFRA': 'IRB.NS',
    'IRB INFRA DEV': 'IRB.NS',
    'IRB INFRA DEV LTD': 'IRB.NS',
    'IRB INFRA DEV LTD.': 'IRB.NS',
    'IRB': 'IRB.NS',
    'GUJ STATE FERT': 'GSFC.NS',
    'GUJ STATE FERT & CHEM': 'GSFC.NS',
    'GUJ STATE FERT & CHEM LTD': 'GSFC.NS',
    'GSFC': 'GSFC.NS',
    'MRPL': 'MRPL.NS',
    'CHOICE INTERNATIONAL': 'CHOICE.NS',
    'CHOICE INTERNATIONAL LTD': 'CHOICE.NS',
    'CHOICE': 'CHOICE.NS',
    # Additional stocks from user portfolio logs
    'VEDANTA': 'VEDL.NS',
    'VEDANTA LIMITED': 'VEDL.NS',
    'VEDL': 'VEDL.NS',
    'SESGOA': 'VEDL.NS',  # Old name for Vedanta
    'TATA STEEL': 'TATASTEEL.NS',
    'TATA STEEL LTD': 'TATASTEEL.NS',
    'TATA STEEL  LTD': 'TATASTEEL.NS',
    'TATIRO': 'TATASTEEL.NS',  # NSE symbol variant
    'TATASTEEL': 'TATASTEEL.NS',
    'ONGC': 'ONGC.NS',
    'ONGC LTD': 'ONGC.NS',
    'OILNAT': 'ONGC.NS',  # NSE symbol variant
    'OIL AND NATURAL GAS': 'ONGC.NS',
    'OIL AND NATURAL GAS CORPORATION': 'ONGC.NS',
    'NHPC': 'NHPC.NS',
    'NHPC LIMITED': 'NHPC.NS',
    'NHPC LTD': 'NHPC.NS',
    'NSDL': 'NSDL.NS',
    'NATIONAL SECURITIES DEPOSITORY': 'NSDL.NS',
    'NATIONAL SECURITIES DEPOSITORY LIMITED': 'NSDL.NS',
    'BAJAJ HOUSING': 'BAJAJHFL.NS',
    'BAJAJ HOUSING FINANCE': 'BAJAJHFL.NS',
    'BAJAJ HOUSING FINANCE LIMITED': 'BAJAJHFL.NS',
    'BAJAJHFL': 'BAJAJHFL.NS',
    'AEROFLEX': 'AEROFLEX.NS',
    'AEROFLEX ENTERPRISES': 'AEROFLEX.NS',
    'AEROFLEX ENTERPRISES LIMITED': 'AEROFLEX.NS',
    'EMCURE': 'EMCURE.NS',
    'EMCURE PHARMACEUTICALS': 'EMCURE.NS',
    'EMCURE PHARMACEUTICALS LIMITED': 'EMCURE.NS',
    'GANDHAR OIL': 'GANDHAR.NS',
    'GANDHAR OIL REFINERY': 'GANDHAR.NS',
    'GANDHAR OIL REFINERY (INDIA) LIMITED': 'GANDHAR.NS',
    'HATHWAY': 'HATHWAY.NS',
    'HATHWAY CABLE': 'HATHWAY.NS',
    'HATHWAY CABLE & DATACOM': 'HATHWAY.NS',
    'HATHWAY CABLE & DATACOM LIMITED': 'HATHWAY.NS',
    'HATHCABLE': 'HATHWAY.NS',  # NSE symbol variant
    'INDIAN OIL': 'IOC.NS',
    'INDIAN OIL CORPORATION': 'IOC.NS',
    'INDIAN OIL CORPORATION LTD': 'IOC.NS',
    'INDOIL': 'IOC.NS',  # NSE symbol variant
    'SHRIRAM PROPERTIES': 'SHRIRAMPP.NS',
    'SHRIRAM PROPERTIES LIMITED': 'SHRIRAMPP.NS',
    'SHRIRAMPP': 'SHRIRAMPP.NS',
    'RELIANCE': 'RELIANCE.NS',
    'RELIANCE INDUSTRIES': 'RELIANCE.NS',
    'RELIANCE INDUSTRIES LTD': 'RELIANCE.NS',
    'RELIND': 'RELIANCE.NS',  # NSE symbol variant
    'CDSL': 'CDSL.NS',
    'CENTRAL DEPOSITORY': 'CDSL.NS',
    'CENTRAL DEPOSITORY SERVICES': 'CDSL.NS',
    'CENTRAL DEPOSITORY SERVICES (INDIA) LTD': 'CDSL.NS',
    'SATIN': 'SATIN.NS',
    'SATIN CREDITCARE': 'SATIN.NS',
    'SATIN CREDITCARE NETWORK': 'SATIN.NS',
    'SATINV': 'SATIN.NS',
    'SATINVEQ': 'SATIN.NS',
    'BPCLTD': 'BPCL.NS',  # NSE symbol variant for BPCL
    'TATIRO': 'TATASTEEL.NS',  # NSE symbol variant
    'SHRIRAMPP': 'SHRIRAMPP.NS',
    'EMCUREEQ': 'EMCURE.NS',
    'GANDHAREQ': 'GANDHAR.NS',
}


def normalize_ticker(stock_name: str) -> str:
    """
    Normalize a stock name to a yfinance-compatible ticker.
    
    Handles:
    1. Already-valid tickers (e.g., RELIANCE.NS) - returns as-is
    2. Common Indian company names -> NSE ticker mapping
    3. Symbols with EQ/BE/BZ suffix (NSE series codes) - strips suffix
    4. Plain symbols without suffix -> adds .NS suffix
    
    Parameters
    ----------
    stock_name : str
        Stock name or ticker from portfolio CSV
        
    Returns
    -------
    str
        Normalized ticker with exchange suffix (e.g., RELIANCE.NS)
    """
    if not stock_name or not isinstance(stock_name, str):
        return stock_name
    
    stock_name = stock_name.strip()
    
    # Already has exchange suffix - check if it has EQ appended before suffix
    if any(stock_name.upper().endswith(suffix) for suffix in ['.NS', '.BO', '.NYSE', '.NASDAQ']):
        # Check for EQ suffix before .NS (e.g., RELIANCEEQ.NS -> RELIANCE.NS)
        upper = stock_name.upper()
        for series_suffix in ['EQ.NS', 'BE.NS', 'BZ.NS', 'EQ.BO', 'BE.BO', 'BZ.BO']:
            if upper.endswith(series_suffix):
                base = stock_name[:-len(series_suffix)].upper()
                exchange = series_suffix[-3:]  # .NS or .BO
                
                # Check if the base symbol is in our mapping table
                if base in INDIAN_COMPANY_TICKER_MAP:
                    return INDIAN_COMPANY_TICKER_MAP[base]
                
                return base + exchange
        return stock_name.upper() if stock_name.endswith(('.NS', '.BO')) else stock_name
    
    # Try exact match in mapping (case-insensitive)
    upper_name = stock_name.upper()
    if upper_name in INDIAN_COMPANY_TICKER_MAP:
        return INDIAN_COMPANY_TICKER_MAP[upper_name]
    
    # Try partial match for company names
    for key, ticker in INDIAN_COMPANY_TICKER_MAP.items():
        if key in upper_name or upper_name in key:
            return ticker
    
    # Strip NSE series suffixes (EQ, BE, BZ, etc.) from raw symbols
    clean_name = upper_name
    for series in ['EQ', 'BE', 'BZ', 'SM', 'MS', 'GS', 'TS']:
        if clean_name.endswith(series) and len(clean_name) > len(series):
            clean_name = clean_name[:-len(series)]
            break
    
    # Default: assume NSE ticker and add .NS suffix
    # Clean the name (remove special chars except hyphen)
    clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '-')
    if clean_name:
        return f"{clean_name}.NS"
    
    return stock_name


def normalize_portfolio_tickers(
    df: pd.DataFrame, 
    stock_column: Optional[str] = None,
    ticker_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Add normalized ticker column to portfolio DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Portfolio DataFrame
    stock_column : str, optional
        Column containing stock names. Auto-detected if not provided.
    ticker_column : str, optional
        Name for new ticker column. Defaults to 'Ticker'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added/updated ticker column
    """
    df = df.copy()
    
    # Find stock name column if not provided
    if not stock_column:
        stock_column = get_stock_name_column(df)
    
    if not stock_column or stock_column not in df.columns:
        return df
    
    # Set default ticker column name
    if not ticker_column:
        ticker_column = 'Ticker'
    
    # Create normalized tickers
    df[ticker_column] = df[stock_column].apply(normalize_ticker)
    
    return df

