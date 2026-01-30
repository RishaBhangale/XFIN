"""
Universal Broker CSV Parser
============================

Handles CSV files from any broker with intelligent format detection.
Supports: Zerodha, Upstox, Angel Broking, HDFC Securities, ICICI Direct,
Kotak Securities, Groww, and generic formats.

Extracted from stress_app.py for reusability.
"""

import pandas as pd
import re
from typing import Dict, List, Tuple, Optional


class UniversalBrokerCSVParser:
    """Universal parser that can handle CSV files from any broker with intelligent format detection"""
    
    def __init__(self):
        # Comprehensive mapping of broker terminologies to standard names
        self.column_mappings = {
            # Stock/Security Name variations
            'stock_name': [
                'Stock Name', 'stock name', 'STOCK NAME', 'StockName',
                'Security Name', 'security name', 'SECURITY NAME', 'SecurityName',
                'Scrip/Contract', 'Scrip', 'Contract', 'Symbol', 'SYMBOL',
                'Company Name', 'company name', 'COMPANY NAME', 'CompanyName',
                'Name of Security', 'Security', 'Instrument', 'ISIN',
                'Script Name', 'Name', 'NAME'
            ],
            
            # Quantity variations
            'quantity': [
                'Quantity', 'quantity', 'QUANTITY', 'Qty', 'QTY', 'qty',
                'Holdings', 'holdings', 'HOLDINGS', 'Units', 'UNITS',
                'Shares', 'shares', 'SHARES', 'No. of Shares'
            ],
            
            # Average/Buy Price variations
            'avg_price': [
                'Average buy price', 'Avg buy price', 'AVG BUY PRICE',
                'Avg Trading Price', 'Average Price', 'Buy Price', 'Purchase Price',
                'Avg Rate', 'Average Rate', 'Rate', 'Price', 'Unit Price',
                'Cost Price', 'Acquisition Price', 'Buy Rate'
            ],
            
            # Current/Market Price variations
            'current_price': [
                'Closing price', 'Closing Price', 'CLOSING PRICE', 'Close Price',
                'Current Price', 'Market Price', 'LTP', 'Last Traded Price',
                'Current Rate', 'Market Rate', 'Live Price', 'CMP'
            ],
            
            # Current/Market Value variations
            'current_value': [
                'Closing value', 'Closing Value', 'CLOSING VALUE',
                'Current Value', 'Market Value', 'Present Value',
                'Market Value as of last trading day', 'Current Market Value',
                'Total Value', 'Value', 'Holdings Value', 'Investment Value'
            ],
            
            # Invested/Buy Value variations
            'invested_value': [
                'Buy value', 'Buy Value', 'BUY VALUE', 'Purchase Value',
                'Invested Value', 'Investment Amount', 'Cost Value',
                'Total Cost', 'Acquisition Value', 'Book Value'
            ],
            
            # P&L variations
            'pnl': [
                'Unrealised P&L', 'P&L', 'PnL', 'Profit/Loss', 'Gain/Loss',
                'Unrealized P&L', 'Total P&L', 'Net P&L', 'Day P&L'
            ]
        }
        
        # Broker format fingerprints for detection
        self.broker_signatures = {
            'Zerodha': ['ISIN', 'Exchange', 'Last traded price'],
            'Upstox': ['Scrip/Contract', 'Market Value as of last trading day'],
            'Angel Broking': ['Symbol', 'Exchange', 'Net Qty'],
            'HDFC Securities': ['Security Name', 'Market Value', 'Unrealised P&L'],
            'ICICI Direct': ['Stock Name', 'Holdings', 'Current Value'],
            'Kotak Securities': ['Scrip Name', 'Quantity', 'Market Price'],
            'Groww': ['Stock Name', 'Qty', 'Current Price'],
            'Generic': []  # Fallback
        }
        
        # Financial indicators for header detection
        self.financial_indicators = [
            'price', 'value', 'quantity', 'qty', 'amount', 'rate', 'cost',
            'market', 'current', 'total', 'investment', 'holding', 'shares',
            'units', 'stock', 'security', 'scrip', 'symbol', 'isin',
            'p&l', 'pnl', 'profit', 'loss', 'gain', 'unrealised', 'unrealized',
            'realised', 'realized', 'buy', 'sell', 'closing', 'average', 'avg',
            'ltp', 'cmp', 'net', 'name', 'company', 'instrument', 'ticker',
            'exchange', 'segment', 'date', 'invested', 'returns', 'change',
            'chg', 'day', 'overall'
        ]
        
        # Section markers for P&L reports
        self.section_markers = {
            'unrealised': ['unrealised', 'unrealized', 'open positions', 'current holdings', 'holdings'],
            'realised': ['realised', 'realized', 'closed positions', 'sold', 'booked']
        }
    
    def detect_broker_format(self, headers: List[str]) -> str:
        """Detect broker format based on column signatures"""
        header_text = ' '.join(headers).lower()
        
        for broker, signatures in self.broker_signatures.items():
            if broker == 'Generic':
                continue
            
            matches = sum(1 for sig in signatures if sig.lower() in header_text)
            if matches >= len(signatures) * 0.6:  # 60% match threshold
                return broker
        
        return 'Generic'
    
    def find_header_row(self, lines: List[str]) -> Tuple[int, str]:
        """Smart header detection using pattern-based approach with fallback"""
        best_match = None
        best_score = 0
        first_data_row = None
        
        for i, line in enumerate(lines):
            if not line.strip():  # Skip empty lines
                continue
                
            # Skip obvious metadata rows
            line_lower = line.lower()
            if any(skip_term in line_lower for skip_term in 
                   ['client', 'account', 'report generated', 'date:', 'total portfolio', 
                    'grand total', 'segment:', 'from:', 'to:']):
                continue
            
            # Skip section headers
            if any(section in line_lower for section in 
                   ['unrealised p&l', 'realised p&l', 'unrealized p&l', 'realized p&l']):
                if ',' not in line or line.count(',') < 2:
                    continue
            
            # Split and clean the line
            cells = [cell.strip() for cell in line.split(',')]
            if len(cells) < 3:  # Need at least 3 columns
                continue
            
            # Count financial indicators in this row
            financial_count = 0
            for cell in cells:
                cell_lower = cell.lower()
                if any(indicator in cell_lower for indicator in self.financial_indicators):
                    financial_count += 1
            
            # Track best match
            if financial_count > best_score:
                best_score = financial_count
                best_match = (i, self.detect_broker_format(cells))
            
            # Track first multi-column row as ultimate fallback
            if first_data_row is None and len(cells) >= 4:
                text_cells = sum(1 for c in cells if c and 
                               not c.replace('.', '').replace('-', '').replace(',', '').isdigit())
                if text_cells >= 2:
                    first_data_row = (i, self.detect_broker_format(cells))
            
            # If this row has 3+ financial indicators, it's likely the header
            if financial_count >= 3:
                return i, self.detect_broker_format(cells)
        
        # Fallback 1: use best match if we found at least 2 indicators
        if best_match and best_score >= 2:
            return best_match
        
        # Fallback 2: use best match with even 1 indicator
        if best_match and best_score >= 1:
            return best_match
        
        # Fallback 3: use first multi-column row as header
        if first_data_row:
            return first_data_row
        
        raise ValueError("Could not detect header row - no row found with sufficient financial indicators")
    
    def extract_unrealised_section(self, lines: List[str]) -> List[str]:
        """
        Extract only the Unrealised P&L section from a P&L report.
        Returns filtered lines containing only unrealised (unsold) positions.
        """
        unrealised_start = None
        unrealised_end = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Look for Unrealised section start
            if unrealised_start is None:
                if any(marker in line_lower for marker in self.section_markers['unrealised']):
                    if 'p&l' in line_lower or 'profit' in line_lower or 'holdings' in line_lower:
                        unrealised_start = i
                        continue
            
            # Look for Realised section start (which ends Unrealised section)
            if unrealised_start is not None and unrealised_end is None:
                if any(marker in line_lower for marker in self.section_markers['realised']):
                    if 'p&l' in line_lower or 'profit' in line_lower or 'closed' in line_lower:
                        unrealised_end = i
                        break
                # Also stop at summary/total rows
                if 'total unrealised' in line_lower or 'sub total' in line_lower:
                    unrealised_end = i
                    break
        
        # If we found an unrealised section, extract it
        if unrealised_start is not None:
            if unrealised_end is not None:
                return lines[unrealised_start:unrealised_end]
            else:
                return lines[unrealised_start:]
        
        # No section markers found - return all lines (backwards compatible)
        return lines
    
    def map_columns(self, headers: List[str]) -> Dict[str, str]:
        """Map broker-specific column names to standard names"""
        mapped = {}
        
        for standard_name, variations in self.column_mappings.items():
            for header in headers:
                if header in variations:
                    mapped[standard_name] = header
                    break
        
        return mapped
    
    def clean_numeric_value(self, value) -> float:
        """Clean various number formats from different brokers"""
        if pd.isna(value) or value == '':
            return 0.0
        
        value_str = str(value)
        
        # Handle negative values in brackets: (1000) -> -1000
        if value_str.startswith('(') and value_str.endswith(')'):
            value_str = '-' + value_str[1:-1]
        
        # Remove currency symbols and commas
        value_str = value_str.replace('₹', '').replace('$', '').replace(',', '')
        
        # Remove percentage signs if present
        value_str = value_str.replace('%', '')
        
        # Extract just numbers, decimal points, and minus signs
        clean_value = re.sub(r'[^\d.-]', '', value_str)
        
        try:
            return float(clean_value) if clean_value else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def extract_data_rows(self, lines: List[str], header_idx: int, 
                         headers: List[str]) -> List[List[str]]:
        """Extract valid data rows, skipping metadata and footer rows"""
        data_rows = []
        
        for line in lines[header_idx + 1:]:
            if not line.strip():  # Skip empty lines
                continue
            
            # Skip summary/total rows
            line_lower = line.lower()
            if any(skip_term in line_lower for skip_term in 
                   ['total', 'grand total', 'portfolio total', 'summary', 'net amount']):
                continue
            
            row_data = [cell.strip() for cell in line.split(',')]
            
            # Ensure row has enough columns
            if len(row_data) < len(headers):
                row_data.extend([''] * (len(headers) - len(row_data)))
            elif len(row_data) > len(headers):
                row_data = row_data[:len(headers)]
            
            # Skip rows without a stock name
            if not row_data[0] or row_data[0].lower() in ['', 'na', 'n/a', 'null']:
                continue
            
            data_rows.append(row_data)
        
        return data_rows
    
    def parse(self, file_content) -> Optional[pd.DataFrame]:
        """
        Parse broker CSV file content into a DataFrame.
        
        Parameters
        ----------
        file_content : bytes or str
            The CSV file content
            
        Returns
        -------
        pd.DataFrame or None
            Parsed portfolio data, or None if parsing fails
        """
        try:
            # Read the entire file content
            if isinstance(file_content, bytes):
                content = file_content.decode('utf-8')
            else:
                content = str(file_content)
            
            lines = content.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Extract only Unrealised P&L section if present
            filtered_lines = self.extract_unrealised_section(lines)
            
            # Smart header detection
            header_idx, broker_format = self.find_header_row(filtered_lines)
            
            # Extract headers and map to standard names
            header_line = filtered_lines[header_idx]
            raw_headers = [col.strip() for col in header_line.split(',')]
            column_mapping = self.map_columns(raw_headers)
            
            # Extract data rows
            data_rows = self.extract_data_rows(filtered_lines, header_idx, raw_headers)
            
            if not data_rows:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=raw_headers)
            
            # Apply universal column cleaning
            for standard_name, broker_column in column_mapping.items():
                if broker_column in df.columns:
                    if standard_name in ['quantity', 'avg_price', 'current_price', 
                                        'current_value', 'invested_value', 'pnl']:
                        df[broker_column] = df[broker_column].apply(self.clean_numeric_value)
            
            # Store metadata
            df.attrs['broker_format'] = broker_format
            df.attrs['column_mapping'] = column_mapping
            
            return df
            
        except Exception as e:
            return None
