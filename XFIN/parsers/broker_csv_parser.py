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
    
    def extract_unrealised_section(self, lines: List[str], prefer_holdings: bool = True) -> List[str]:
        """
        Extract the Unrealised P&L / Holdings section from a P&L report.
        
        Handles formats where:
        - Unrealised section is at the end (after Realised)
        - Unrealised section is at the beginning
        - Only one section exists
        
        Parameters
        ----------
        lines : List[str]
            All lines from the CSV file
        prefer_holdings : bool
            If True, prefer current holdings (unrealised) over closed trades (realised)
            
        Returns
        -------
        List[str]
            Filtered lines containing only the relevant section, or all lines if no sections found
        """
        sections = []
        current_section = None
        section_start = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Skip empty lines for section detection
            if not line_lower:
                continue
            
            # Count commas - section headers usually have few columns
            comma_count = line_lower.count(',')
            
            # Check if this looks like a section header (standalone row, not data row)
            # Section headers usually have many empty columns (commas) or few columns total
            is_likely_section_header = comma_count >= 3 and line_lower.replace(',', '').strip() == line_lower.replace(',', '').split(',')[0].strip()
            
            # If this line has meaningful data in multiple columns, skip it as section header
            cells = [c.strip() for c in line_lower.split(',')]
            non_empty_cells = [c for c in cells if c]
            if len(non_empty_cells) > 3:
                # This is likely a data row, not a section header
                continue
            
            # Detect section headers - be more specific
            # "Unrealised trades" or "Unrealised (Holdings as on ...)" are section headers
            # "Unrealised P&L" alone (in summary) is NOT a section header
            is_unrealised_header = (
                ('unrealised' in line_lower or 'unrealized' in line_lower) and
                ('trades' in line_lower or 'holdings' in line_lower or 'as on' in line_lower)
            )
            
            is_realised_header = (
                ('realised' in line_lower or 'realized' in line_lower) and
                ('trades' in line_lower)
            ) and not is_unrealised_header
            
            # Close previous section and start new one
            if is_unrealised_header or is_realised_header:
                if current_section is not None and section_start is not None:
                    sections.append({
                        'type': current_section,
                        'start': section_start,
                        'end': i
                    })
                
                current_section = 'unrealised' if is_unrealised_header else 'realised'
                section_start = i
        
        # Close the last section
        if current_section is not None and section_start is not None:
            sections.append({
                'type': current_section,
                'start': section_start,
                'end': len(lines)
            })
        
        # If no sections found, return all lines
        if not sections:
            return lines
        
        # Find the preferred section (prefer the LAST unrealised section - it's usually the holdings)
        preferred_section = None
        for section in reversed(sections):
            if section['type'] == 'unrealised':
                preferred_section = section
                break
        
        # Fall back to last section if no unrealised found
        if preferred_section is None:
            preferred_section = sections[-1] if prefer_holdings else sections[0]
        
        # Extract the section lines
        extracted = lines[preferred_section['start']:preferred_section['end']]
        
        # Remove the section header line itself if it's not the data header
        if extracted and ',' in extracted[0] and extracted[0].lower().count(',') < 3:
            extracted = extracted[1:]  # Skip the section title line
        
        return extracted if extracted else lines
    
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
            
            # Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]
            
            # Normalize line endings (handle Windows CRLF)
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            lines = content.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Try to extract Unrealised P&L section if present
            filtered_lines = self.extract_unrealised_section(lines)
            
            # If section filtering resulted in too few lines, use all lines
            if len(filtered_lines) < 3:
                filtered_lines = lines
            
            # Smart header detection
            header_idx, broker_format = self.find_header_row(filtered_lines)
            
            # Extract headers and map to standard names
            header_line = filtered_lines[header_idx]
            raw_headers = [col.strip() for col in header_line.split(',')]
            column_mapping = self.map_columns(raw_headers)
            
            # Extract data rows
            data_rows = self.extract_data_rows(filtered_lines, header_idx, raw_headers)
            
            # If no data rows found, try without section filtering
            if not data_rows and filtered_lines != lines:
                header_idx, broker_format = self.find_header_row(lines)
                header_line = lines[header_idx]
                raw_headers = [col.strip() for col in header_line.split(',')]
                column_mapping = self.map_columns(raw_headers)
                data_rows = self.extract_data_rows(lines, header_idx, raw_headers)
            
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

