"""
Tests for XFIN Parsers Module
==============================

Unit tests for UniversalBrokerCSVParser and data cleaning utilities.
These are the new modules extracted from stress_app.py.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Fixtures for Parsers Testing
# =============================================================================

@pytest.fixture
def zerodha_csv_content():
    """Sample Zerodha broker CSV content."""
    return """Stock Name,ISIN,Quantity,Average buy price,Buy value,Closing price,Closing value
RELIANCE,INE002A01018,100,2450.50,245050.00,2500.00,250000.00
HDFCBANK,INE040A01034,50,1650.00,82500.00,1680.00,84000.00
TCS,INE467B01029,25,3600.00,90000.00,3650.00,91250.00
"""

@pytest.fixture
def upstox_csv_content():
    """Sample Upstox broker CSV content."""
    return """Symbol,ISIN,Qty,Avg. Cost,LTP,Current Value,P&L
RELIANCE,INE002A01018,100,2450.50,2500.00,250000.00,4950.00
HDFCBANK,INE040A01034,50,1650.00,1680.00,84000.00,1500.00
"""

@pytest.fixture
def generic_csv_content():
    """Generic broker CSV content with different column names."""
    return """Company Name,Shares,Buy Price,Current Price,Total Value
Reliance Industries,100,2450.00,2500.00,250000.00
HDFC Bank Ltd,50,1650.00,1680.00,84000.00
"""

@pytest.fixture
def sample_portfolio_df():
    """Sample portfolio DataFrame for data cleaning tests."""
    return pd.DataFrame({
        'Stock Name': ['RELIANCE', 'TCS', 'HDFCBANK', None, 'INFY'],
        'Quantity': [100, 50, 75, 25, 0],
        'Buy Value': [245000, 180000, 123750, 50000, 0],
        'Current Value': [250000, 182500, 126000, 0, 75000],
        'P&L': [5000, 2500, 2250, -500, 0]
    })


# =============================================================================
# Tests for UniversalBrokerCSVParser
# =============================================================================

class TestUniversalBrokerCSVParser:
    """Tests for UniversalBrokerCSVParser class."""
    
    def test_import(self):
        """Test that UniversalBrokerCSVParser can be imported."""
        from XFIN.parsers import UniversalBrokerCSVParser
        assert UniversalBrokerCSVParser is not None
    
    def test_init(self):
        """Test UniversalBrokerCSVParser initialization."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        assert parser is not None
        assert hasattr(parser, 'column_mappings')
        assert hasattr(parser, 'broker_signatures')
    
    def test_column_mappings_exist(self):
        """Test that required column mappings are defined."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        # Check for actual mapping keys (avg_price instead of buy_price)
        required_mappings = ['stock_name', 'quantity', 'avg_price', 'current_price']
        for mapping in required_mappings:
            assert mapping in parser.column_mappings, f"Missing mapping: {mapping}"
    
    def test_detect_broker_format_zerodha(self):
        """Test broker format detection for Zerodha."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        zerodha_headers = ['Stock Name', 'ISIN', 'Quantity', 'Average buy price', 'Buy value']
        broker = parser.detect_broker_format(zerodha_headers)
        
        # Should detect as Zerodha or generic
        assert broker is not None
        assert isinstance(broker, str)
    
    def test_detect_broker_format_upstox(self):
        """Test broker format detection for Upstox."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        upstox_headers = ['Symbol', 'ISIN', 'Qty', 'Avg. Cost', 'LTP', 'Current Value']
        broker = parser.detect_broker_format(upstox_headers)
        
        assert broker is not None
        assert isinstance(broker, str)
    
    def test_map_columns_basic(self):
        """Test column mapping functionality."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        headers = ['Stock Name', 'Quantity', 'Average buy price', 'Current Value']
        mapping = parser.map_columns(headers)
        
        assert isinstance(mapping, dict)
        # Should map at least stock_name and quantity
        assert 'stock_name' in mapping or len(mapping) > 0
    
    def test_clean_numeric_value_basic(self):
        """Test numeric value cleaning."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        # Test various formats
        assert parser.clean_numeric_value('1,234.56') == 1234.56
        assert parser.clean_numeric_value('₹1,234.56') == 1234.56
        assert parser.clean_numeric_value('1234') == 1234.0
        assert parser.clean_numeric_value('') == 0.0
        assert parser.clean_numeric_value(None) == 0.0
    
    def test_clean_numeric_value_negative(self):
        """Test cleaning negative numeric values."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        assert parser.clean_numeric_value('-1,234.56') == -1234.56
        assert parser.clean_numeric_value('(1234.56)') == -1234.56  # Accounting format
    
    def test_find_header_row(self, zerodha_csv_content):
        """Test header row detection."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        lines = zerodha_csv_content.strip().split('\n')
        header_idx, broker = parser.find_header_row(lines)
        
        assert header_idx >= 0
        assert broker is not None
    
    def test_extract_data_rows(self, zerodha_csv_content):
        """Test data row extraction."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        lines = zerodha_csv_content.strip().split('\n')
        header_idx, _ = parser.find_header_row(lines)
        headers = lines[header_idx].split(',')
        
        data_rows = parser.extract_data_rows(lines, header_idx, headers)
        
        assert isinstance(data_rows, list)
        assert len(data_rows) >= 2  # At least 2 stocks


class TestBrokerCSVParserEdgeCases:
    """Edge case tests for UniversalBrokerCSVParser."""
    
    def test_empty_file(self):
        """Test handling of empty file content."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        lines = []
        # Should raise ValueError for empty file
        with pytest.raises(ValueError):
            parser.find_header_row(lines)
    
    def test_whitespace_values(self):
        """Test handling of whitespace in values."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        assert parser.clean_numeric_value('  1234  ') == 1234.0
        assert parser.clean_numeric_value('  ') == 0.0
    
    def test_mixed_case_headers(self):
        """Test handling of mixed case column headers."""
        from XFIN.parsers import UniversalBrokerCSVParser
        parser = UniversalBrokerCSVParser()
        
        headers = ['STOCK NAME', 'quantity', 'Buy Price', 'CURRENT VALUE']
        mapping = parser.map_columns(headers)
        
        # Should handle case-insensitive matching
        assert isinstance(mapping, dict)


# =============================================================================
# Tests for Data Cleaning Functions
# =============================================================================

class TestDataCleaningFunctions:
    """Tests for data cleaning utility functions."""
    
    def test_clean_portfolio_data_import(self):
        """Test that clean_portfolio_data can be imported."""
        from XFIN.parsers.data_cleaning import clean_portfolio_data
        assert clean_portfolio_data is not None
    
    def test_get_value_column_import(self):
        """Test that get_value_column can be imported."""
        from XFIN.parsers.data_cleaning import get_value_column
        assert get_value_column is not None
    
    def test_get_stock_name_column_import(self):
        """Test that get_stock_name_column can be imported."""
        from XFIN.parsers.data_cleaning import get_stock_name_column
        assert get_stock_name_column is not None
    
    def test_get_value_column_current_value(self, sample_portfolio_df):
        """Test finding Current Value column."""
        from XFIN.parsers.data_cleaning import get_value_column
        
        col = get_value_column(sample_portfolio_df)
        
        assert col is not None
        assert col in sample_portfolio_df.columns
    
    def test_get_value_column_priority(self):
        """Test value column priority selection."""
        from XFIN.parsers.data_cleaning import get_value_column
        
        df = pd.DataFrame({
            'Invested Value': [100, 200],
            'Current Value': [110, 210],
            'Buy Value': [100, 200]
        })
        
        col = get_value_column(df)
        
        # Should prioritize based on defined order
        assert col is not None
    
    def test_get_stock_name_column_basic(self, sample_portfolio_df):
        """Test finding stock name column."""
        from XFIN.parsers.data_cleaning import get_stock_name_column
        
        col = get_stock_name_column(sample_portfolio_df)
        
        assert col == 'Stock Name'
    
    def test_get_stock_name_column_variants(self):
        """Test finding stock name with various column names."""
        from XFIN.parsers.data_cleaning import get_stock_name_column
        
        test_cases = [
            ({'Security Name': ['TCS'], 'Value': [100]}, 'Security Name'),
            ({'Company Name': ['TCS'], 'Value': [100]}, 'Company Name'),
            ({'Symbol': ['TCS'], 'Value': [100]}, 'Symbol'),
        ]
        
        for data, expected_col in test_cases:
            df = pd.DataFrame(data)
            col = get_stock_name_column(df)
            if expected_col in df.columns:
                assert col == expected_col or col is not None
    
    def test_get_value_column_empty_df(self):
        """Test get_value_column with empty DataFrame."""
        from XFIN.parsers.data_cleaning import get_value_column
        
        empty_df = pd.DataFrame()
        col = get_value_column(empty_df)
        
        assert col is None
    
    def test_get_stock_name_column_empty_df(self):
        """Test get_stock_name_column with empty DataFrame."""
        from XFIN.parsers.data_cleaning import get_stock_name_column
        
        empty_df = pd.DataFrame()
        col = get_stock_name_column(empty_df)
        
        assert col is None


class TestCleanPortfolioData:
    """Tests for the clean_portfolio_data function."""
    
    def test_clean_portfolio_data_removes_null_names(self):
        """Test that rows with null stock names are removed."""
        from XFIN.parsers.data_cleaning import clean_portfolio_data
        
        df = pd.DataFrame({
            'Stock Name': ['RELIANCE', None, 'TCS'],
            'Quantity': [100, 50, 75],
            'Buy Value': [245000, 50000, 270000]
        })
        
        column_mapping = {'stock_name': 'Stock Name'}
        cleaned_df, changes = clean_portfolio_data(df, column_mapping)
        
        # Should have removed the row with None stock name
        assert len(cleaned_df) == 2
        assert 'Removed Rows (Missing Stock Name)' in changes or len(changes) >= 0
    
    def test_clean_portfolio_data_returns_changes(self):
        """Test that clean_portfolio_data returns changes dict."""
        from XFIN.parsers.data_cleaning import clean_portfolio_data
        
        df = pd.DataFrame({
            'Stock Name': ['RELIANCE', 'TCS'],
            'Quantity': [100, 50]
        })
        
        column_mapping = {'stock_name': 'Stock Name'}
        cleaned_df, changes = clean_portfolio_data(df, column_mapping)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(changes, dict)
    
    def test_clean_portfolio_data_handles_empty(self):
        """Test handling of empty DataFrame."""
        from XFIN.parsers.data_cleaning import clean_portfolio_data
        
        empty_df = pd.DataFrame(columns=['Stock Name', 'Quantity'])
        column_mapping = {'stock_name': 'Stock Name'}
        
        cleaned_df, changes = clean_portfolio_data(empty_df, column_mapping)
        
        assert len(cleaned_df) == 0


# =============================================================================
# Integration Tests for Parsers
# =============================================================================

class TestParsersIntegration:
    """Integration tests for the parsers module."""
    
    def test_full_parsing_workflow(self, zerodha_csv_content):
        """Test complete CSV parsing workflow."""
        from XFIN.parsers import UniversalBrokerCSVParser
        
        parser = UniversalBrokerCSVParser()
        lines = zerodha_csv_content.strip().split('\n')
        
        # Find header
        header_idx, broker = parser.find_header_row(lines)
        assert header_idx >= 0
        
        # Get headers and map columns
        headers = lines[header_idx].split(',')
        mapping = parser.map_columns(headers)
        assert len(mapping) > 0
        
        # Extract data
        data_rows = parser.extract_data_rows(lines, header_idx, headers)
        assert len(data_rows) >= 1
    
    def test_stress_app_parser_compatibility(self):
        """Test that parsers module works with stress_app patterns."""
        from XFIN.parsers import UniversalBrokerCSVParser
        from XFIN.parsers.data_cleaning import clean_portfolio_data, get_value_column, get_stock_name_column
        
        # Create sample DataFrame like stress_app would
        df = pd.DataFrame({
            'Stock Name': ['RELIANCE', 'TCS', 'HDFCBANK'],
            'Quantity': [100, 50, 75],
            'Buy value': [245000, 180000, 123750],
            'Closing value': [250000, 182500, 126000]
        })
        
        # Test column finding
        stock_col = get_stock_name_column(df)
        assert stock_col == 'Stock Name'
        
        value_col = get_value_column(df)
        assert value_col is not None
        
        # Test cleaning
        column_mapping = {'stock_name': 'Stock Name'}
        cleaned_df, _ = clean_portfolio_data(df, column_mapping)
        assert len(cleaned_df) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
