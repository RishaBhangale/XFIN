"""
XFIN Test Configuration and Fixtures
=====================================

Shared fixtures and test utilities for the XFIN test suite.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


# =============================================================================
# Sample Portfolio Fixtures
# =============================================================================

@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio DataFrame for testing."""
    return pd.DataFrame({
        'Ticker': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'WIPRO.NS'],
        'Quantity': [500, 300, 1000, 400, 600],
        'Current_Price': [2450.0, 3600.0, 1650.0, 1500.0, 400.0],
        'Sector': ['Energy', 'IT', 'Financials', 'IT', 'IT']
    })


@pytest.fixture
def sample_portfolio_with_value(sample_portfolio):
    """Portfolio with pre-calculated Current_Value."""
    df = sample_portfolio.copy()
    df['Current_Value'] = df['Quantity'] * df['Current_Price']
    return df


@pytest.fixture
def minimal_portfolio():
    """Minimal single-stock portfolio for edge case testing."""
    return pd.DataFrame({
        'Ticker': ['TCS.NS'],
        'Quantity': [100],
        'Current_Price': [3600.0],
        'Sector': ['IT']
    })


@pytest.fixture
def diversified_portfolio():
    """Diversified portfolio across multiple sectors."""
    return pd.DataFrame({
        'Ticker': [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'SUNPHARMA.NS',
            'MARUTI.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS', 'COALINDIA.NS'
        ],
        'Quantity': [100, 100, 100, 100, 100, 100, 100, 100],
        'Current_Price': [2450, 3600, 1650, 1200, 9500, 2600, 950, 380],
        'Sector': [
            'Energy', 'IT', 'Financials', 'Healthcare',
            'Consumer Discretionary', 'Consumer Staples', 
            'Communication Services', 'Materials'
        ]
    })


# =============================================================================
# Security Data Fixtures
# =============================================================================

@pytest.fixture
def sample_security_data():
    """Sample security data for ESG scoring."""
    return {
        'name': 'Reliance Industries',
        'ticker': 'RELIANCE.NS',
        'sector': 'Energy',
        'market_cap': 15000000000000  # 15 lakh crore
    }


# =============================================================================
# Stress Scenario Fixtures
# =============================================================================

@pytest.fixture
def scenario_names():
    """List of common scenario names for testing."""
    return [
        'market_correction',
        'recession_scenario',
        'credit_crisis',
        'geopolitical_shock'
    ]


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_api_response():
    """Mock API response for LLM calls."""
    return {
        'choices': [{
            'message': {
                'content': 'This is a mock LLM response for testing.'
            }
        }]
    }


@pytest.fixture
def mock_esg_data():
    """Mock ESG data for testing."""
    return {
        'environmental_score': 65.0,
        'social_score': 70.0,
        'governance_score': 75.0,
        'overall_score': 70.0,
        'data_source': 'mock'
    }


# =============================================================================
# Test Utilities
# =============================================================================

def assert_portfolio_columns(df, required_cols=None):
    """Assert that a DataFrame has required portfolio columns."""
    if required_cols is None:
        required_cols = ['Ticker', 'Quantity', 'Current_Price', 'Sector']
    
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def assert_valid_esg_score(score_dict):
    """Assert that an ESG score dictionary has valid structure."""
    required_keys = ['overall_score', 'environmental_score', 'social_score', 'governance_score']
    
    for key in required_keys:
        assert key in score_dict, f"Missing ESG score key: {key}"
        assert 0 <= score_dict[key] <= 100, f"Invalid score range for {key}: {score_dict[key]}"


def assert_valid_stress_result(result_dict):
    """Assert that a stress test result has valid structure."""
    required_keys = ['portfolio_value', 'stressed_value', 'impact_percent']
    
    for key in required_keys:
        assert key in result_dict, f"Missing stress result key: {key}"
