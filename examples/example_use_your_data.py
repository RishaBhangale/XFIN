#!/usr/bin/env python3
"""
XFIN - Use Your Own Data Example
=================================

⭐ START HERE! This example shows how to use XFIN with your own portfolio data.

Supports:
- CSV files from any broker (Zerodha, Groww, Angel One, etc.)
- Excel files (.xlsx)
- Custom data formats

Author: XFIN Team
"""

import pandas as pd
import os

# =============================================================================
# CONFIGURATION - Edit these values!
# =============================================================================

# Path to your portfolio file (CSV or Excel)
CUSTOM_PORTFOLIO_FILE = None  # e.g., 'my_portfolio.csv' or '/path/to/data.xlsx'

# Optional: OpenRouter API key for AI-powered insights
OPENROUTER_API_KEY = None  # e.g., 'sk-or-v1-...'

# =============================================================================
# XFIN IMPORTS
# =============================================================================

try:
    from XFIN import StressTestingEngine, ESGScoringEngine
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from XFIN import StressTestingEngine, ESGScoringEngine


def load_portfolio(file_path: str = None) -> pd.DataFrame:
    """
    Load portfolio data from file or create sample data.
    
    XFIN supports flexible column names:
    - Stock: 'Stock Name', 'Symbol', 'Scrip', 'Name', 'Security'
    - Quantity: 'Quantity', 'Qty', 'Units', 'Shares'
    - Value: 'Current Value', 'Value', 'Amount', 'Market Value'
    """
    
    if file_path and os.path.exists(file_path):
        print(f"📂 Loading portfolio from: {file_path}")
        
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"   Loaded {len(df)} holdings")
        return df
    
    # Create sample portfolio if no file provided
    print("📊 Using sample portfolio data...")
    print("   (Set CUSTOM_PORTFOLIO_FILE to use your own data)")
    
    return pd.DataFrame({
        'Stock Name': [
            'RELIANCE', 'TCS', 'HDFC BANK', 'INFOSYS', 'ICICI BANK',
            'BHARTI AIRTEL', 'ITC', 'KOTAK BANK', 'AXIS BANK', 'SBI'
        ],
        'Quantity': [100, 50, 75, 60, 80, 120, 200, 40, 90, 150],
        'Buy Price': [2500, 3500, 1600, 1400, 950, 800, 450, 1800, 1000, 600],
        'Current Price': [2650, 3700, 1550, 1480, 1020, 850, 470, 1750, 1050, 650],
        'Current Value': [265000, 185000, 116250, 88800, 81600, 102000, 94000, 70000, 94500, 97500]
    })


def run_stress_analysis(portfolio_df: pd.DataFrame, api_key: str = None):
    """Run comprehensive stress testing on portfolio."""
    
    print("\n" + "=" * 60)
    print("  🧪 STRESS TESTING ANALYSIS")
    print("=" * 60)
    
    # Initialize engine
    engine = StressTestingEngine(api_key=api_key)
    
    # Run multiple scenarios
    scenarios = ['market_correction', 'recession', 'interest_rate_shock']
    
    print(f"\n📋 Running {len(scenarios)} stress scenarios...")
    
    result = engine.run_stress_test(
        portfolio_df=portfolio_df,
        scenarios=scenarios
    )
    
    if result and 'scenario_results' in result:
        print("\n📊 Results Summary:")
        print("-" * 50)
        
        for scenario_name, scenario_result in result['scenario_results'].items():
            impact = scenario_result.get('impact_percentage', 0)
            stressed = scenario_result.get('stressed_value', 0)
            
            # Risk indicator
            if abs(impact) < 10:
                indicator = "🟢"
            elif abs(impact) < 20:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            print(f"\n  {indicator} {scenario_name}")
            print(f"     Impact: {impact:+.2f}%")
            print(f"     Stressed Value: ₹{stressed:,.0f}")
    
    return result


def run_esg_analysis(portfolio_df: pd.DataFrame, api_key: str = None):
    """Run ESG analysis on portfolio."""
    
    print("\n" + "=" * 60)
    print("  🌱 ESG ANALYSIS")
    print("=" * 60)
    
    # Initialize engine
    engine = ESGScoringEngine(api_key=api_key)
    
    print("\n📊 Analyzing portfolio ESG scores...")
    
    result = engine.calculate_portfolio_esg(portfolio_df)
    
    if result:
        # Overall scores
        if 'portfolio_score' in result:
            score = result['portfolio_score']
            stars = result.get('star_rating', 3)
            print(f"\n  Overall ESG Score: {score:.1f}/100")
            print(f"  Rating: {'⭐' * stars}")
        
        # Component scores
        if 'component_scores' in result:
            print("\n  Component Breakdown:")
            for component, score in result['component_scores'].items():
                print(f"    {component}: {score:.1f}")
        
        # Coverage
        if 'coverage' in result:
            print(f"\n  ESG Data Coverage: {result['coverage']:.1f}%")
    
    return result


def main():
    """Main entry point."""
    
    print("\n" + "=" * 60)
    print("  XFIN - Comprehensive Portfolio Analysis")
    print("=" * 60)
    
    # Load portfolio
    portfolio = load_portfolio(CUSTOM_PORTFOLIO_FILE)
    
    # Display portfolio summary
    total_value = portfolio['Current Value'].sum() if 'Current Value' in portfolio.columns else 0
    print(f"\n💼 Portfolio Summary:")
    print(f"   Holdings: {len(portfolio)}")
    print(f"   Total Value: ₹{total_value:,.0f}")
    
    # Run analyses
    stress_result = run_stress_analysis(portfolio, OPENROUTER_API_KEY)
    esg_result = run_esg_analysis(portfolio, OPENROUTER_API_KEY)
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ Analysis Complete!")
    print("=" * 60)
    
    print("\n📚 To use your own data:")
    print("   1. Set CUSTOM_PORTFOLIO_FILE = 'your_file.csv'")
    print("   2. Ensure columns include: Stock Name, Quantity, Current Value")
    print("   3. Optionally set OPENROUTER_API_KEY for AI insights")
    
    print("\n🎯 Try interactive dashboards:")
    print("   xfin stress  - Stress Testing Dashboard")
    print("   xfin esg     - ESG Analysis Dashboard")


if __name__ == "__main__":
    main()
