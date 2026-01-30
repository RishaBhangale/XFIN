#!/usr/bin/env python3
"""
XFIN Stress Testing Deep Dive
==============================

Comprehensive stress testing example demonstrating:
- Multiple stress scenarios
- Custom scenario creation
- Portfolio risk metrics
- Recovery timeline analysis
- VaR calculations

Author: XFIN Team
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

CUSTOM_PORTFOLIO_FILE = None  # Your portfolio CSV/Excel file
OPENROUTER_API_KEY = None     # For AI-powered insights (optional)

# =============================================================================
# XFIN IMPORTS
# =============================================================================

try:
    from XFIN import StressTestingEngine
    from XFIN.stress_testing import ScenarioGenerator, PortfolioAnalyzer
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from XFIN import StressTestingEngine
    from XFIN.stress_testing import ScenarioGenerator, PortfolioAnalyzer


def create_sample_portfolio() -> pd.DataFrame:
    """Create a diversified sample portfolio."""
    return pd.DataFrame({
        'Stock Name': [
            # Technology
            'TCS', 'INFOSYS', 'WIPRO', 'HCLTECH',
            # Banking
            'HDFC BANK', 'ICICI BANK', 'AXIS BANK', 'KOTAK BANK',
            # Energy
            'RELIANCE', 'ONGC', 'IOC',
            # Consumer
            'HINDUSTAN UNILEVER', 'ITC', 'NESTLE',
            # Pharma
            'SUN PHARMA', 'DR REDDY'
        ],
        'Quantity': [50, 60, 100, 80, 75, 80, 90, 40, 100, 150, 200, 30, 200, 20, 80, 50],
        'Buy Price': [3500, 1400, 450, 1200, 1600, 950, 1000, 1800, 2500, 180, 90, 2500, 450, 22000, 1100, 5500],
        'Current Price': [3700, 1480, 470, 1250, 1550, 1020, 1050, 1750, 2650, 190, 95, 2650, 470, 23500, 1150, 5800],
        'Sector': [
            'Technology', 'Technology', 'Technology', 'Technology',
            'Banking', 'Banking', 'Banking', 'Banking',
            'Energy', 'Energy', 'Energy',
            'Consumer', 'Consumer', 'Consumer',
            'Pharma', 'Pharma'
        ]
    })


def explore_scenarios():
    """Explore available stress testing scenarios."""
    
    print("\n" + "=" * 60)
    print("  📋 AVAILABLE STRESS SCENARIOS")
    print("=" * 60)
    
    gen = ScenarioGenerator()
    scenarios = gen.list_scenarios()
    
    print(f"\n  Found {len(scenarios)} pre-configured scenarios:\n")
    
    for scenario in scenarios:
        details = gen.get_scenario(scenario)
        if details:
            severity = details.get('severity', 'Unknown')
            print(f"  • {scenario}")
            print(f"    Severity: {severity}")
    
    return scenarios


def run_single_scenario(portfolio: pd.DataFrame, scenario_name: str):
    """Run a single stress scenario with detailed output."""
    
    print(f"\n\n🔬 Detailed Analysis: {scenario_name}")
    print("-" * 50)
    
    engine = StressTestingEngine(api_key=OPENROUTER_API_KEY)
    
    result = engine.run_stress_test(
        portfolio_df=portfolio,
        scenarios=[scenario_name]
    )
    
    if result and 'scenario_results' in result:
        scenario_result = result['scenario_results'].get(scenario_name, {})
        
        print(f"\n  📉 Expected Impact: {scenario_result.get('impact_percentage', 0):+.2f}%")
        print(f"  💵 Original Value: ₹{result.get('original_value', 0):,.0f}")
        print(f"  💵 Stressed Value: ₹{scenario_result.get('stressed_value', 0):,.0f}")
        
        # Risk level
        impact = abs(scenario_result.get('impact_percentage', 0))
        if impact < 10:
            risk = "LOW 🟢"
        elif impact < 20:
            risk = "MODERATE 🟡"
        elif impact < 30:
            risk = "HIGH 🟠"
        else:
            risk = "SEVERE 🔴"
        print(f"  ⚠️  Risk Level: {risk}")
        
        # Recovery info
        if 'recovery_months' in scenario_result:
            print(f"  🔄 Est. Recovery: {scenario_result['recovery_months']} months")
    
    return result


def run_multi_scenario_comparison(portfolio: pd.DataFrame):
    """Compare portfolio impact across multiple scenarios."""
    
    print("\n" + "=" * 60)
    print("  📊 MULTI-SCENARIO COMPARISON")
    print("=" * 60)
    
    engine = StressTestingEngine(api_key=OPENROUTER_API_KEY)
    
    # Test various severity levels
    scenarios = [
        'market_correction',      # Mild
        'recession',              # Moderate
        'interest_rate_shock',    # Sector-specific
        'global_crisis',          # Severe
        'black_swan'              # Extreme
    ]
    
    result = engine.run_stress_test(
        portfolio_df=portfolio,
        scenarios=scenarios
    )
    
    if result and 'scenario_results' in result:
        print("\n  Scenario Comparison:")
        print("  " + "-" * 45)
        print(f"  {'Scenario':<25} {'Impact':>10} {'Risk':>8}")
        print("  " + "-" * 45)
        
        for scenario, data in result['scenario_results'].items():
            impact = data.get('impact_percentage', 0)
            
            if abs(impact) < 10:
                risk = "🟢"
            elif abs(impact) < 20:
                risk = "🟡"
            elif abs(impact) < 30:
                risk = "🟠"
            else:
                risk = "🔴"
            
            print(f"  {scenario:<25} {impact:>+9.2f}%   {risk}")
    
    return result


def analyze_portfolio_concentration(portfolio: pd.DataFrame):
    """Analyze portfolio concentration and diversification."""
    
    print("\n" + "=" * 60)
    print("  🎯 PORTFOLIO CONCENTRATION ANALYSIS")
    print("=" * 60)
    
    # Calculate current values
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    total_value = portfolio['Current Value'].sum()
    
    # Sector concentration
    if 'Sector' in portfolio.columns:
        sector_values = portfolio.groupby('Sector')['Current Value'].sum()
        sector_pct = (sector_values / total_value * 100).sort_values(ascending=False)
        
        print("\n  Sector Allocation:")
        for sector, pct in sector_pct.items():
            bar = "█" * int(pct / 5)
            print(f"    {sector:<20} {bar} {pct:.1f}%")
    
    # Top holdings
    portfolio['Weight'] = portfolio['Current Value'] / total_value * 100
    top_5 = portfolio.nlargest(5, 'Current Value')
    
    print("\n  Top 5 Holdings:")
    for _, row in top_5.iterrows():
        print(f"    {row['Stock Name']:<20} {row['Weight']:.1f}%")
    
    # Concentration metrics
    top_5_weight = top_5['Weight'].sum()
    print(f"\n  📊 Concentration Metrics:")
    print(f"     Top 5 holdings: {top_5_weight:.1f}% of portfolio")
    
    if top_5_weight > 50:
        print("     ⚠️  High concentration risk")
    elif top_5_weight > 30:
        print("     🟡 Moderate concentration")
    else:
        print("     🟢 Well diversified")


def main():
    """Main entry point."""
    
    print("\n" + "=" * 60)
    print("  XFIN Stress Testing - Deep Dive")
    print("=" * 60)
    
    # Load or create portfolio
    if CUSTOM_PORTFOLIO_FILE:
        import os
        if os.path.exists(CUSTOM_PORTFOLIO_FILE):
            portfolio = pd.read_csv(CUSTOM_PORTFOLIO_FILE)
        else:
            print(f"⚠️  File not found: {CUSTOM_PORTFOLIO_FILE}")
            portfolio = create_sample_portfolio()
    else:
        portfolio = create_sample_portfolio()
    
    # Display portfolio value
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    total = portfolio['Current Value'].sum()
    print(f"\n💼 Portfolio: {len(portfolio)} holdings, ₹{total:,.0f} total")
    
    # Run analyses
    explore_scenarios()
    run_single_scenario(portfolio, 'market_correction')
    run_multi_scenario_comparison(portfolio)
    analyze_portfolio_concentration(portfolio)
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ Stress Testing Analysis Complete!")
    print("=" * 60)
    print("\n🎯 Launch interactive dashboard: xfin stress")


if __name__ == "__main__":
    main()
