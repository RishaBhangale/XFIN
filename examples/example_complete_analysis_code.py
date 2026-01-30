#!/usr/bin/env python3
"""
XFIN Complete Analysis Example
===============================

Demonstrates using all XFIN modules together:
- Stress Testing
- ESG Analysis
- Combined Risk Assessment

This shows how to get a comprehensive view of portfolio risk.

Author: XFIN Team
"""

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

CUSTOM_PORTFOLIO_FILE = None  # Your portfolio CSV/Excel file
OPENROUTER_API_KEY = None     # For AI-powered insights (optional)

# =============================================================================
# XFIN IMPORTS
# =============================================================================

try:
    from XFIN import StressTestingEngine, ESGScoringEngine
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from XFIN import StressTestingEngine, ESGScoringEngine


def create_sample_portfolio() -> pd.DataFrame:
    """Create sample portfolio."""
    return pd.DataFrame({
        'Stock Name': [
            'RELIANCE', 'TCS', 'HDFC BANK', 'INFOSYS', 'ICICI BANK',
            'BHARTI AIRTEL', 'ITC', 'SUN PHARMA', 'HINDUSTAN UNILEVER',
            'ADANI GREEN'
        ],
        'Quantity': [100, 50, 75, 60, 80, 120, 200, 80, 30, 200],
        'Buy Price': [2500, 3500, 1600, 1400, 950, 800, 450, 1100, 2500, 1500],
        'Current Price': [2650, 3700, 1550, 1480, 1020, 850, 470, 1150, 2650, 1600],
        'Sector': [
            'Energy', 'Technology', 'Banking', 'Technology', 'Banking',
            'Telecom', 'Consumer', 'Pharma', 'Consumer', 'Green Energy'
        ]
    })


def run_stress_analysis(portfolio: pd.DataFrame) -> dict:
    """Run stress testing analysis."""
    
    print("\n  🧪 Running Stress Tests...")
    
    engine = StressTestingEngine(api_key=OPENROUTER_API_KEY)
    
    result = engine.run_stress_test(
        portfolio_df=portfolio,
        scenarios=['market_correction', 'recession', 'interest_rate_shock']
    )
    
    return result


def run_esg_analysis(portfolio: pd.DataFrame) -> dict:
    """Run ESG analysis."""
    
    print("\n  🌱 Running ESG Analysis...")
    
    engine = ESGScoringEngine(api_key=OPENROUTER_API_KEY)
    result = engine.calculate_portfolio_esg(portfolio)
    
    return result


def calculate_combined_risk_score(stress_result: dict, esg_result: dict) -> dict:
    """
    Calculate a combined risk score incorporating both stress and ESG factors.
    
    This is a simplified model. Real implementations would use more sophisticated
    weighting and factor integration.
    """
    
    # Stress risk component (0-100, higher = more risk)
    stress_risk = 50  # Default
    if stress_result and 'scenario_results' in stress_result:
        impacts = [
            abs(r.get('impact_percentage', 0))
            for r in stress_result['scenario_results'].values()
        ]
        avg_impact = sum(impacts) / len(impacts) if impacts else 0
        stress_risk = min(100, avg_impact * 3)  # Scale to 0-100
    
    # ESG risk component (0-100, lower ESG = higher risk)
    esg_risk = 50  # Default
    if esg_result:
        esg_score = esg_result.get('portfolio_score', 50)
        esg_risk = 100 - esg_score  # Invert: high ESG = low risk
    
    # Combined score (simple average)
    combined_risk = (stress_risk * 0.6 + esg_risk * 0.4)
    
    # Risk level
    if combined_risk < 30:
        level = "LOW"
        indicator = "🟢"
    elif combined_risk < 50:
        level = "MODERATE"
        indicator = "🟡"
    elif combined_risk < 70:
        level = "HIGH"
        indicator = "🟠"
    else:
        level = "SEVERE"
        indicator = "🔴"
    
    return {
        'stress_risk': stress_risk,
        'esg_risk': esg_risk,
        'combined_risk': combined_risk,
        'risk_level': level,
        'indicator': indicator
    }


def generate_report(
    portfolio: pd.DataFrame,
    stress_result: dict,
    esg_result: dict,
    combined_risk: dict
):
    """Generate comprehensive analysis report."""
    
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    total_value = portfolio['Current Value'].sum()
    
    print("\n" + "=" * 60)
    print("  📋 COMPREHENSIVE PORTFOLIO REPORT")
    print("=" * 60)
    
    # Portfolio Overview
    print("\n  💼 PORTFOLIO OVERVIEW")
    print("  " + "-" * 40)
    print(f"     Total Holdings: {len(portfolio)}")
    print(f"     Total Value: ₹{total_value:,.0f}")
    
    # Stress Testing Summary
    print("\n  🧪 STRESS TESTING SUMMARY")
    print("  " + "-" * 40)
    
    if stress_result and 'scenario_results' in stress_result:
        for scenario, data in stress_result['scenario_results'].items():
            impact = data.get('impact_percentage', 0)
            print(f"     {scenario}: {impact:+.2f}%")
    
    # ESG Summary
    print("\n  🌱 ESG SUMMARY")
    print("  " + "-" * 40)
    
    if esg_result:
        score = esg_result.get('portfolio_score', 0)
        stars = esg_result.get('star_rating', 3)
        coverage = esg_result.get('coverage', 0)
        
        print(f"     ESG Score: {score:.1f}/100")
        print(f"     Rating: {'⭐' * stars}")
        print(f"     Data Coverage: {coverage:.1f}%")
    
    # Combined Risk Assessment
    print("\n  ⚡ COMBINED RISK ASSESSMENT")
    print("  " + "-" * 40)
    print(f"     Stress Risk Component: {combined_risk['stress_risk']:.1f}")
    print(f"     ESG Risk Component: {combined_risk['esg_risk']:.1f}")
    print(f"     Combined Risk Score: {combined_risk['combined_risk']:.1f}")
    print(f"     Risk Level: {combined_risk['indicator']} {combined_risk['risk_level']}")
    
    # Recommendations
    print("\n  💡 RECOMMENDATIONS")
    print("  " + "-" * 40)
    
    if combined_risk['combined_risk'] > 50:
        print("     • Consider diversifying to reduce concentration risk")
        print("     • Review holdings with high stress sensitivity")
    
    if combined_risk['esg_risk'] > 40:
        print("     • Consider improving ESG profile")
        print("     • Look into sustainable investment alternatives")
    
    if combined_risk['combined_risk'] < 30:
        print("     • Portfolio shows good resilience")
        print("     • Continue monitoring for changes")


def main():
    """Main entry point."""
    
    print("\n" + "=" * 60)
    print("  XFIN Complete Portfolio Analysis")
    print("=" * 60)
    
    # Load portfolio
    if CUSTOM_PORTFOLIO_FILE:
        import os
        if os.path.exists(CUSTOM_PORTFOLIO_FILE):
            portfolio = pd.read_csv(CUSTOM_PORTFOLIO_FILE)
        else:
            portfolio = create_sample_portfolio()
    else:
        portfolio = create_sample_portfolio()
    
    # Calculate current values
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    total = portfolio['Current Value'].sum()
    
    print(f"\n  📊 Analyzing portfolio: {len(portfolio)} holdings, ₹{total:,.0f}")
    
    # Run analyses
    stress_result = run_stress_analysis(portfolio)
    esg_result = run_esg_analysis(portfolio)
    
    # Calculate combined risk
    combined_risk = calculate_combined_risk_score(stress_result, esg_result)
    
    # Generate report
    generate_report(portfolio, stress_result, esg_result, combined_risk)
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ Complete Analysis Done!")
    print("=" * 60)
    print("\n🎯 Interactive Dashboards:")
    print("   xfin stress  - Stress Testing")
    print("   xfin esg     - ESG Analysis")


if __name__ == "__main__":
    main()
