#!/usr/bin/env python3
"""
XFIN Basic Usage Example
========================

A simple introduction to XFIN's core functionality.
This example demonstrates basic usage without external API keys.

Author: XFIN Team
"""

import pandas as pd

# =============================================================================
# XFIN IMPORTS
# =============================================================================

try:
    from XFIN import StressTestingEngine, ESGScoringEngine
    from XFIN.stress_testing import ScenarioGenerator
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from XFIN import StressTestingEngine, ESGScoringEngine
    from XFIN.stress_testing import ScenarioGenerator


def main():
    """Basic XFIN usage demonstration."""
    
    print("=" * 60)
    print("  XFIN Basic Usage Example")
    print("=" * 60)
    
    # =========================================================================
    # 1. Create Sample Portfolio Data
    # =========================================================================
    
    print("\n📊 Step 1: Creating sample portfolio...")
    
    portfolio_data = pd.DataFrame({
        'Stock Name': ['RELIANCE', 'TCS', 'HDFC BANK', 'INFOSYS', 'ICICI BANK'],
        'Quantity': [100, 50, 75, 60, 80],
        'Buy Price': [2500, 3500, 1600, 1400, 950],
        'Current Price': [2650, 3700, 1550, 1480, 1020],
    })
    
    # Calculate current values
    portfolio_data['Current Value'] = (
        portfolio_data['Quantity'] * portfolio_data['Current Price']
    )
    
    print(portfolio_data.to_string(index=False))
    print(f"\n💰 Total Portfolio Value: ₹{portfolio_data['Current Value'].sum():,.0f}")
    
    # =========================================================================
    # 2. List Available Stress Scenarios
    # =========================================================================
    
    print("\n\n📋 Step 2: Available Stress Testing Scenarios")
    print("-" * 40)
    
    scenario_gen = ScenarioGenerator()
    scenarios = scenario_gen.list_scenarios()
    
    for i, scenario in enumerate(scenarios[:5], 1):
        print(f"  {i}. {scenario}")
    print(f"  ... and {len(scenarios) - 5} more")
    
    # =========================================================================
    # 3. Run Basic Stress Test
    # =========================================================================
    
    print("\n\n🧪 Step 3: Running Stress Test (Market Correction)")
    print("-" * 40)
    
    # Initialize engine
    engine = StressTestingEngine()
    
    # Run stress test
    result = engine.run_stress_test(
        portfolio_df=portfolio_data,
        scenarios=['market_correction']
    )
    
    # Display results
    if result and 'scenario_results' in result:
        for scenario_name, scenario_result in result['scenario_results'].items():
            print(f"\n  Scenario: {scenario_name}")
            if 'impact_percentage' in scenario_result:
                print(f"  📉 Expected Impact: {scenario_result['impact_percentage']:.2f}%")
            if 'stressed_value' in scenario_result:
                print(f"  💵 Stressed Value: ₹{scenario_result['stressed_value']:,.0f}")
    
    # =========================================================================
    # 4. Basic ESG Check
    # =========================================================================
    
    print("\n\n🌱 Step 4: Basic ESG Overview")
    print("-" * 40)
    
    esg_engine = ESGScoringEngine()
    
    # Score a single security
    security_score = esg_engine.score_security({
        'name': 'RELIANCE',
        'sector': 'Energy',
        'market_cap': 1500000
    })
    
    if security_score:
        print(f"  Company: RELIANCE")
        if 'overall_score' in security_score:
            print(f"  ESG Score: {security_score['overall_score']:.1f}/100")
        if 'star_rating' in security_score:
            print(f"  Rating: {'⭐' * security_score['star_rating']}")
    
    # =========================================================================
    # 5. Summary
    # =========================================================================
    
    print("\n\n" + "=" * 60)
    print("  ✅ Basic Example Complete!")
    print("=" * 60)
    print("\n📚 Next Steps:")
    print("  1. Try example_use_your_data.py with your own portfolio")
    print("  2. Explore example_stress_testing_code.py for advanced stress tests")
    print("  3. Check example_esg_analysis_code.py for ESG deep dives")
    print("  4. Run 'xfin stress' or 'xfin esg' for interactive dashboards")
    

if __name__ == "__main__":
    main()
