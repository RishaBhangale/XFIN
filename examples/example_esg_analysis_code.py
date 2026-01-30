#!/usr/bin/env python3
"""
XFIN ESG Analysis Deep Dive
============================

Comprehensive ESG (Environmental, Social, Governance) analysis example.

Features demonstrated:
- Single security ESG scoring
- Portfolio-level ESG analysis
- Sector breakdown
- Star rating system
- ESG data coverage tracking

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
    from XFIN import ESGScoringEngine
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from XFIN import ESGScoringEngine


def create_sample_portfolio() -> pd.DataFrame:
    """Create a diversified sample portfolio for ESG analysis."""
    return pd.DataFrame({
        'Stock Name': [
            'RELIANCE', 'TCS', 'HDFC BANK', 'INFOSYS', 'ICICI BANK',
            'BHARTI AIRTEL', 'ITC', 'HINDUSTAN UNILEVER', 'SUN PHARMA',
            'ASIAN PAINTS', 'TATA MOTORS', 'ADANI GREEN', 'POWER GRID',
            'NTPC', 'VEDANTA'
        ],
        'Quantity': [100, 50, 75, 60, 80, 120, 200, 30, 80, 40, 150, 200, 300, 250, 100],
        'Buy Price': [2500, 3500, 1600, 1400, 950, 800, 450, 2500, 1100, 3200, 700, 1500, 240, 180, 280],
        'Current Price': [2650, 3700, 1550, 1480, 1020, 850, 470, 2650, 1150, 3400, 750, 1600, 250, 190, 300],
        'Sector': [
            'Energy', 'Technology', 'Banking', 'Technology', 'Banking',
            'Telecom', 'Consumer', 'Consumer', 'Pharma', 'Consumer',
            'Auto', 'Green Energy', 'Utilities', 'Utilities', 'Mining'
        ]
    })


def score_single_security(engine: ESGScoringEngine, name: str, sector: str):
    """Score a single security."""
    
    print(f"\n  Analyzing: {name}")
    
    result = engine.score_security({
        'name': name,
        'sector': sector,
        'market_cap': 100000  # Default
    })
    
    if result:
        score = result.get('overall_score', 0)
        stars = result.get('star_rating', 0)
        
        print(f"    ESG Score: {score:.1f}/100")
        print(f"    Rating: {'⭐' * stars} ({stars}/5)")
        
        # Component scores
        if 'component_scores' in result:
            print("    Components:")
            for comp, val in result['component_scores'].items():
                print(f"      {comp}: {val:.1f}")
    
    return result


def analyze_portfolio_esg(portfolio: pd.DataFrame, engine: ESGScoringEngine):
    """Run comprehensive portfolio ESG analysis."""
    
    print("\n" + "=" * 60)
    print("  📊 PORTFOLIO ESG ANALYSIS")
    print("=" * 60)
    
    # Calculate values
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    total_value = portfolio['Current Value'].sum()
    
    print(f"\n  💼 Portfolio: {len(portfolio)} holdings")
    print(f"  💵 Total Value: ₹{total_value:,.0f}")
    
    # Run ESG analysis
    result = engine.calculate_portfolio_esg(portfolio)
    
    if result:
        # Overall score
        score = result.get('portfolio_score', 0)
        stars = result.get('star_rating', 3)
        
        print(f"\n  🌱 Portfolio ESG Score: {score:.1f}/100")
        print(f"  ⭐ Rating: {'⭐' * stars} ({get_rating_label(stars)})")
        
        # Component breakdown
        if 'component_scores' in result:
            print("\n  📊 ESG Component Breakdown:")
            components = result['component_scores']
            for name, val in components.items():
                bar = "█" * int(val / 5)
                print(f"     {name:>12}: {bar} {val:.1f}")
        
        # Coverage
        coverage = result.get('coverage', 0)
        print(f"\n  📈 Data Coverage: {coverage:.1f}%")
        
        if coverage < 50:
            print("     ⚠️  Low coverage - results use sector proxies")
        elif coverage < 80:
            print("     🟡 Moderate coverage")
        else:
            print("     🟢 Good data coverage")
    
    return result


def get_rating_label(stars: int) -> str:
    """Get text label for star rating."""
    labels = {
        5: "Leader",
        4: "Strong",
        3: "Average",
        2: "Below Average",
        1: "Laggard"
    }
    return labels.get(stars, "Unknown")


def analyze_sector_breakdown(portfolio: pd.DataFrame, engine: ESGScoringEngine):
    """Analyze ESG scores by sector."""
    
    print("\n" + "=" * 60)
    print("  🏭 SECTOR ESG BREAKDOWN")
    print("=" * 60)
    
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    
    if 'Sector' not in portfolio.columns:
        print("\n  ⚠️  No sector data available")
        return
    
    sectors = portfolio['Sector'].unique()
    sector_scores = []
    
    for sector in sectors:
        sector_df = portfolio[portfolio['Sector'] == sector]
        sector_value = sector_df['Current Value'].sum()
        
        # Get average ESG for sector
        result = engine.apply_sector_proxy(sector, sector_value)
        if result:
            sector_scores.append({
                'Sector': sector,
                'Value': sector_value,
                'ESG Score': result.get('overall_score', 50)
            })
    
    if sector_scores:
        scores_df = pd.DataFrame(sector_scores).sort_values('ESG Score', ascending=False)
        
        print("\n  Sector ESG Rankings:")
        print("  " + "-" * 45)
        
        for _, row in scores_df.iterrows():
            score = row['ESG Score']
            
            # Color code
            if score >= 60:
                indicator = "🟢"
            elif score >= 40:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            bar = "█" * int(score / 5)
            print(f"  {indicator} {row['Sector']:<15} {bar} {score:.0f}")


def identify_esg_risks_and_opportunities(portfolio: pd.DataFrame, engine: ESGScoringEngine):
    """Identify ESG risks and opportunities in portfolio."""
    
    print("\n" + "=" * 60)
    print("  ⚠️  ESG RISKS & OPPORTUNITIES")
    print("=" * 60)
    
    portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
    total_value = portfolio['Current Value'].sum()
    portfolio['Weight'] = portfolio['Current Value'] / total_value * 100
    
    high_risk_sectors = ['Mining', 'Oil & Gas', 'Tobacco', 'Coal']
    opportunity_sectors = ['Green Energy', 'Technology', 'Healthcare', 'Renewable']
    
    # Find risks
    risks = portfolio[portfolio['Sector'].isin(high_risk_sectors)]
    opportunities = portfolio[portfolio['Sector'].isin(opportunity_sectors)]
    
    print("\n  🔴 ESG Risk Holdings:")
    if len(risks) > 0:
        for _, row in risks.iterrows():
            print(f"     {row['Stock Name']}: {row['Weight']:.1f}% ({row['Sector']})")
    else:
        print("     No high-risk holdings identified ✓")
    
    print("\n  🟢 ESG Opportunity Holdings:")
    if len(opportunities) > 0:
        for _, row in opportunities.iterrows():
            print(f"     {row['Stock Name']}: {row['Weight']:.1f}% ({row['Sector']})")
    else:
        print("     Consider adding green/sustainable investments")
    
    # Recommendations
    print("\n  💡 Recommendations:")
    if len(risks) > 0:
        risk_weight = risks['Weight'].sum()
        print(f"     • High-risk sectors: {risk_weight:.1f}% of portfolio")
        print(f"     • Consider reducing exposure to improve ESG profile")
    
    if len(opportunities) == 0:
        print("     • No green energy exposure detected")
        print("     • Consider adding SFDR Article 8/9 compliant funds")


def main():
    """Main entry point."""
    
    print("\n" + "=" * 60)
    print("  XFIN ESG Analysis - Deep Dive")
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
    
    # Initialize ESG engine
    engine = ESGScoringEngine(api_key=OPENROUTER_API_KEY)
    
    # Individual security analysis
    print("\n" + "=" * 60)
    print("  🔍 INDIVIDUAL SECURITY ESG SCORES")
    print("=" * 60)
    
    # Score top 3 holdings
    for i in range(min(3, len(portfolio))):
        row = portfolio.iloc[i]
        score_single_security(
            engine,
            row['Stock Name'],
            row.get('Sector', 'Unknown')
        )
    
    # Portfolio analysis
    analyze_portfolio_esg(portfolio, engine)
    analyze_sector_breakdown(portfolio, engine)
    identify_esg_risks_and_opportunities(portfolio, engine)
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ ESG Analysis Complete!")
    print("=" * 60)
    print("\n🎯 Launch interactive dashboard: xfin esg")


if __name__ == "__main__":
    main()
