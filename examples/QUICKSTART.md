# XFIN Quick Start (3 Steps)

Get your first portfolio analysis in under 5 minutes!

## Step 1: Install XFIN

```bash
pip install xfin-xai
```

## Step 2: Create Portfolio CSV

Create `my_portfolio.csv`:

```csv
Ticker,Quantity,Current_Price,Sector
RELIANCE.NS,500,2450,Energy
TCS.NS,300,3600,IT
HDFCBANK.NS,1000,1650,Financials
```

## Step 3: Run Analysis

```python
import XFIN as xfin
import pandas as pd

portfolio = pd.read_csv('my_portfolio.csv')

# Stress Test
stress = xfin.StressAnalyzer()
result = stress.explain_stress_impact(portfolio, 'market_correction')
print(f"Impact: {result['impact_percent']:.2f}%")
```

## 🎉 Done!

For more options, see the full [Quick Start Guide](../docs/QUICK_START.md).
