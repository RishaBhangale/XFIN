# XFIN Quick Start Guide

Get up and running with XFIN in 5 minutes!

## 📦 Installation

```bash
pip install xfin-xai
```

Or install from source:

```bash
git clone https://github.com/RishaBhangale/XFIN.git
cd XFIN
pip install -e .
```

## 🚀 Three Steps to Your First Analysis

### Step 1: Prepare Your Portfolio

Create a CSV file (`my_portfolio.csv`):

```csv
Ticker,Quantity,Current_Price,Sector
RELIANCE.NS,500,2450,Energy
TCS.NS,300,3600,IT
HDFCBANK.NS,1000,1650,Financials
INFY.NS,400,1500,IT
```

### Step 2: Run Your First Stress Test

```python
import XFIN as xfin
import pandas as pd

# Load portfolio
portfolio = pd.read_csv('my_portfolio.csv')

# Create stress analyzer
stress = xfin.StressAnalyzer()

# Run stress test
result = stress.explain_stress_impact(portfolio, 'market_correction')

print(f"Portfolio Value: ₹{result['portfolio_value']:,.2f}")
print(f"Stressed Value: ₹{result['stressed_value']:,.2f}")
print(f"Impact: {result['impact_percent']:.2f}%")
```

### Step 3: Explore Available Scenarios

```python
import xfin

# List all scenarios
scenarios = xfin.list_scenarios()
print(scenarios)

# Show scenario details
xfin.show_scenarios()
```

## 🖥️ Launch Interactive Dashboard

```bash
# Stress Testing Dashboard
xfin stress

# Credit Risk Dashboard
xfin credit
```

## 📊 Available Stress Scenarios

| Scenario | Description |
|----------|-------------|
| `market_correction` | 10-15% equity drop |
| `recession_scenario` | 20-30% drop, economic downturn |
| `credit_crisis` | Credit market stress |
| `geopolitical_shock` | War/conflict impact |
| `commodity_shock` | Oil/commodity price spike |
| `pandemic_scenario` | Health crisis impact |

## 🌱 ESG Analysis

```python
# Score your portfolio's sustainability
esg = xfin.ESGAnalyzer()
score = esg.score_portfolio(portfolio)

print(f"ESG Score: {score['overall_score']:.2f}/100")
print(f"Rating: {score['rating']}")
```

## 📚 Next Steps

- [Stress Testing Guide](../README_STRESS_TESTING.md)
- [ESG Analysis Guide](../README_ESG.md)
- [Full API Documentation](API_GUIDE.md)

## 📞 Need Help?

- 📧 Email: rishabhbhangale@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/RishaBhangale/XFIN/issues)
