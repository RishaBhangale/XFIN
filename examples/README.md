# XFIN Examples

Welcome to the XFIN examples directory! This folder contains practical examples demonstrating how to use XFIN's three main modules.

## 📂 Contents

### Quick Start
- See the main [README.md](../README.md) for installation and quick start

### Example Files

| Example | Description |
|---------|-------------|
| Portfolio Template | Sample CSV format for portfolio data |

## 🚀 Quick Start

### 1. Create your portfolio file

Create a CSV file with this format:

```csv
Ticker,Quantity,Current_Price,Sector
RELIANCE.NS,500,2450,Energy
TCS.NS,300,3600,IT
HDFCBANK.NS,1000,1650,Financials
INFY.NS,400,1500,IT
```

### 2. Run an analysis

```python
import XFIN as xfin
import pandas as pd

# Load your portfolio
portfolio = pd.read_csv('my_portfolio.csv')

# Stress Testing
stress = xfin.StressAnalyzer()
result = stress.explain_stress_impact(portfolio, 'market_correction')
print(f"Impact: {result['impact_percent']:.2f}%")

# ESG Analysis
esg = xfin.ESGAnalyzer()
score = esg.score_portfolio(portfolio)
print(f"ESG Score: {score['overall_score']:.2f}/100")
```

### 3. Launch dashboards

```bash
# Stress Testing Dashboard
xfin stress

# ESG Dashboard (if available)
xfin esg

# Credit Risk Dashboard
xfin credit
```

## 📋 Portfolio CSV Format

Your portfolio file should include these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `Ticker` | ✅ | Stock ticker (e.g., RELIANCE.NS) |
| `Quantity` | ✅ | Number of shares |
| `Current_Price` | ✅ | Current price per share |
| `Sector` | ✅ | Sector classification |
| `Company_Name` | ❌ | Optional company name |

## 📞 Support

- 📧 Email: rishabhbhangale@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/RishaBhangale/XFIN/issues)
