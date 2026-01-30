# Using Custom Portfolio Files

This guide explains how to use your own portfolio data with XFIN.

## Supported Formats

- **CSV** (.csv) - Recommended
- **Excel** (.xlsx, .xls)

## Required Columns

Your file must include these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Ticker` | Stock symbol | `RELIANCE.NS`, `TCS.NS` |
| `Quantity` | Number of shares | `500` |
| `Current_Price` | Price per share | `2450.00` |
| `Sector` | Sector classification | `Energy`, `IT`, `Financials` |

## Example CSV Format

```csv
Ticker,Quantity,Current_Price,Sector,Company_Name
RELIANCE.NS,500,2450,Energy,Reliance Industries
TCS.NS,300,3600,IT,Tata Consultancy Services
HDFCBANK.NS,1000,1650,Financials,HDFC Bank
INFY.NS,400,1500,IT,Infosys
WIPRO.NS,600,400,IT,Wipro
ICICIBANK.NS,800,950,Financials,ICICI Bank
```

## Sector Classifications

XFIN recognizes these sectors:

- Energy
- IT / Technology
- Financials
- Healthcare
- Consumer Discretionary
- Consumer Staples
- Industrials
- Materials
- Utilities
- Real Estate
- Communication Services

## Loading Your Data

### In Python

```python
import pandas as pd
import XFIN as xfin

# Load CSV
portfolio = pd.read_csv('my_portfolio.csv')

# Or load Excel
portfolio = pd.read_excel('my_portfolio.xlsx')

# Run analysis
stress = xfin.StressAnalyzer()
result = stress.explain_stress_impact(portfolio, 'market_correction')
```

### Via CLI

```bash
xfin stress-analyze --portfolio my_portfolio.csv --scenario recession_scenario
xfin esg-analyze --portfolio my_portfolio.csv --output results.csv
```

## Troubleshooting

### "File not found"
- Use the full path to your file
- Check file extension is correct

### "Missing columns"
- Ensure all required columns exist
- Check column names match exactly (case-sensitive)

### "Invalid sector"
- Use standard sector names listed above
- Sector names are case-insensitive

## Tips

1. **Indian Stocks**: Use `.NS` suffix for NSE stocks (e.g., `RELIANCE.NS`)
2. **US Stocks**: Use plain ticker (e.g., `AAPL`)
3. **Multiple Currencies**: Currently INR is default; USD also supported

## Example Templates

Download sample templates from the [examples](../examples/) directory.
