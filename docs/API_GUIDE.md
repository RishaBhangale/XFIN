# XFIN API Guide

Complete API reference for the XFIN library.

## Core Classes

### StressAnalyzer (StressTestingEngine)

The main class for portfolio stress testing.

```python
from XFIN import StressAnalyzer

stress = StressAnalyzer(api_key=None)  # Optional API key for LLM features
```

#### Methods

##### `explain_stress_impact(portfolio_data, scenario_name)`
Calculate stress impact on a portfolio.

```python
result = stress.explain_stress_impact(portfolio_df, 'recession_scenario')
```

**Returns:**
```python
{
    'portfolio_value': float,      # Current portfolio value
    'stressed_value': float,       # Value after stress
    'impact_percent': float,       # Percentage impact
    'dollar_loss': float,          # Absolute loss
    'scenario_name': str,          # Scenario applied
    'sector_breakdown': dict       # Impact by sector
}
```

##### `compare_scenarios(portfolio_data, scenario_names)`
Compare multiple stress scenarios.

```python
comparison = stress.compare_scenarios(portfolio_df, ['recession_scenario', 'market_correction'])
```

##### `generate_recommendations(portfolio_data, stress_analysis, fast_mode=False)`
Generate portfolio improvement recommendations.

```python
recommendations = stress.generate_recommendations(portfolio_df, result, fast_mode=True)
```

---

### ESGAnalyzer (ESGScoringEngine)

ESG scoring and sustainability analysis.

```python
from XFIN import ESGAnalyzer

esg = ESGAnalyzer(api_key=None, scoring_methodology="sector_adjusted")
```

#### Methods

##### `score_security(security_data)`
Score a single security.

```python
score = esg.score_security({
    'name': 'Reliance Industries',
    'ticker': 'RELIANCE.NS',
    'sector': 'Energy'
})
```

**Returns:**
```python
{
    'overall_score': float,        # 0-100 ESG score
    'environmental_score': float,  # E component
    'social_score': float,         # S component
    'governance_score': float,     # G component
    'star_rating': int,            # 1-5 stars
    'rating_label': str            # 'Leader', 'Strong', etc.
}
```

##### `score_portfolio(portfolio_data)`
Score entire portfolio.

```python
portfolio_score = esg.score_portfolio(portfolio_df)
```

---

### ScenarioGenerator

Generate and manage stress scenarios.

```python
from XFIN.stress_testing import ScenarioGenerator

gen = ScenarioGenerator()
```

#### Methods

##### `list_scenarios()`
List available scenario names.

##### `get_scenario(name)`
Get scenario details.

##### `calculate_dynamic_scenario_impact(scenario_key, portfolio_data)`
Calculate portfolio-specific impact.

---

## Utility Functions

### `xfin.configure(alpha_vantage_key=None, openrouter_key=None)`
Configure API keys.

### `xfin.info()`
Display library information.

### `xfin.list_scenarios()`
List available stress scenarios.

### `xfin.show_scenarios()`
Display detailed scenario information.

---

## Portfolio DataFrame Format

Required columns:
| Column | Type | Description |
|--------|------|-------------|
| `Ticker` | str | Stock symbol |
| `Quantity` | int/float | Number of shares |
| `Current_Price` | float | Current share price |
| `Sector` | str | Sector classification |

Optional columns:
| Column | Type | Description |
|--------|------|-------------|
| `Company_Name` | str | Company name |
| `Current_Value` | float | Pre-calculated value |
| `ISIN` | str | ISIN code |

---

## Error Handling

```python
try:
    result = stress.explain_stress_impact(portfolio_df, 'invalid_scenario')
except ValueError as e:
    print(f"Invalid scenario: {e}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Examples

See the [examples](../examples/) directory for complete working examples.
