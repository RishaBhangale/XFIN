"""
XFIN FastAPI REST API
======================

Provides REST API endpoints for:
- Portfolio stress testing
- ESG scoring
- Credit risk analysis

Run with: uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import io
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from XFIN.stress_testing import StressTestingEngine
from XFIN.esg import ESGScoringEngine

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="XFIN API",
    description="Financial analysis API for stress testing, ESG scoring, and credit risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Pydantic Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str


class Holding(BaseModel):
    ticker: str = Field(..., description="Stock ticker (e.g., RELIANCE.NS)")
    quantity: float = Field(..., gt=0, description="Number of shares")
    current_price: float = Field(..., gt=0, description="Current price per share")
    sector: Optional[str] = Field(None, description="Sector classification")


class PortfolioRequest(BaseModel):
    holdings: List[Holding]
    api_key: Optional[str] = Field(None, description="OpenRouter API key for AI explanations")


class StressTestRequest(BaseModel):
    holdings: List[Holding]
    scenario: str = Field("market_correction", description="Stress scenario name")
    api_key: Optional[str] = None


class ESGRequest(BaseModel):
    holdings: List[Holding]
    api_key: Optional[str] = None


class SecurityESGRequest(BaseModel):
    name: str = Field(..., description="Company name")
    ticker: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None


class StressTestResponse(BaseModel):
    scenario_name: str
    portfolio_value: float
    stressed_value: float
    impact_percent: float
    explanation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ESGScoreResponse(BaseModel):
    overall_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    star_rating: int
    rating_label: str
    coverage_percentage: Optional[float] = None
    holdings_detail: Optional[List[Dict]] = None


class ScenarioInfo(BaseModel):
    name: str
    description: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def holdings_to_dataframe(holdings: List[Holding]) -> pd.DataFrame:
    """Convert holdings list to pandas DataFrame."""
    data = []
    for h in holdings:
        data.append({
            'Ticker': h.ticker,
            'Quantity': h.quantity,
            'Current_Price': h.current_price,
            'Sector': h.sector or 'Unknown'
        })
    return pd.DataFrame(data)


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint returning API health status."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for container orchestration."""
    return HealthResponse(status="healthy", version="1.0.0")


# =============================================================================
# Stress Testing Endpoints
# =============================================================================

@app.get("/api/v1/scenarios", response_model=List[str], tags=["Stress Testing"])
async def list_scenarios():
    """List all available stress test scenarios."""
    engine = StressTestingEngine()
    return engine.scenario_generator.list_scenarios()


@app.post("/api/v1/stress-test", response_model=StressTestResponse, tags=["Stress Testing"])
async def run_stress_test(request: StressTestRequest):
    """
    Run a stress test on a portfolio.
    
    Analyzes the impact of a specified stress scenario on the given portfolio.
    """
    try:
        engine = StressTestingEngine(api_key=request.api_key)
        portfolio_df = holdings_to_dataframe(request.holdings)
        
        result = engine.explain_stress_impact(portfolio_df, request.scenario)
        
        return StressTestResponse(
            scenario_name=request.scenario,
            portfolio_value=result.get('portfolio_value', 0),
            stressed_value=result.get('stressed_value', 0),
            impact_percent=result.get('impact_percent', 0),
            explanation=result.get('explanation'),
            details=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/compare-scenarios", tags=["Stress Testing"])
async def compare_scenarios(request: PortfolioRequest, scenarios: List[str] = None):
    """
    Compare multiple stress scenarios on a portfolio.
    
    If scenarios not provided, uses default set of scenarios.
    """
    try:
        engine = StressTestingEngine(api_key=request.api_key)
        portfolio_df = holdings_to_dataframe(request.holdings)
        
        if scenarios is None:
            scenarios = ['market_correction', 'recession_scenario', 'credit_crisis']
        
        result = engine.compare_scenarios(portfolio_df, scenarios)
        
        # Convert DataFrame to dict if needed
        if hasattr(result, 'to_dict'):
            return result.to_dict('records')
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ESG Scoring Endpoints
# =============================================================================

@app.post("/api/v1/esg/portfolio", response_model=ESGScoreResponse, tags=["ESG"])
async def score_portfolio_esg(request: ESGRequest):
    """
    Calculate ESG scores for a portfolio.
    
    Returns weighted average ESG scores based on holdings.
    """
    try:
        engine = ESGScoringEngine(api_key=request.api_key)
        portfolio_df = holdings_to_dataframe(request.holdings)
        
        result = engine.calculate_portfolio_esg(portfolio_df)
        
        return ESGScoreResponse(
            overall_score=result.get('weighted_esg_score', result.get('overall_score', 0)),
            environmental_score=result.get('environmental_score', 0),
            social_score=result.get('social_score', 0),
            governance_score=result.get('governance_score', 0),
            star_rating=result.get('star_rating', 3),
            rating_label=result.get('rating_label', 'Average'),
            coverage_percentage=result.get('coverage_percentage'),
            holdings_detail=result.get('holdings_esg')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/esg/security", tags=["ESG"])
async def score_security_esg(request: SecurityESGRequest):
    """
    Calculate ESG score for a single security.
    """
    try:
        engine = ESGScoringEngine()
        
        security_data = {
            'name': request.name,
            'ticker': request.ticker,
            'sector': request.sector or 'Unknown',
            'market_cap': request.market_cap or 0
        }
        
        result = engine.score_security(security_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/esg/risk-multiplier/{score}", tags=["ESG"])
async def get_esg_risk_multiplier(score: float):
    """
    Get ESG-based risk multiplier for stress testing integration.
    
    Higher ESG scores result in lower risk multipliers.
    """
    try:
        engine = ESGScoringEngine()
        multiplier = engine.get_esg_risk_multiplier(score)
        return {"score": score, "risk_multiplier": multiplier}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# File Upload Endpoints
# =============================================================================

@app.post("/api/v1/upload/portfolio", tags=["Upload"])
async def upload_portfolio_csv(file: UploadFile = File(...)):
    """
    Upload a portfolio CSV file and parse it.
    
    Supports multiple broker formats (Zerodha, Upstox, etc.)
    """
    try:
        from XFIN.parsers import UniversalBrokerCSVParser
        
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        parser = UniversalBrokerCSVParser()
        lines = content_str.strip().split('\n')
        
        header_idx, broker = parser.find_header_row(lines)
        headers = lines[header_idx].split(',')
        column_mapping = parser.map_columns(headers)
        data_rows = parser.extract_data_rows(lines, header_idx, headers)
        
        return {
            "status": "success",
            "broker_detected": broker,
            "columns_mapped": column_mapping,
            "rows_parsed": len(data_rows),
            "sample_data": data_rows[:3] if data_rows else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")


# =============================================================================
# Run with: uvicorn api.main:app --reload --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
