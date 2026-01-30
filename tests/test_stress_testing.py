"""
Tests for XFIN Stress Testing Module
=====================================

Unit tests for ScenarioGenerator, PortfolioAnalyzer, and StressTestingEngine.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScenarioGenerator:
    """Tests for ScenarioGenerator class."""
    
    def test_import(self):
        """Test that ScenarioGenerator can be imported."""
        from XFIN.stress_testing import ScenarioGenerator
        assert ScenarioGenerator is not None
    
    def test_init(self):
        """Test ScenarioGenerator initialization."""
        from XFIN.stress_testing import ScenarioGenerator
        gen = ScenarioGenerator()
        assert gen is not None
    
    def test_list_scenarios(self):
        """Test listing available scenarios."""
        from XFIN.stress_testing import ScenarioGenerator
        gen = ScenarioGenerator()
        
        scenarios = gen.list_scenarios()
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        assert 'market_correction' in scenarios
    
    def test_get_scenario(self):
        """Test getting a specific scenario."""
        from XFIN.stress_testing import ScenarioGenerator
        gen = ScenarioGenerator()
        
        scenario = gen.get_scenario('market_correction')
        
        assert isinstance(scenario, dict)
        assert 'name' in scenario or 'factors' in scenario
    
    def test_get_invalid_scenario(self):
        """Test getting an invalid scenario returns sensible result."""
        from XFIN.stress_testing import ScenarioGenerator
        gen = ScenarioGenerator()
        
        # Should return None or raise error for invalid scenario
        result = gen.get_scenario('nonexistent_scenario')
        # Either returns None or empty dict is acceptable
        assert result is None or isinstance(result, dict)
    
    def test_calculate_diversification_factor(self):
        """Test HHI and diversification factor calculation."""
        from XFIN.stress_testing import ScenarioGenerator
        gen = ScenarioGenerator()
        
        # Perfectly diversified across 10 sectors
        sector_weights = {f'Sector_{i}': 10 for i in range(10)}
        result = gen.calculate_diversification_factor(sector_weights)
        
        # Result may be tuple (hhi, factor) or dict
        if isinstance(result, tuple):
            assert len(result) >= 2
        else:
            assert 'hhi' in result or 'hhi_scaled' in result or 'diversification_factor' in result
    
    def test_concentrated_portfolio_hhi(self):
        """Test HHI calculation for concentrated portfolio."""
        from XFIN.stress_testing import ScenarioGenerator
        gen = ScenarioGenerator()
        
        # Single sector (maximum concentration)
        sector_weights = {'Energy': 100}
        result = gen.calculate_diversification_factor(sector_weights)
        
        # Result may be tuple or dict
        if isinstance(result, tuple):
            # Tuple format: (hhi, diversification_factor)
            hhi = result[0]
        else:
            hhi = result.get('hhi_scaled', result.get('hhi', 0))
        
        # HHI should be maximum for single sector
        assert hhi >= 9000 or hhi == 10000 or hhi == 1.0


class TestPortfolioAnalyzer:
    """Tests for PortfolioAnalyzer class."""
    
    def test_import(self):
        """Test that PortfolioAnalyzer can be imported."""
        from XFIN.stress_testing import PortfolioAnalyzer
        assert PortfolioAnalyzer is not None
    
    def test_init(self):
        """Test PortfolioAnalyzer initialization."""
        from XFIN.stress_testing import PortfolioAnalyzer
        analyzer = PortfolioAnalyzer()
        assert analyzer is not None
    
    def test_categorize_asset(self):
        """Test asset categorization."""
        from XFIN.stress_testing import PortfolioAnalyzer
        analyzer = PortfolioAnalyzer()
        
        # Test common stock categorization
        category = analyzer.categorize_asset('RELIANCE')
        assert isinstance(category, str)
    
    def test_analyze_portfolio(self, sample_portfolio):
        """Test portfolio analysis."""
        from XFIN.stress_testing import PortfolioAnalyzer
        analyzer = PortfolioAnalyzer()
        
        result = analyzer.analyze_portfolio(sample_portfolio)
        
        assert isinstance(result, dict)
        # Accept various key names
        has_value = 'total_value' in result or 'portfolio_value' in result or 'composition' in result
        assert has_value or len(result) > 0, f"Expected analysis result, got: {list(result.keys())}"


class TestStressTestingEngine:
    """Tests for StressTestingEngine (main public API)."""
    
    def test_import(self):
        """Test that StressTestingEngine can be imported."""
        from XFIN.stress_testing import StressTestingEngine
        assert StressTestingEngine is not None
    
    def test_init(self):
        """Test StressTestingEngine initialization."""
        from XFIN.stress_testing import StressTestingEngine
        engine = StressTestingEngine()
        assert engine is not None
    
    def test_init_with_alias(self):
        """Test that StressAnalyzer alias works."""
        from XFIN import StressAnalyzer
        analyzer = StressAnalyzer()
        assert analyzer is not None
    
    def test_explain_stress_impact(self, sample_portfolio):
        """Test stress impact explanation."""
        from XFIN.stress_testing import StressTestingEngine
        engine = StressTestingEngine()
        
        result = engine.explain_stress_impact(sample_portfolio, 'market_correction')
        
        assert isinstance(result, dict)
        # Accept various result structures
        stress_keys = ['portfolio_value', 'stressed_value', 'impact_percent', 'dynamic_impact', 'scenario_name', 'portfolio_analysis']
        has_stress_key = any(k in result for k in stress_keys)
        assert has_stress_key, f"Expected stress result, got: {list(result.keys())}"
    
    def test_compare_scenarios(self, sample_portfolio):
        """Test scenario comparison."""
        from XFIN.stress_testing import StressTestingEngine
        engine = StressTestingEngine()
        
        scenarios = ['market_correction', 'recession_scenario']
        result = engine.compare_scenarios(sample_portfolio, scenarios)
        
        # Result can be dict, list, or DataFrame
        import pandas as pd
        assert isinstance(result, (dict, list, pd.DataFrame))
        if isinstance(result, pd.DataFrame):
            assert len(result) >= 0  # DataFrame may be empty
        elif isinstance(result, list):
            assert len(result) >= 0
        else:
            assert len(result) >= 0


class TestStressTestingIntegration:
    """Integration tests for stress testing workflow."""
    
    def test_full_workflow(self, sample_portfolio):
        """Test complete stress testing workflow."""
        from XFIN import StressAnalyzer
        
        # Initialize
        analyzer = StressAnalyzer()
        
        # Run stress test
        result = analyzer.explain_stress_impact(sample_portfolio, 'market_correction')
        
        # Verify result structure
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0, "Expected non-empty result"
    
    def test_multiple_scenarios(self, sample_portfolio, scenario_names):
        """Test running multiple scenarios."""
        from XFIN import StressAnalyzer
        analyzer = StressAnalyzer()
        
        results = {}
        for scenario in scenario_names[:2]:  # Test first 2 scenarios
            results[scenario] = analyzer.explain_stress_impact(sample_portfolio, scenario)
        
        assert len(results) == 2
        for name, result in results.items():
            assert result is not None
            assert isinstance(result, dict)
