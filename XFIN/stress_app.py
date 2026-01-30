import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import os

# Handle imports whether running as module or standalone
try:
    from XFIN.stress_testing import StressTestingEngine
    from XFIN.stress_plots import StressPlotGenerator
    from XFIN.esg import ESGScoringEngine
    from XFIN.esg_plots import ESGPlotGenerator
    from XFIN.parsers import UniversalBrokerCSVParser
    from XFIN.parsers.data_cleaning import clean_portfolio_data, get_value_column, get_stock_name_column
except ImportError:
    # Running from within XFIN directory - add parent to path for proper package resolution
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_current_dir)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    
    # Now import using XFIN package prefix (this works because parent is in path)
    from XFIN.stress_testing import StressTestingEngine
    from XFIN.stress_plots import StressPlotGenerator
    from XFIN.esg import ESGScoringEngine
    from XFIN.esg_plots import ESGPlotGenerator
    from XFIN.parsers import UniversalBrokerCSVParser
    from XFIN.parsers.data_cleaning import clean_portfolio_data, get_value_column, get_stock_name_column

# Alias for backward compatibility
clean_and_handle_missing_data = clean_portfolio_data

# UniversalBrokerCSVParser and helper functions are now imported from XFIN.parsers

def clean_and_handle_missing_data_streamlit(df, column_mapping):
    """
    Streamlit wrapper for clean_portfolio_data.
    Adds Streamlit info messages for user feedback.
    """
    original_count = len(df)
    df, changes = clean_portfolio_data(df, column_mapping, show_info=False)
    
    final_count = len(df)
    total_removed = original_count - final_count
    
    if total_removed > 0:
        st.info(f"ℹ️ Processed {original_count} rows → Kept {final_count} valid rows (removed {total_removed})")
    
    return df, changes

def parse_broker_csv(file_content):
    """Enhanced universal broker CSV parser with intelligent format detection"""
    parser = UniversalBrokerCSVParser()
    
    try:
        # Read the entire file content
        if isinstance(file_content, bytes):
            content = file_content.decode('utf-8')
        else:
            content = str(file_content)
        
        lines = content.strip().split('\n')
        if len(lines) < 2:
            st.error("File appears to be empty or has insufficient data")
            return None
        
        # Extract only Unrealised P&L section if present (skip sold stocks)
        filtered_lines = parser.extract_unrealised_section(lines)
        
        # Check if section filtering was applied
        if len(filtered_lines) < len(lines):
            st.info(f"📊 Filtered to Unrealised P&L section ({len(filtered_lines)} rows from {len(lines)} total)")
        
        # Smart header detection on filtered lines
        header_idx, broker_format = parser.find_header_row(filtered_lines)
        
        # Extract headers and map to standard names
        header_line = filtered_lines[header_idx]
        raw_headers = [col.strip() for col in header_line.split(',')]
        column_mapping = parser.map_columns(raw_headers)
        
        # Extract data rows from filtered lines only
        data_rows = parser.extract_data_rows(filtered_lines, header_idx, raw_headers)
        
        if not data_rows:
            st.error("No valid data rows found after header")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=raw_headers)
        
        # Apply universal column cleaning
        for standard_name, broker_column in column_mapping.items():
            if broker_column in df.columns:
                if standard_name in ['quantity', 'avg_price', 'current_price', 'current_value', 'invested_value', 'pnl']:
                    df[broker_column] = df[broker_column].apply(parser.clean_numeric_value)
        
        # Calculate missing values using available data
        if 'current_value' not in column_mapping:
            # Try to calculate current value
            current_price_col = column_mapping.get('current_price')
            quantity_col = column_mapping.get('quantity')
            if current_price_col and quantity_col and current_price_col in df.columns and quantity_col in df.columns:
                df['Current Value'] = df[current_price_col] * df[quantity_col]
                
        if 'invested_value' not in column_mapping:
            # Try to calculate invested value
            avg_price_col = column_mapping.get('avg_price')
            quantity_col = column_mapping.get('quantity')
            if avg_price_col and quantity_col and avg_price_col in df.columns and quantity_col in df.columns:
                df['Invested Value'] = df[avg_price_col] * df[quantity_col]
        
        # Enhanced missing data handling
        df, data_changes = clean_and_handle_missing_data(df, column_mapping)
        
        # Simple success message for users
        st.success("✅ Portfolio loaded successfully!")
        
        # Display data cleaning summary if changes were made
        if data_changes:
            with st.expander("📋 Data Cleaning Summary - Click to see what was processed"):
                st.markdown("### Changes Made to Your Portfolio Data")
                for change_type, changes in data_changes.items():
                    if changes:
                        st.markdown(f"**{change_type}:**")
                        for change in changes:
                            st.write(f"- {change}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        st.info("💡 **Tip**: Make sure your CSV has columns for stock names, quantities, and prices/values")
        return None

# get_value_column and get_stock_name_column are now imported from XFIN.parsers.data_cleaning

def calculate_enhanced_value(row, df, col_map, date_price_columns):
    """
    Calculate current value for a holding using multiple methods.
    Mirrors the logic from esg.py for consistency.
    """
    from XFIN.data_utils import find_column_case_insensitive, safe_float_conversion
    
    # Method 1: Check existing "Current Value" column (skip if zero)
    current_value_col = find_column_case_insensitive(
        col_map, 
        ['current value', 'market value']
    )
    if current_value_col:
        val = safe_float_conversion(row.get(current_value_col, 0))
        if val > 0:
            return val
    
    # Method 2: Qty × Price
    qty_col = find_column_case_insensitive(
        col_map, 
        ['qty', 'quantity', 'portfolio holdings', 'shares', 'units']
    )
    price_col = find_column_case_insensitive(
        col_map, 
        ['ltp', 'cmp', 'current price', 'price']
    )
    
    if qty_col and price_col:
        qty = safe_float_conversion(row.get(qty_col, 0))
        price = safe_float_conversion(row.get(price_col, 0))
        if qty > 0 and price > 0:
            return qty * price
    
    # Method 3: Date columns
    if date_price_columns and qty_col:
        qty = safe_float_conversion(row.get(qty_col, 0))
        for date_col in date_price_columns:
            date_price = safe_float_conversion(row.get(date_col, 0))
            if date_price > 0 and qty > 0:
                return qty * date_price
    
    # Method 4: Invested + P&L
    invested_col = find_column_case_insensitive(
        col_map, 
        ['invested value', 'invested', 'cost']
    )
    pnl_col = find_column_case_insensitive(
        col_map, 
        ['p&l', 'pnl', 'profit/loss', 'unrealized profit/loss', 
         'unrealized profit / loss', 'unrealised profit/loss']
    )
    
    if invested_col and pnl_col:
        invested = safe_float_conversion(row.get(invested_col, 0))
        pnl = safe_float_conversion(row.get(pnl_col, 0))
        if invested > 0:
            return invested + pnl
    
    # Method 5: Invested × (1 + Change%)
    change_col = find_column_case_insensitive(
        col_map, 
        ['% chg', '% change', 'change%']
    )
    
    if invested_col and change_col:
        invested = safe_float_conversion(row.get(invested_col, 0))
        change_pct = safe_float_conversion(row.get(change_col, 0))
        if invested > 0:
            if abs(change_pct) > 1:
                change_pct = change_pct / 100.0
            return invested * (1 + change_pct)
    
    # Method 6: Qty × Avg Cost
    avg_cost_col = find_column_case_insensitive(
        col_map, 
        ['avg cost', 'average cost', 'average cost value']
    )
    
    if qty_col and avg_cost_col:
        qty = safe_float_conversion(row.get(qty_col, 0))
        avg_cost = safe_float_conversion(row.get(avg_cost_col, 0))
        if qty > 0 and avg_cost > 0:
            return qty * avg_cost
    
    # Method 7: Invested fallback
    if invested_col:
        invested = safe_float_conversion(row.get(invested_col, 0))
        if invested > 0:
            return invested
    
    return 0.0

def create_stress_dashboard():
    # Streamlit page config
    st.set_page_config(page_title="XFIN Portfolio Stress Testing", page_icon="📈", layout="wide")
    
    st.title("📈 XFIN Portfolio Stress Testing")
    st.markdown("**AI-Powered Portfolio Analysis for Real-World Scenarios**")
    
    # Initialize ESG engine early (before any usage)
    try:
        esg_engine = ESGScoringEngine()
        esg_plot_gen = ESGPlotGenerator()
        esg_available = True
    except Exception as e:
        esg_available = False
        esg_engine = None
        esg_plot_gen = None
    
    # Sidebar: Portfolio upload and display
    st.sidebar.header("📂 Portfolio Configuration")
    portfolio_file = st.sidebar.file_uploader("Upload Portfolio CSV", type=['csv'])
    
    # LLM Configuration section
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AI Analysis Settings")
    
    # API Key input
    st.sidebar.markdown("### 🔑 AI Configuration")
    api_key = st.sidebar.text_input(
        "OpenRouter API Key (Optional)",
        type="password",
        help="Enter your OpenRouter API key for enhanced AI explanations. Leave blank to use environment variable."
    )
    
    if not api_key:
        st.sidebar.info("💡 **Get a free API key**: Visit [OpenRouter.ai](https://openrouter.ai/) to enable AI-powered recommendations")
    
    # LLM Model selection
    llm_model = st.sidebar.selectbox(
        "AI Model for Analysis",
        ["x-ai/grok-code-fast-1", "anthropic/claude-3-haiku", "openai/gpt-4o-mini"],
        help="Select the AI model for generating explanations"
    )
    
    # Enable/disable LLM
    use_llm = st.sidebar.checkbox(
        "Enable AI-Powered Recommendations",
        value=True,
        help="Get detailed AI-generated analysis and recommendations"
    )
    
    # Fast Mode toggle for backend performance (not for charts)
    fast_mode = st.sidebar.checkbox(
        "⚡ Fast Mode (Skip Market Data)",
        value=True,
        help="Skip live market data fetching for faster backend analysis. Charts will still be generated."
    )
    
    if use_llm:
        st.sidebar.success("✅ AI recommendations enabled")
    else:
        st.sidebar.info("ℹ️ Using basic analysis only")
        
    if fast_mode:
        st.sidebar.success("⚡ Fast mode: Quick backend analysis")
        st.sidebar.info("📊 All charts generated + fast core analysis")
    else:
        st.sidebar.warning("🐌 Full mode: With live market data")
        st.sidebar.info("📈 All charts + detailed market analysis")
    
    # Portfolio processing
    if portfolio_file:
        try:
            # Read file content
            file_content = portfolio_file.read()
            portfolio_data = parse_broker_csv(file_content)
            
            if portfolio_data is not None:
                st.sidebar.success("✅ Portfolio loaded!")
                
                # Debug: Show column names
                st.sidebar.write("**Columns detected:**")
                for col in portfolio_data.columns:
                    st.sidebar.write(f"- {col}")
                    
            else:
                st.sidebar.error("Failed to parse portfolio file")
        except Exception as e:
            st.sidebar.error(f"Error loading portfolio: {e}")
            portfolio_data = None
    else:
        portfolio_data = None
        st.sidebar.info("⬇️ Please upload your portfolio CSV")
    
    if portfolio_data is not None:
        # Import enhanced value calculation utilities from consolidated data_utils
        from XFIN.data_utils import (
            detect_date_price_columns,
            find_column_case_insensitive, 
            safe_float_conversion
        )
        
        # Create column mapping
        col_map = {col.lower(): col for col in portfolio_data.columns}
        date_price_columns = detect_date_price_columns(portfolio_data)
        
        # Calculate enhanced values for each holding
        enhanced_values = []
        for idx, row in portfolio_data.iterrows():
            value = calculate_enhanced_value(row, portfolio_data, col_map, date_price_columns)
            enhanced_values.append(value)
        
        # Add enhanced values to dataframe
        portfolio_data['_enhanced_value'] = enhanced_values
        
        # Find the appropriate columns
        value_col = '_enhanced_value'  # Use our calculated values
        name_col = get_stock_name_column(portfolio_data)
        
        if value_col and name_col:
            # Calculate total portfolio value using enhanced calculation
            total_value = portfolio_data[value_col].sum()
            
            st.sidebar.markdown("**Your Portfolio Summary:**")
            st.sidebar.write(f"**Total Portfolio Value:** ₹{total_value:,.0f}")
            st.sidebar.write(f"**Value Source:** Enhanced Calculation")
            st.sidebar.write(f"**Number of Holdings:** {len(portfolio_data)}")
            
            # Show investment vs current value using enhanced calculation
            invested_col = find_column_case_insensitive(
                col_map, 
                ['invested value', 'invested', 'cost']
            )
            
            if invested_col:
                # Ensure numeric conversion (some broker CSVs keep currency symbols/commas)
                total_invested = portfolio_data[invested_col].apply(lambda x: safe_float_conversion(x)).sum()
                total_current = total_value  # This is our enhanced calculation
                total_pnl = total_current - total_invested
                pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0

                st.sidebar.write(f"**Total Invested:** ₹{total_invested:,.0f}")
                st.sidebar.write(f"**Current Value:** ₹{total_current:,.0f}")

                color = "🟢" if total_pnl >= 0 else "🔴"
                st.sidebar.write(f"**Total P&L:** {color} ₹{total_pnl:,.0f} ({pnl_pct:+.1f}%)")
            
            # Show portfolio stats from existing columns if available
            elif 'Unrealised P&L' in portfolio_data.columns:
                total_pnl = portfolio_data['Unrealised P&L'].sum()
                pnl_pct = (total_pnl / total_value) * 100 if total_value > 0 else 0
                color = "🟢" if total_pnl >= 0 else "🔴"
                st.sidebar.write(f"**Total P&L:** {color} ₹{total_pnl:,.0f} ({pnl_pct:+.1f}%)")
            
            # Show top holdings
            st.sidebar.markdown("**Top Holdings:**")
            try:
                top_holdings = portfolio_data.nlargest(5, value_col)
                for _, row in top_holdings.iterrows():
                    holding_val = row[value_col]
                    weight = (holding_val / total_value) * 100
                    stock_name = row[name_col]
                    
                    # Show quantity and average price if available
                    extra_info = ""
                    if 'Quantity' in row and 'Average buy price' in row:
                        qty = row['Quantity']
                        avg_price = row['Average buy price']
                        if pd.notna(qty) and pd.notna(avg_price):
                            extra_info = f" (Qty: {qty:.0f} @ ₹{avg_price:.2f})"
                    
                    st.sidebar.write(f"- {stock_name}: ₹{holding_val:,.0f} ({weight:.1f}%){extra_info}")
            except Exception as e:
                st.sidebar.write("Unable to display top holdings")
            
            # Show portfolio composition chart in sidebar
            st.sidebar.markdown("### 📊 Portfolio Overview")
            try:
                temp_plot_gen = StressPlotGenerator()
                fig_composition = temp_plot_gen.create_portfolio_composition_pie(portfolio_data, "Your Portfolio")
                st.sidebar.pyplot(fig_composition)
                plt.close(fig_composition)
            except Exception as e:
                st.sidebar.info("💡 Chart temporarily unavailable")
            
            # ESG Analysis will run when "Analyze My Portfolio" is clicked
            # Display placeholder in sidebar if ESG was previously analyzed
            if esg_available and 'esg_result' in st.session_state:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🌱 ESG Score")
                esg_result = st.session_state['esg_result']
                
                if esg_result and 'portfolio_esg_scores' in esg_result:
                    overall_score = esg_result['portfolio_esg_scores']['overall']
                    star_rating = esg_result['star_rating_text']
                    rating_label = esg_result['rating_label']
                    coverage = esg_result.get('coverage_percentage', 0)
                    risk_mult = esg_result.get('risk_multiplier', 1.0)
                    
                    # Display ESG metrics
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        st.metric("ESG Score", f"{overall_score:.0f}/100")
                    with col2:
                        st.metric("Rating", star_rating)
                    
                    # Risk multiplier indicator
                    mult_color = "🟢" if risk_mult < 1.0 else "🟡" if risk_mult == 1.0 else "🔴"
                    st.sidebar.write(f"**{rating_label}** | Risk: {mult_color} {risk_mult:.2f}x")
                    st.sidebar.caption(f"📊 Data Coverage: {coverage:.0f}%")
            elif esg_available:
                st.sidebar.markdown("---")
                st.sidebar.info("🌱 ESG analysis will run when you click 'Analyze My Portfolio'")
                            
        else:
            st.sidebar.warning("⚠️ Required columns not found. Please check your CSV format.")
    
    # Initialize modules
    model = type("MockModel", (), {"predict_stress_impact": lambda self, d, f: None})()
    
    # Initialize StressTestingEngine - use defaults first, then set API key
    try:
        explainer = StressTestingEngine()
        if use_llm and api_key:
            explainer.api_key = api_key  # Set API key after initialization
        st.success("✅ StressTestingEngine initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing StressTestingEngine: {e}")
        st.info("Using mock explainer as fallback...")
        # Fallback: create a minimal mock explainer
        explainer = type("MockExplainer", (), {
            "scenario_generator": type("MockGen", (), {
                "list_scenarios": lambda self: ["market_correction", "recession_scenario", "inflation_spike", "tech_sector_crash", "us_bond_yields_impact"],
                "get_scenario": lambda self, x: {
                    "name": x.replace("_", " ").title(), 
                    "description": f"Mock description for {x.replace('_', ' ')}", 
                    "factors": {}
                }
            })(),
            "portfolio_analyzer": type("MockAnalyzer", (), {
                "analyze_stress_impact": lambda self, data, scenario: {
                    "total_impact": -15.5,
                    "impact_percentage": -15.5,
                    "recovery_months": 12,
                    "risk_level": "Medium"
                }
            })(),
            "explain_stress_impact": lambda self, data, scenario: {
                "explanation": f"Mock explanation for {scenario} scenario with portfolio impact analysis.",
                "risk_level": "Medium",
                "recommendations": ["Diversify portfolio", "Consider hedging strategies"]
            },
            "api_key": None
        })()
    
    plot_gen = StressPlotGenerator()
    
    # Update explainer with user's API key if provided
    if api_key and use_llm:
        explainer.api_key = api_key
    
    # Scenario selection
    scenarios = explainer.scenario_generator.list_scenarios()
    scenario_names = {}
    for s in scenarios:
        try:
            scenario_info = explainer.scenario_generator.get_scenario(s)
            scenario_names[s] = scenario_info.get('name', s.replace('_', ' ').title())
        except (KeyError, AttributeError):
            scenario_names[s] = s.replace('_', ' ').title()
    
    st.subheader("🎯 Choose Your Scenario")
    selected_scenario = st.selectbox("Select Scenario:", scenarios, format_func=lambda x: scenario_names[x])
    
    # Show scenario description
    scenario_info = explainer.scenario_generator.get_scenario(selected_scenario)
    scenario_name = scenario_info.get('name', selected_scenario.replace('_', ' ').title())
    scenario_desc = scenario_info.get('description', 'Stress testing scenario')
    st.info(f"**{scenario_name}**: {scenario_desc}")
    
    # Analysis type
    st.subheader("⚙️ Analysis Options")
    analysis_type = st.radio("What would you like to see?", ["Single Scenario Impact", "Compare Multiple Scenarios"])
    
    compare_keys = []
    if analysis_type == "Compare Multiple Scenarios" and portfolio_data is not None:
        # Fix for "No results" issue - ensure we don't filter out the selected scenario
        st.write("**Select Additional Scenarios:**")
        for scenario_key in scenarios:
            if st.checkbox(scenario_names[scenario_key], value=(scenario_key == selected_scenario), key=scenario_key):
                if scenario_key not in compare_keys:
                    compare_keys.append(scenario_key)
    
    # Add debug section
    with st.expander("🔧 Debug & Test", expanded=False):
        st.write("**Debug Information:**")
        if portfolio_data is not None:
            st.write(f"Portfolio shape: {portfolio_data.shape}")
            st.write(f"Portfolio columns: {list(portfolio_data.columns)}")
            
            # Test analysis button
            if st.button("🧪 Test Analysis (Debug Mode)", key="debug_test"):
                with st.spinner("Testing analysis..."):
                    try:
                        test_result = explainer.explain_stress_impact(portfolio_data, selected_scenario)
                        st.success("✅ Analysis test successful!")
                        st.json(test_result)
                    except Exception as e:
                        st.error(f"❌ Analysis test failed: {e}")
                        st.write("Error details:", str(e))
        else:
            st.info("Upload a portfolio first to enable debug testing")
    
    # Run analysis
    if st.button("🚀 Analyze My Portfolio", type="primary"):
        if portfolio_data is None:
            st.error("Please upload a portfolio CSV first.")
            st.stop()
        
        # Run ESG analysis first (only in non-fast mode for Single Scenario)
        if analysis_type == "Single Scenario Impact" and not fast_mode and esg_available and esg_engine:
            with st.spinner("🌱 Analyzing ESG scores..."):
                try:
                    esg_result = esg_engine.score_portfolio(portfolio_data)
                    st.session_state['esg_result'] = esg_result
                except Exception as e:
                    st.warning(f"ESG analysis skipped: {e}")
                    st.session_state['esg_result'] = None
        
        if analysis_type == "Single Scenario Impact":
            # PROGRESSIVE LOADING: Charts first, then AI
            
            # STEP 1: Generate core analysis quickly
            with st.spinner("⚡ Analyzing portfolio... (generating charts)"):
                try:
                    result = explainer.explain_stress_impact(portfolio_data, selected_scenario)
                    
                    # Handle different result structures
                    impact = None
                    portfolio_analysis = None
                    
                    if result and isinstance(result, dict):
                        # Try different result structures
                        if 'stress_impact' in result:
                            impact = result['stress_impact']
                            portfolio_analysis = result.get('portfolio_analysis')
                        elif 'total_impact' in result:
                            # Direct impact format
                            impact = result
                        
                        # Debug info (always show for troubleshooting)
                        with st.expander("🔍 Analysis Details", expanded=False):
                            st.write("**Raw Result Structure:**")
                            st.json(result)
                            if impact:
                                st.write("**Impact Data:**")
                                st.json(impact)
                    
                    if not impact:
                        st.error("❌ Could not extract stress impact from analysis")
                        st.info("💡 This might be due to:")
                        st.info("- CSV format issues (check column names)")
                        st.info("- Missing required columns (Symbol, Quantity, Price)")
                        st.info("- Data processing errors")
                        if result:
                            st.write("**Available result keys:**", list(result.keys()) if isinstance(result, dict) else "Not a dictionary")
                        st.stop()
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please check your CSV format and try again.")
                    st.stop()
                
                st.subheader("📋 Your Portfolio Analysis Results")
                
                # Calculate values for display
                if portfolio_data is not None and value_col:
                    total_portfolio_value = portfolio_data[value_col].sum()
                    # Ensure impact_percentage is converted to float
                    impact_percentage = float(impact['impact_percentage'])
                    impact_amount = total_portfolio_value * abs(impact_percentage) / 100
                else:
                    total_portfolio_value = 0
                    impact_percentage = float(impact['impact_percentage'])
                    impact_amount = 0
                
                # STEP 2: Display results based on fast_mode
                if fast_mode:
                    # FAST MODE: Basic text analysis only (no charts)
                    st.success("⚡ Fast Analysis Complete!")
                    st.markdown("### ⚡ Quick Impact Assessment")
                    
                    # Show basic metrics without charts
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Impact", f"{impact_percentage:.1f}%", 
                                f"₹{-impact_amount:,.0f}",
                                delta_color="inverse")
                    with col2:
                        st.metric("Risk Level", impact.get('risk_level', 'Medium'))
                    with col3:
                        st.metric("Recovery Time", f"{impact.get('recovery_months', 12):.0f} months")
                    with col4:
                        st.metric("VaR (95%)", f"{impact.get('var_95', 0)*100:.1f}%")
                    
                    # Display dynamic calculation details (NEW)
                    if 'dynamic_calculation' in impact:
                        dynamic = impact['dynamic_calculation']
                        st.markdown("---")
                        st.markdown("### 🎯 Portfolio-Specific Impact Calculation")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Base Impact", f"{dynamic.get('base_impact', 0):.2f}%",
                                    help="Sector-weighted baseline impact before diversification adjustment")
                        with col2:
                            hhi = dynamic.get('hhi', 0)
                            div_factor = dynamic.get('diversification_factor', 0)
                            div_label = "Well Diversified" if div_factor > 0.7 else "Concentrated" if div_factor < 0.4 else "Moderate"
                            st.metric("Diversification", f"{div_factor:.2f}",
                                    help=f"HHI: {hhi:.0f} - {div_label} portfolio")
                        with col3:
                            penalty = dynamic.get('concentration_penalty', 0)
                            st.metric("Concentration Penalty", f"+{penalty*100:.1f}%",
                                    help="Additional risk from portfolio concentration")
                        
                        # Show sector breakdown
                        with st.expander("📊 Sector Composition & Impact Breakdown"):
                            sector_comp = dynamic.get('sector_composition', {})
                            if sector_comp:
                                st.write("**Your Portfolio by Sector:**")
                                for sector, weight in sorted(sector_comp.items(), key=lambda x: x[1], reverse=True):
                                    st.write(f"• {sector}: {weight:.1f}%")
                            
                            # Show the detailed explanation
                            details_text = dynamic.get('details', '')
                            if details_text:
                                st.markdown(details_text)
                    
                    # Basic portfolio info without charts
                    st.markdown("**📋 Basic Portfolio Information:**")
                    st.write(f"• **Portfolio Value**: ₹{total_portfolio_value:,.0f}")
                    st.write(f"• **Potential Loss**: ₹{impact_amount:,.0f}")
                    st.write(f"• **Risk Assessment**: {impact.get('risk_level', 'Medium')} risk scenario")
                    st.write(f"• **Recovery Expectation**: {impact.get('recovery_months', 12):.0f} months")
                    
                else:
                    # FULL MODE: Generate all charts + comprehensive analysis
                    st.success("✅ Analysis Complete! Charts generated successfully.")
                    st.markdown("### 📊 Portfolio Analysis & Risk Assessment")
                
                # Chart generation (only for non-fast mode)  
                if not fast_mode:
                    try:
                        # Create tabs for different chart types - these should be fast
                        if esg_available and 'esg_result' in st.session_state:
                            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Risk Assessment", "📊 Portfolio Analysis", "⏰ Recovery Timeline", "🔍 Risk Breakdown", "🌱 ESG Analysis"])
                        else:
                            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Assessment", "📊 Portfolio Analysis", "⏰ Recovery Timeline", "🔍 Risk Breakdown"])
                        
                        with tab1:
                            st.markdown("### Risk Gauge Assessment")
                            try:
                                fig_gauge = plot_gen.create_risk_gauge_chart(impact, f"{selected_scenario.replace('_', ' ').title()} Risk Assessment")
                                st.pyplot(fig_gauge)
                                plt.close(fig_gauge)
                            except Exception as e:
                                st.error(f"Could not create risk gauge: {e}")
                                st.info("Using basic risk display instead")
                                st.write(f"**Risk Level**: {impact.get('risk_level', 'Unknown')}")
                        
                        with tab2:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### Portfolio Composition")
                                try:
                                    fig_composition = plot_gen.create_portfolio_composition_pie(portfolio_data, "Current Portfolio")
                                    st.pyplot(fig_composition)
                                    plt.close(fig_composition)
                                except Exception as e:
                                    st.error(f"Could not create portfolio composition chart: {e}")
                                    # Show basic portfolio info instead
                                    if result.get('portfolio_analysis'):
                                        composition = result['portfolio_analysis'].get('composition', {})
                                        st.write("**Portfolio Composition:**")
                                        for sector, weight in composition.items():
                                            st.write(f"- {sector}: {weight*100:.1f}%")
                            
                            with col2:
                                st.markdown("### Sector Risk Analysis")
                                try:
                                    fig_sector = plot_gen.create_sector_exposure_vs_impact_scatter(
                                        portfolio_data, impact, "Sector Risk vs Exposure"
                                    )
                                    st.pyplot(fig_sector)
                                    plt.close(fig_sector)
                                except Exception as e:
                                    st.error(f"Could not create sector analysis chart: {e}")
                                    st.info("Check portfolio data format - ensure it has Symbol and sector information")
                        
                        with tab3:
                            st.markdown("### Recovery Timeline")
                            try:
                                fig_recovery = plot_gen.create_recovery_timeline_chart(impact, "Expected Recovery Path")
                                st.pyplot(fig_recovery)
                                plt.close(fig_recovery)
                            except Exception as e:
                                st.error(f"Could not create recovery timeline: {e}")
                                # Show basic recovery info
                                recovery_months = impact.get('recovery_months', 12)
                                st.write(f"**Expected Recovery Time**: {recovery_months:.0f} months")
                                st.write("Recovery assumes 12.5% annual market return")
                        
                        with tab4:
                            st.markdown("### Risk Impact Breakdown")
                            try:
                                fig_waterfall = plot_gen.create_risk_impact_waterfall(impact, portfolio_data, "Risk Factor Analysis")
                                st.pyplot(fig_waterfall)
                                plt.close(fig_waterfall)
                            except Exception as e:
                                st.error(f"Could not create risk breakdown chart: {e}")
                                # Show basic risk breakdown
                                st.write("**Risk Summary:**")
                                st.write(f"- Total Impact: {impact.get('impact_percentage', 0):.1f}%")
                                st.write(f"- Risk Level: {impact.get('risk_level', 'Unknown')}")
                                st.write(f"- VaR 95%: {impact.get('var_95', 0)*100:.1f}%")
                        
                        # ESG Analysis Tab (only if ESG is available)
                        if esg_available and 'esg_result' in st.session_state:
                            with tab5:
                                st.markdown("### 🌱 ESG Analysis")
                                esg_result = st.session_state['esg_result']
                                
                                # Check if ESG result is valid (not None)
                                if esg_result is None or not isinstance(esg_result, dict):
                                    st.warning("⚠️ ESG analysis data is not available. This might happen if:")
                                    st.markdown("""
                                    - Portfolio has no stocks with valid ESG data
                                    - All stocks have zero values (no holdings)
                                    - ESG analysis encountered an error during processing
                                    
                                    **To fix**: Ensure your portfolio CSV contains stocks with non-zero values.
                                    """)
                                    st.info("💡 **Tip**: Check the console output for detailed error messages about ESG data fetching.")
                                else:
                                    try:
                                        # Row 1: ESG Gauge and Breakdown
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("#### Overall ESG Score")
                                            fig_gauge = esg_plot_gen.create_esg_gauge(esg_result)
                                            st.pyplot(fig_gauge)
                                            plt.close(fig_gauge)
                                        
                                        with col2:
                                            st.markdown("#### E-S-G Breakdown")
                                            fig_breakdown = esg_plot_gen.create_esg_breakdown(esg_result)
                                            st.pyplot(fig_breakdown)
                                            plt.close(fig_breakdown)
                                        
                                        # Row 2: Sector Heatmap
                                        st.markdown("#### ESG by Sector")
                                        fig_heatmap = esg_plot_gen.create_sector_esg_heatmap(esg_result)
                                        st.pyplot(fig_heatmap)
                                        plt.close(fig_heatmap)
                                        
                                        # Row 3: Coverage and Star Distribution
                                        col3, col4 = st.columns(2)
                                        with col3:
                                            st.markdown("#### Data Coverage")
                                            fig_coverage = esg_plot_gen.create_coverage_donut(esg_result)
                                            st.pyplot(fig_coverage)
                                            plt.close(fig_coverage)
                                        
                                        with col4:
                                            st.markdown("#### Star Rating Distribution")
                                            fig_stars = esg_plot_gen.create_esg_star_distribution(esg_result)
                                            st.pyplot(fig_stars)
                                            plt.close(fig_stars)
                                        
                                        # Holdings Detail Table
                                        st.markdown("#### Holdings ESG Detail")
                                        holdings_table = esg_plot_gen.create_holdings_esg_table(esg_result)
                                        st.dataframe(holdings_table, width='stretch')
                                        
                                        # ESG Risk Impact
                                        st.markdown("#### ESG Risk Adjustment")
                                        risk_mult = esg_result.get('risk_multiplier', 1.0)
                                        overall_score = esg_result['portfolio_esg_scores']['overall']
                                        
                                        if risk_mult < 1.0:
                                            st.success(f"✅ **Strong ESG performance** (Score: {overall_score:.0f}/100) reduces stress risk by {(1-risk_mult)*100:.0f}%")
                                        elif risk_mult > 1.0:
                                            st.warning(f"⚠️ **Weak ESG performance** (Score: {overall_score:.0f}/100) increases stress risk by {(risk_mult-1)*100:.0f}%")
                                        else:
                                            st.info(f"ℹ️ **Average ESG performance** (Score: {overall_score:.0f}/100) has neutral risk impact")
                                        
                                        # Show ESG-adjusted stress impact
                                        if impact:
                                            base_impact = impact.get('impact_percentage', 0)
                                            esg_adjusted_impact = base_impact * risk_mult
                                            impact_diff = esg_adjusted_impact - base_impact
                                            
                                            st.markdown("**ESG-Adjusted Stress Impact:**")
                                            col_a, col_b, col_c = st.columns(3)
                                            with col_a:
                                                st.metric("Base Impact", f"{base_impact:.1f}%")
                                            with col_b:
                                                st.metric("ESG Multiplier", f"{risk_mult:.2f}x")
                                            with col_c:
                                                st.metric("Adjusted Impact", f"{esg_adjusted_impact:.1f}%", 
                                                         delta=f"{impact_diff:+.1f}%", delta_color="inverse")
                                        
                                        # SHAP Waterfall Chart - ML Explainability (NEW!)
                                        st.markdown("---")
                                        st.markdown("#### 🔍 ML Model Explainability - What Drives Your ESG Score?")
                                        
                                        # Check if SHAP analysis is available
                                        shap_available = (esg_result.get('portfolio_shap_analysis') is not None and 
                                                        esg_result['portfolio_shap_analysis'] is not None)
                                        
                                        if shap_available:
                                            st.info("💡 This chart shows which factors most influenced the ML model's ESG predictions for your portfolio. Green bars increase the score, red bars decrease it.")
                                            
                                            try:
                                                fig_shap = esg_plot_gen.create_portfolio_shap_waterfall(esg_result, top_n=15)
                                                st.pyplot(fig_shap)
                                                plt.close(fig_shap)
                                                
                                                # Show SHAP data sources
                                                shap_data = esg_result['portfolio_shap_analysis']
                                                if 'data_sources' in esg_result:
                                                    sources = esg_result['data_sources']
                                                    ml_count = sources.get('ml_predictions', 0)
                                                    api_count = sources.get('api_data', 0)
                                                    proxy_count = sources.get('sector_proxy', 0)
                                                    
                                                    st.caption(f"📊 Analysis based on: {ml_count} ML predictions, {api_count} API data, {proxy_count} sector proxies")
                                                
                                                # Optional: Show grouped feature contributions
                                                with st.expander("📈 View Feature Category Breakdown"):
                                                    fig_grouped = esg_plot_gen.create_shap_grouped_bars(esg_result)
                                                    st.pyplot(fig_grouped)
                                                    plt.close(fig_grouped)
                                            
                                            except Exception as e:
                                                st.error(f"Could not create SHAP visualization: {e}")
                                                st.caption("💡 SHAP analysis is only available when ML predictions are used for ESG scoring.")
                                        else:
                                            # Show why SHAP is not available
                                            st.info("💡 **ML Explainability Not Available for This Portfolio**")
                                            
                                            # Show data sources breakdown
                                            if 'data_sources' in esg_result:
                                                sources = esg_result['data_sources']
                                                ml_count = sources.get('ml_predictions', 0)
                                                api_count = sources.get('api_data', 0)
                                                proxy_count = sources.get('sector_proxy', 0)
                                                total_stocks = api_count + ml_count + proxy_count
                                                
                                                # Explain based on actual data composition and 3-tier fallback
                                                if ml_count == 0 and api_count > 0 and proxy_count == 0:
                                                    st.write("✅ All stocks in your portfolio have **direct ESG ratings** from data providers (Yahoo Finance, BRSR).")
                                                    st.write("**No ML predictions were needed** - all stocks had real ESG data available.")
                                                elif ml_count == 0 and proxy_count > 0:
                                                    st.write(f"**3-Tier Fallback System:** API Data → ML Prediction → Sector Proxy")
                                                    st.write(f"For your portfolio: {api_count} stocks have real ESG data, but {proxy_count} stocks fell back to **sector proxy** (tier 3) because ML predictions failed (tier 2 - requires market data from yfinance).")
                                                    st.write("💡 ML predictions require valid ticker symbols and market data. Stocks with incorrect tickers or delisted stocks skip ML and go straight to sector proxy.")
                                                else:
                                                    st.write(f"**3-Tier Fallback System Active:** API Data → ML Prediction → Sector Proxy")
                                                
                                                st.write("**Your portfolio's ESG data sources:**")
                                                col_x, col_y, col_z = st.columns(3)
                                                with col_x:
                                                    st.metric("🌐 Tier 1: Real ESG", api_count, 
                                                             help="Stocks with actual ESG ratings from Yahoo Finance, BRSR, etc. (Highest quality)",
                                                             delta=f"{(api_count/total_stocks*100):.1f}% coverage" if total_stocks > 0 else None)
                                                with col_y:
                                                    st.metric("🤖 Tier 2: ML Predictions", ml_count, 
                                                             help="Stocks where ML model predicted ESG from market features (SHAP explainable)",
                                                             delta="SHAP available ✓" if ml_count > 0 else "Not used")
                                                with col_z:
                                                    st.metric("📊 Tier 3: Sector Proxy", proxy_count, 
                                                             help="Fallback: Stocks using sector average when tier 1 & 2 unavailable",
                                                             delta=f"{(proxy_count/total_stocks*100):.1f}% estimated" if total_stocks > 0 else None)
                                                
                                                st.markdown("---")
                                                if ml_count == 0:
                                                    st.write("**💡 Why no ML predictions?**")
                                                    st.write("ML predictions (tier 2) require:")
                                                    st.write("• Valid NSE ticker symbol (e.g., RELIANCE.NS)")
                                                    st.write("• Available market data on Yahoo Finance")
                                                    st.write("• Recent price history for volatility/returns calculation")
                                                    st.caption("🔍 If stocks have real ESG data (tier 1), ML is skipped. If ML can't fetch features, system falls back to sector proxy (tier 3).")
                                            else:
                                                # Fallback if data_sources not tracked
                                                st.warning("⚠️ Data source tracking not available for this analysis.")
                                                st.write("**XFIN uses 3-tier fallback:** API Data → ML Prediction → Sector Proxy")
                                                st.caption("💡 ML predictions are used when stocks lack public ESG ratings AND have valid market data available.")
                                    
                                    except Exception as e:
                                        st.error(f"Could not create ESG visualizations: {e}")
                                        st.write(f"Error details: {str(e)}")
                                        import traceback
                                        st.code(traceback.format_exc())
                
                    except Exception as e:
                        st.warning(f"Enhanced visualizations unavailable: {e}")
                        # Fallback to old chart
                        try:
                            fig = plot_gen.create_stress_impact_plot(result)
                            st.pyplot(fig)
                            plt.close(fig)
                        except:
                            st.error("All visualizations failed to load")
                
                # ============================================================================
                # STEP 3: AI-Powered Recommendations (separate phase, may take 3-5 seconds)
                # ============================================================================
                
                st.markdown("---")  # Visual separator
                st.markdown("### 💡 What Should You Do?")
                
                # Show analysis summary (for both modes)
                if not fast_mode:
                    # Full mode gets detailed summary with horizontal metrics
                    st.markdown("**📊 Comprehensive Analysis Summary:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Impact", f"{impact['impact_percentage']:.1f}%", 
                                f"₹{-impact_amount:,.0f}",
                                delta_color="inverse")
                    with col2:
                        st.metric("Risk Level", impact['risk_level'])
                    with col3:
                        st.metric("Recovery Time", f"{impact.get('recovery_months', 12):.0f} months")
                    with col4:
                        st.metric("VaR (95%)", f"{impact.get('var_95', 0)*100:.1f}%")
                # Fast mode already showed basic metrics above
                
                if use_llm:
                    # STEP 4: Generate AI recommendations (this may take 3-5 seconds)
                    st.info("🤖 Generating personalized AI recommendations... (this may take a few seconds)")
                    with st.spinner("🤖 AI is analyzing your portfolio..."):
                        try:
                            # Get ESG result if available
                            esg_data = st.session_state.get('esg_result') if esg_available else None
                            
                            # Generate recommendations with ESG integration
                            recs = explainer.generate_recommendations(portfolio_data, result, fast_mode=fast_mode, esg_result=esg_data)
                            
                            # Check if the response contains an error message
                            if recs.startswith("❌"):
                                st.warning(recs)
                                st.info("💡 **Tip**: You can get a free API key from [OpenRouter.ai](https://openrouter.ai/) to enable AI-powered recommendations.")
                            else:
                                st.success("✅ AI recommendations generated!")
                                st.markdown("### 🤖 AI-Powered Recommendations")
                                st.markdown(recs)
                        except Exception as e:
                            st.error(f"AI analysis failed: {e}")
                            st.markdown(f"""
                            **Basic Analysis:**
                            - Portfolio Impact: {impact['impact_percentage']:.1f}%
                            - Risk Level: {impact['risk_level']}
                            - Recovery Time: {impact['recovery_months']:.0f} months
                            
                            For detailed recommendations, please check your API key or try again later.
                            """)
                else:
                    # If AI is disabled, show encouraging message
                    st.info("💡 Enable 'AI-Powered Recommendations' in the sidebar for personalized advice!")
                    st.markdown("""
                    **📋 Basic Recommendations:**
                    - Review your risk tolerance based on the analysis above
                    - Consider diversification if risk level is High/Extreme  
                    - Maintain 6-12 months emergency fund
                    - Consult a financial advisor for personalized advice
                    """)
        
        else:
            # Comparison - fix for when compare_keys is empty
            if not compare_keys:
                st.warning("Please select at least one scenario to compare.")
            else:
                with st.spinner("Comparing scenarios..."):
                    comp_df = explainer.compare_scenarios(portfolio_data, compare_keys)
                    
                    st.subheader("📊 Multi-Scenario Comparison")
                    st.dataframe(comp_df.style.format({'impact_percentage': '{:.1f}%'}), width='stretch')
                    
                    try:
                        # Use the new radar chart for multi-scenario comparison
                        fig_radar = plot_gen.create_multi_scenario_radar(comp_df, "Multi-Scenario Risk Comparison")
                        st.pyplot(fig_radar)
                        plt.close(fig_radar)
                    except Exception as e:
                        st.warning(f"Enhanced comparison chart unavailable: {e}")
                        # Fallback to old comparison chart
                        try:
                            fig2 = plot_gen.create_scenario_comparison(comp_df)
                            st.pyplot(fig2)
                            plt.close(fig2)
                        except:
                            st.error("All comparison visualizations failed")
                    
                    # AI Summary for comparison
                    if use_llm and len(compare_keys) > 1:
                        st.markdown("---")
                        st.markdown("### 🤖 AI Multi-Scenario Analysis")
                        with st.spinner("Generating comparative insights..."):
                            try:
                                # Create a detailed summary for multiple scenarios
                                portfolio_value = portfolio_data[value_col].sum()
                                scenario_names_list = [scenario_names[k] for k in compare_keys]
                                
                                # Find worst, best, and median scenarios
                                worst_scenario = comp_df.loc[comp_df['impact_percentage'].idxmin()]
                                best_scenario = comp_df.loc[comp_df['impact_percentage'].idxmax()]
                                median_impact = comp_df['impact_percentage'].median()
                                
                                # Calculate financial impacts
                                worst_loss = portfolio_value * abs(worst_scenario['impact_percentage']) / 100
                                best_loss = portfolio_value * abs(best_scenario['impact_percentage']) / 100
                                impact_spread = abs(worst_scenario['impact_percentage'] - best_scenario['impact_percentage'])
                                
                                # Direct LLM call for cleaner output
                                try:
                                    from XFIN.utils import _get_openrouter_key
                                except ImportError:
                                    from utils import _get_openrouter_key
                                    
                                import requests
                                
                                comparison_prompt = f"""You are analyzing a portfolio stress test comparing {len(compare_keys)} different scenarios.

PORTFOLIO OVERVIEW:
• Total Value: ₹{portfolio_value:,.0f}
• Scenarios Analyzed: {', '.join(scenario_names_list)}

STRESS TEST RESULTS (sorted by severity):
{comp_df.sort_values('impact_percentage').to_string(index=False)}

KEY FINDINGS:
• Worst Case: {worst_scenario['scenario']} → {worst_scenario['impact_percentage']:.1f}% loss (₹{worst_loss:,.0f})
• Best Case: {best_scenario['scenario']} → {best_scenario['impact_percentage']:.1f}% loss (₹{best_loss:,.0f})
• Impact Range: {impact_spread:.1f}% difference between scenarios
• Median Impact: {median_impact:.1f}%
• Longest Recovery: {comp_df['recovery_months'].max():.0f} months

Provide a focused comparative analysis with these sections (use markdown headers):

## 🎯 SCENARIO IMPACT RANKING
Rank all {len(compare_keys)} scenarios from most to least severe. For EACH scenario, explain:
- Why this scenario affects the portfolio differently
- Specific vulnerabilities it exposes
- Financial impact in rupees

## ⚠️ BIGGEST THREAT
Identify THE ONE most dangerous scenario and explain:
- Why it's the worst case for this portfolio
- Which sectors/assets would be hit hardest
- When this scenario might occur

## 💪 PORTFOLIO RESILIENCE
Based on {impact_spread:.1f}% impact range:
- Is diversification effective?
- Which scenarios expose similar weaknesses?
- What recovery patterns emerge?

## 🔄 COMPARATIVE INSIGHTS
- Why is there a {impact_spread:.1f}% difference between best and worst?
- Which scenarios might occur together?
- Which is most likely in current Indian market conditions?

## 🛡️ STRATEGIC RECOMMENDATIONS
Provide THREE specific actions:
1. Primary hedge against worst scenario
2. Portfolio adjustment to reduce impact spread
3. Key monitoring indicator

## ⚡ PRIORITY ACTION
THE SINGLE MOST IMPORTANT action to take now.

Keep analysis comparative and specific to Indian markets. DO NOT add introductory text before the first header."""

                                # Use user-provided API key if available, otherwise get from environment
                                final_api_key = api_key if api_key else _get_openrouter_key()
                                
                                if not final_api_key:
                                    st.error("❌ No API key configured!")
                                    st.info("""
                                    **To enable AI analysis, you need an OpenRouter API key:**
                                    
                                    1. Get a free key from [OpenRouter.ai](https://openrouter.ai/)
                                    2. Either:
                                       - Enter it in the sidebar under "OpenRouter API Key"
                                       - Or create a `.env` file with: `OPENROUTER_API_KEY=your_key_here`
                                    3. Refresh the page
                                    """)
                                else:
                                    headers = {
                                        "Authorization": f"Bearer {final_api_key}",
                                        "Content-Type": "application/json"
                                    }
                                    data = {
                                        "model": "x-ai/grok-code-fast-1",
                                        "messages": [{"role": "user", "content": comparison_prompt}],
                                        "max_tokens": 3500,  # Increased for complete analysis
                                        "temperature": 0.7
                                    }
                                    
                                    response = requests.post(
                                        "https://openrouter.ai/api/v1/chat/completions",
                                        headers=headers,
                                        json=data,
                                        timeout=20  # Increased timeout for complete responses
                                    )
                                    
                                    response.raise_for_status()  # Raise exception for bad status codes
                                    
                                    if response.status_code == 200:
                                        comparison_analysis = response.json()['choices'][0]['message']['content']
                                        st.success("✅ Multi-scenario analysis complete!")
                                        st.markdown(comparison_analysis)
                                    else:
                                        st.warning(f"AI analysis returned status {response.status_code} - showing comparison table above")
                                    
                            except requests.exceptions.RequestException as req_err:
                                st.error(f"AI API connection failed: {str(req_err)}")
                                st.info("💡 Check your internet connection and API key configuration.")
                            except KeyError as key_err:
                                st.error(f"AI response format error: {str(key_err)}")
                                st.info("💡 The API response was unexpected. Please try again.")
                            except Exception as e:
                                st.error(f"AI comparison analysis failed: {str(e)}")
                                st.info("The scenario comparison data above shows the key metrics for manual analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("**XFIN Portfolio Stress Testing** - AI-Powered Risk Analysis for Smarter Investment Decisions")
    if use_llm:
        st.caption("🤖 Powered by Advanced AI for Personalized Investment Insights")
    else:
        st.caption("💡 Enable AI recommendations for personalized insights")

def launch_stress_dashboard():
    create_stress_dashboard()

if __name__ == "__main__":
    launch_stress_dashboard()