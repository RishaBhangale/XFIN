#!/usr/bin/env python3
"""
XFIN Credit Risk & XAI Example
===============================

Demonstrates credit risk assessment with Explainable AI:
- Credit score prediction
- SHAP/LIME explanations
- Adverse action notices
- Counterfactual recommendations

Author: XFIN Team
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENROUTER_API_KEY = None  # For AI-powered explanations (optional)

# =============================================================================
# XFIN IMPORTS
# =============================================================================

try:
    from XFIN.explainer import XAIExplainer
    from XFIN.credit_risk import CreditRiskAnalyzer
except ImportError:
    import sys
    sys.path.insert(0, '..')
    try:
        from XFIN.explainer import XAIExplainer
        from XFIN.credit_risk import CreditRiskAnalyzer
    except ImportError:
        print("⚠️  Credit risk module requires additional setup")
        print("   Run: pip install shap lime")
        XAIExplainer = None
        CreditRiskAnalyzer = None


def create_sample_applicant():
    """Create a sample loan applicant."""
    return {
        'age': 35,
        'income': 75000,
        'employment_years': 5,
        'debt_to_income': 0.35,
        'credit_utilization': 0.45,
        'num_credit_accounts': 4,
        'late_payments_last_2y': 1,
        'credit_history_months': 84,
        'loan_amount_requested': 250000,
        'existing_debt': 50000
    }


def analyze_credit_application(applicant: dict):
    """Analyze a credit application with XAI explanations."""
    
    print("\n" + "=" * 60)
    print("  📋 CREDIT APPLICATION ANALYSIS")
    print("=" * 60)
    
    # Display applicant info
    print("\n  Applicant Profile:")
    print("  " + "-" * 40)
    print(f"     Age: {applicant['age']} years")
    print(f"     Annual Income: ₹{applicant['income']:,}")
    print(f"     Employment: {applicant['employment_years']} years")
    print(f"     Debt-to-Income: {applicant['debt_to_income']*100:.1f}%")
    print(f"     Credit Utilization: {applicant['credit_utilization']*100:.1f}%")
    print(f"     Credit Accounts: {applicant['num_credit_accounts']}")
    print(f"     Late Payments (2y): {applicant['late_payments_last_2y']}")
    print(f"     Credit History: {applicant['credit_history_months']} months")
    print(f"     Loan Requested: ₹{applicant['loan_amount_requested']:,}")
    
    # Calculate basic risk metrics (simplified model)
    risk_score = calculate_risk_score(applicant)
    
    print("\n  📊 Risk Assessment:")
    print("  " + "-" * 40)
    print(f"     Risk Score: {risk_score:.1f}/100")
    
    if risk_score < 30:
        decision = "APPROVED"
        indicator = "🟢"
    elif risk_score < 50:
        decision = "CONDITIONAL APPROVAL"
        indicator = "🟡"
    else:
        decision = "DECLINED"
        indicator = "🔴"
    
    print(f"     Decision: {indicator} {decision}")
    
    return risk_score, decision


def calculate_risk_score(applicant: dict) -> float:
    """
    Calculate a simplified risk score.
    
    In production, this would use a trained ML model (GradientBoosting, XGBoost, etc.)
    """
    
    score = 50  # Base score
    
    # Income factor
    if applicant['income'] > 100000:
        score -= 15
    elif applicant['income'] > 50000:
        score -= 5
    else:
        score += 10
    
    # Employment stability
    if applicant['employment_years'] >= 5:
        score -= 10
    elif applicant['employment_years'] >= 2:
        score -= 5
    else:
        score += 15
    
    # Debt-to-income
    dti = applicant['debt_to_income']
    if dti < 0.30:
        score -= 10
    elif dti > 0.50:
        score += 20
    
    # Credit utilization
    util = applicant['credit_utilization']
    if util < 0.30:
        score -= 10
    elif util > 0.70:
        score += 15
    
    # Late payments
    late = applicant['late_payments_last_2y']
    score += late * 10
    
    # Credit history length
    if applicant['credit_history_months'] >= 60:
        score -= 10
    elif applicant['credit_history_months'] < 24:
        score += 15
    
    return max(0, min(100, score))


def generate_explanations(applicant: dict, risk_score: float):
    """Generate explanations for the credit decision."""
    
    print("\n" + "=" * 60)
    print("  🔍 DECISION EXPLANATIONS (XAI)")
    print("=" * 60)
    
    # Feature importance (simulated SHAP values)
    factors = []
    
    if applicant['debt_to_income'] > 0.40:
        factors.append(("High Debt-to-Income Ratio", "negative", 
                       f"{applicant['debt_to_income']*100:.1f}% exceeds 40% threshold"))
    
    if applicant['credit_utilization'] > 0.50:
        factors.append(("High Credit Utilization", "negative",
                       f"{applicant['credit_utilization']*100:.1f}% utilization"))
    
    if applicant['late_payments_last_2y'] > 0:
        factors.append(("Recent Late Payments", "negative",
                       f"{applicant['late_payments_last_2y']} late payment(s) in last 2 years"))
    
    if applicant['income'] > 60000:
        factors.append(("Strong Income", "positive",
                       f"₹{applicant['income']:,} annual income"))
    
    if applicant['employment_years'] >= 3:
        factors.append(("Stable Employment", "positive",
                       f"{applicant['employment_years']} years with employer"))
    
    if applicant['credit_history_months'] >= 48:
        factors.append(("Established Credit History", "positive",
                       f"{applicant['credit_history_months']} months of history"))
    
    # Display factors
    print("\n  📈 Positive Factors:")
    positive = [f for f in factors if f[1] == "positive"]
    if positive:
        for name, _, desc in positive:
            print(f"     🟢 {name}: {desc}")
    else:
        print("     (None identified)")
    
    print("\n  📉 Negative Factors:")
    negative = [f for f in factors if f[1] == "negative"]
    if negative:
        for name, _, desc in negative:
            print(f"     🔴 {name}: {desc}")
    else:
        print("     (None identified)")
    
    return factors


def generate_counterfactuals(applicant: dict, decision: str):
    """Generate counterfactual recommendations."""
    
    if decision == "APPROVED":
        return
    
    print("\n" + "=" * 60)
    print("  💡 COUNTERFACTUAL RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n  To improve chances of approval:\n")
    
    recommendations = []
    
    if applicant['debt_to_income'] > 0.40:
        target_income = applicant['existing_debt'] / 0.35
        recommendations.append(
            f"  1. Reduce debt-to-income ratio to <35%\n"
            f"     • Pay down ₹{int(applicant['existing_debt']*0.2):,} of existing debt\n"
            f"     • Or increase income to ₹{int(target_income):,}/year"
        )
    
    if applicant['credit_utilization'] > 0.50:
        recommendations.append(
            f"  2. Lower credit utilization to <30%\n"
            f"     • Current: {applicant['credit_utilization']*100:.0f}%\n"
            f"     • Target: 30%"
        )
    
    if applicant['late_payments_last_2y'] > 0:
        recommendations.append(
            f"  3. Maintain on-time payments for next 12 months\n"
            f"     • This will clear recent payment history"
        )
    
    if not recommendations:
        recommendations.append(
            "  • Consider a smaller loan amount\n"
            "  • Add a co-applicant with strong credit"
        )
    
    for rec in recommendations:
        print(rec)
        print()


def generate_adverse_action_notice(applicant: dict, decision: str, factors: list):
    """Generate regulatory-compliant adverse action notice."""
    
    if decision == "APPROVED":
        return
    
    print("\n" + "=" * 60)
    print("  📜 ADVERSE ACTION NOTICE")
    print("=" * 60)
    
    print("""
    NOTICE OF ADVERSE ACTION
    
    Date: [Current Date]
    Applicant: [Name Redacted]
    
    We regret to inform you that your application for credit
    has been declined or conditionally approved.
    
    Principal reason(s) for this decision:
    """)
    
    negative = [f for f in factors if f[1] == "negative"]
    for i, (name, _, desc) in enumerate(negative[:4], 1):  # Top 4 reasons
        print(f"    {i}. {name}")
    
    print("""
    You have the right to:
    • Request a free copy of your credit report
    • Dispute inaccurate information
    • Reapply after addressing the above factors
    
    For questions, contact: credit-support@company.com
    """)


def main():
    """Main entry point."""
    
    print("\n" + "=" * 60)
    print("  XFIN Credit Risk & XAI Example")
    print("=" * 60)
    
    if XAIExplainer is None:
        print("\n⚠️  Running in demo mode (full XAI requires shap/lime)")
    
    # Create sample applicant
    applicant = create_sample_applicant()
    
    # Analyze application
    risk_score, decision = analyze_credit_application(applicant)
    
    # Generate explanations
    factors = generate_explanations(applicant, risk_score)
    
    # Generate counterfactuals
    generate_counterfactuals(applicant, decision)
    
    # Generate adverse action notice if declined
    generate_adverse_action_notice(applicant, decision, factors)
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ Credit Analysis Complete!")
    print("=" * 60)
    print("\n🎯 Launch interactive dashboard: xfin credit")


if __name__ == "__main__":
    main()
