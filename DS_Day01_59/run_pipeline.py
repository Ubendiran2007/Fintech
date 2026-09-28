"""
run_pipeline.py
Master script — runs all steps in order
DS_Day01_59: Fintech — Where Are Digital Payments Failing?
"""
import sys
import io
# Force UTF-8 stdout so Unicode chars (e.g. currency symbols) print correctly on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.data_cleaning import clean, save_cleaned, PROCESSED_PATH
from src.eda import run_full_eda
from src.train_model import train_and_evaluate

REPORTS_DIR = os.path.join(BASE, 'outputs', 'reports')
METRICS_PATH = os.path.join(BASE, 'outputs', 'metrics', 'model_metrics.json')
RAW_PATH = os.path.join(BASE, 'data', 'raw', 'upi_transactions_2024.csv')


def generate_insights(eda_results: dict, metrics: dict) -> list:
    """Generate 5-7 evidence-based insights from actual computed values."""
    overall = eda_results['overall']
    bank_df = eda_results['bank_df']
    pm_df = eda_results['pm_df']
    dev_df = eda_results['dev_df']
    tp_df = eda_results['tp_df']
    hourly = eda_results['hourly']
    amount_stats = eda_results['amount_stats']
    daily = eda_results['daily']

    # Bank with highest failure rate (with reasonable sample)
    bank_sorted = bank_df.sort_values('failure_rate_pct', ascending=False)
    top_bank = bank_sorted.iloc[0]
    
    # Payment method with highest failure rate
    pm_sorted = pm_df.sort_values('failure_rate_pct', ascending=False)
    top_pm = pm_sorted.iloc[0]
    low_pm = pm_sorted.iloc[-1]

    # Highest failure time period
    top_period = tp_df.sort_values('failure_rate_pct', ascending=False).iloc[0]

    # Highest failure hour
    top_hour = hourly.sort_values('failure_rate_pct', ascending=False).iloc[0]

    # Device with highest failure
    top_device = dev_df.sort_values('failure_rate_pct', ascending=False).iloc[0]

    # Spikes
    spikes = daily[daily['is_spike'] == True]
    spike_text = (
        f"Statistical failure spikes (>rolling_mean + 2×STD) detected on {len(spikes)} day(s)."
        if len(spikes) > 0
        else "No statistical daily failure spikes detected based on 7-day rolling window analysis."
    )

    # Amount difference
    s_median = amount_stats['success']['median']
    f_median = amount_stats['failed']['median']
    amt_diff_pct = round((f_median - s_median) / s_median * 100, 1) if s_median > 0 else 0

    insights = [
        {
            'id': 1,
            'title': 'Overall Failure Rate',
            'text': (
                f"The overall transaction failure rate is {overall['failure_rate_pct']}%, "
                f"meaning approximately {round(overall['failure_rate_pct'])} out of every 100 "
                f"transactions fail. Total failures: {overall['failed_transactions']:,} "
                f"out of {overall['total_transactions']:,} transactions."
            )
        },
        {
            'id': 2,
            'title': 'Bank with Highest Failure Rate',
            'text': (
                f"Among sender banks, {top_bank['sender_bank']} records the highest observed "
                f"failure rate at {top_bank['failure_rate_pct']}% across {top_bank['transaction_count']:,} "
                f"transactions. Note: this is an observed association, not a proven causal relationship. "
                f"The overall average bank failure rate is {bank_df['failure_rate_pct'].mean():.2f}%."
            )
        },
        {
            'id': 3,
            'title': 'Payment Method Failure Rates',
            'text': (
                f"Transaction type '{top_pm['transaction_type']}' has the highest failure rate at "
                f"{top_pm['failure_rate_pct']}% ({top_pm['transaction_count']:,} transactions), "
                f"while '{low_pm['transaction_type']}' has the lowest at {low_pm['failure_rate_pct']}% "
                f"({low_pm['transaction_count']:,} transactions)."
            )
        },
        {
            'id': 4,
            'title': 'Peak Failure Time Period',
            'text': (
                f"The '{top_period['time_period']}' period shows the highest failure rate at "
                f"{top_period['failure_rate_pct']}% ({top_period['failed_count']:,} failures out of "
                f"{top_period['transaction_count']:,} transactions). "
                f"Peak failure hour: {int(top_hour['hour']):02d}:00 with {top_hour['failure_rate_pct']}% failure rate."
            )
        },
        {
            'id': 5,
            'title': 'Failure Spike Detection',
            'text': spike_text
        },
        {
            'id': 6,
            'title': 'Transaction Amount & Failure',
            'text': (
                f"Median transaction amount for successful transactions: INR {s_median:,.0f} vs "
                f"INR {f_median:,.0f} for failed transactions "
                f"({'higher' if amt_diff_pct > 0 else 'lower'} by {abs(amt_diff_pct)}%). "
                f"The difference is {'notable' if abs(amt_diff_pct) > 5 else 'marginal'}, "
                f"though this is a correlation, not a confirmed cause."
            )
        },
        {
            'id': 7,
            'title': 'Device Type Failure Pattern',
            'text': (
                f"Device '{top_device['device_type']}' shows the highest observed failure rate at "
                f"{top_device['failure_rate_pct']}% across {top_device['transaction_count']:,} transactions. "
                f"Device-specific failure differences should trigger compatibility and UX reviews "
                f"for higher-failure device categories."
            )
        },
    ]

    return insights


def generate_action_plan(eda_results: dict, insights: list) -> list:
    """Generate operational recommendations linked to evidence."""
    overall = eda_results['overall']
    bank_df = eda_results['bank_df']
    tp_df = eda_results['tp_df']
    dev_df = eda_results['dev_df']
    daily = eda_results['daily']
    spikes = daily[daily['is_spike'] == True]

    top_bank = bank_df.sort_values('failure_rate_pct', ascending=False).iloc[0]
    top_period = tp_df.sort_values('failure_rate_pct', ascending=False).iloc[0]
    top_device = dev_df.sort_values('failure_rate_pct', ascending=False).iloc[0]

    plan = [
        {
            'category': 'Bank Operations',
            'priority': 'HIGH',
            'recommendation': (
                f"Initiate a technical review with {top_bank['sender_bank']} (failure rate: "
                f"{top_bank['failure_rate_pct']}%) to identify root causes — network connectivity, "
                f"API timeouts, and authentication issues should be investigated first."
            ),
            'evidence': f"Bank analysis shows {top_bank['sender_bank']} exceeds the average failure rate of {bank_df['failure_rate_pct'].mean():.2f}%."
        },
        {
            'category': 'Payment Routing',
            'priority': 'MEDIUM',
            'recommendation': (
                "Implement intelligent payment routing and retry strategies for transaction types "
                "showing elevated failure rates. Consider fallback routing to alternate payment rails."
            ),
            'evidence': "Payment method analysis identified differential failure rates across transaction types."
        },
        {
            'category': 'Peak-Time Capacity',
            'priority': 'HIGH',
            'recommendation': (
                f"Review infrastructure capacity during the '{top_period['time_period']}' time slot "
                f"(failure rate: {top_period['failure_rate_pct']}%). "
                "Implement auto-scaling and pre-provisioned capacity for peak hours."
            ),
            'evidence': f"Time analysis shows '{top_period['time_period']}' has the highest failure rate."
        },
        {
            'category': 'Device Compatibility',
            'priority': 'MEDIUM',
            'recommendation': (
                f"Investigate '{top_device['device_type']}' device-specific failure patterns. "
                "Review SDK compatibility, browser/app version support, and UX flows for this device type."
            ),
            'evidence': f"Device analysis shows '{top_device['device_type']}' has the highest failure rate at {top_device['failure_rate_pct']}%."
        },
        {
            'category': 'Real-time Monitoring',
            'priority': 'HIGH',
            'recommendation': (
                "Deploy real-time failure-rate monitoring with automated threshold alerts. "
                "Set alert threshold at mean + 2×STD of rolling hourly failure rate. "
                f"Historical data shows {len(spikes)} spike day(s) that required immediate investigation."
            ),
            'evidence': 'Spike detection analysis identified statistically unusual failure periods.'
        },
        {
            'category': 'Data Quality & Fraud',
            'priority': 'LOW',
            'recommendation': (
                "Continue monitoring fraud_flag in parallel with transaction failures. "
                "Ensure fraudulent transaction patterns do not artificially inflate failure rates "
                "in operational reporting."
            ),
            'evidence': "Dataset includes fraud_flag; separating fraud from operational failures improves root-cause analysis accuracy."
        },
    ]
    return plan


def main():
    print("=" * 70)
    print("DS_Day01_59 — FINTECH: WHERE ARE DIGITAL PAYMENTS FAILING?")
    print("=" * 70)

    # STEP 1-3: Clean
    print("\n[STEP 1-3] Data Cleaning ...")
    df_clean = clean(RAW_PATH)
    save_cleaned(df_clean)

    # STEP 4-5: EDA
    print("\n[STEP 4-5] Exploratory Data Analysis ...")
    eda_results = run_full_eda(df_clean)

    # STEP 6-10: ML
    print("\n[STEP 6-10] Machine Learning ...")
    pipeline, metrics, fi_df = train_and_evaluate()

    # STEP 11: Insights
    print("\n[STEP 11] Generating Insights ...")
    insights = generate_insights(eda_results, metrics)
    action_plan = generate_action_plan(eda_results, insights)

    # Save insights and action plan
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, 'insights.json'), 'w') as f:
        json.dump(insights, f, indent=2)
    with open(os.path.join(REPORTS_DIR, 'action_plan.json'), 'w') as f:
        json.dump(action_plan, f, indent=2)

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT SUMMARY")
    print("=" * 70)
    for ins in insights:
        print(f"\n[INSIGHT {ins['id']}] {ins['title']}")
        safe_text = ins['text'].encode('ascii', errors='replace').decode('ascii')
        print(f"  {safe_text}")

    print(f"\n[ML METRICS]")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1-Score  : {metrics['f1_score']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"  PR-AUC    : {metrics['pr_auc']}")

    print(f"\n[TOP 5 FEATURES]")
    for _, row in fi_df.head(5).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")

    print(f"\n[ACTION PLAN]")
    for item in action_plan:
        print(f"  [{item['priority']}] {item['category']}: {item['recommendation'][:80]}...")

    print("\n[COMPLETE] All outputs saved to DS_Day01_59/outputs/")
    print("Run dashboard with: streamlit run DS_Day01_59/dashboard/app.py")


if __name__ == '__main__':
    main()
