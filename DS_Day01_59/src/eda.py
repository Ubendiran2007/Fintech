"""
eda.py
Phase 4 & 5 — Descriptive Analysis + Required Visualizations
DS_Day01_59: Fintech — Where Are Digital Payments Failing?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import json

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports')
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_transactions.csv')

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    'success': '#2ECC71',
    'failure': '#E74C3C',
    'primary': '#2C3E50',
    'accent': '#3498DB',
    'bg': '#FAFAFA',
    'grid': '#EEEEEE',
}

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': PALETTE['bg'],
    'figure.facecolor': PALETTE['bg'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': PALETTE['grid'],
    'grid.linewidth': 0.8,
})


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_overall_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    failed = int(df['failure'].sum())
    success = total - failed
    failure_rate = round(failed / total * 100, 2)
    success_rate = round(100 - failure_rate, 2)
    return {
        'total_transactions': total,
        'successful_transactions': success,
        'failed_transactions': failed,
        'success_rate_pct': success_rate,
        'failure_rate_pct': failure_rate,
    }


def compute_bank_metrics(df: pd.DataFrame) -> pd.DataFrame:
    bank_df = df.groupby('sender_bank').agg(
        transaction_count=('failure', 'count'),
        failed_count=('failure', 'sum'),
    ).reset_index()
    bank_df['success_count'] = bank_df['transaction_count'] - bank_df['failed_count']
    bank_df['failure_rate_pct'] = round(bank_df['failed_count'] / bank_df['transaction_count'] * 100, 2)
    bank_df['success_rate_pct'] = 100 - bank_df['failure_rate_pct']
    return bank_df.sort_values('transaction_count', ascending=False)


def compute_payment_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    pm_df = df.groupby('transaction_type').agg(
        transaction_count=('failure', 'count'),
        failed_count=('failure', 'sum'),
    ).reset_index()
    pm_df['success_count'] = pm_df['transaction_count'] - pm_df['failed_count']
    pm_df['failure_rate_pct'] = round(pm_df['failed_count'] / pm_df['transaction_count'] * 100, 2)
    pm_df['success_rate_pct'] = 100 - pm_df['failure_rate_pct']
    return pm_df.sort_values('transaction_count', ascending=False)


def compute_device_metrics(df: pd.DataFrame) -> pd.DataFrame:
    dev_df = df.groupby('device_type').agg(
        transaction_count=('failure', 'count'),
        failed_count=('failure', 'sum'),
    ).reset_index()
    dev_df['failure_rate_pct'] = round(dev_df['failed_count'] / dev_df['transaction_count'] * 100, 2)
    return dev_df.sort_values('failure_rate_pct', ascending=False)


def compute_location_metrics(df: pd.DataFrame) -> pd.DataFrame:
    loc_df = df.groupby('sender_state').agg(
        transaction_count=('failure', 'count'),
        failed_count=('failure', 'sum'),
    ).reset_index()
    loc_df['failure_rate_pct'] = round(loc_df['failed_count'] / loc_df['transaction_count'] * 100, 2)
    return loc_df.sort_values('failure_rate_pct', ascending=False)


def compute_hourly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.groupby('hour').agg(
        transaction_count=('failure', 'count'),
        failed_count=('failure', 'sum'),
    ).reset_index()
    hourly['failure_rate_pct'] = round(hourly['failed_count'] / hourly['transaction_count'] * 100, 2)
    return hourly.sort_values('hour')


def compute_time_period_metrics(df: pd.DataFrame) -> pd.DataFrame:
    period_order = ['Early Morning', 'Morning', 'Afternoon', 'Evening']
    tp_df = df.groupby('time_period').agg(
        transaction_count=('failure', 'count'),
        failed_count=('failure', 'sum'),
    ).reset_index()
    tp_df['failure_rate_pct'] = round(tp_df['failed_count'] / tp_df['transaction_count'] * 100, 2)
    tp_df['time_period'] = pd.Categorical(tp_df['time_period'], categories=period_order, ordered=True)
    return tp_df.sort_values('time_period')


def compute_amount_metrics(df: pd.DataFrame) -> dict:
    s = df[df['failure'] == 0]['amount_inr']
    f = df[df['failure'] == 1]['amount_inr']
    return {
        'success': {
            'mean': round(float(s.mean()), 2),
            'median': round(float(s.median()), 2),
            'std': round(float(s.std()), 2),
            'q25': round(float(s.quantile(0.25)), 2),
            'q75': round(float(s.quantile(0.75)), 2),
        },
        'failed': {
            'mean': round(float(f.mean()), 2),
            'median': round(float(f.median()), 2),
            'std': round(float(f.std()), 2),
            'q25': round(float(f.quantile(0.25)), 2),
            'q75': round(float(f.quantile(0.75)), 2),
        },
    }


def detect_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate failures by date.
    Flag days where failure_count > rolling_mean + 2 * rolling_std (window=7).
    """
    daily = df.groupby('transaction_date').agg(
        transaction_count=('failure', 'count'),
        failed_count=('failure', 'sum'),
    ).reset_index()
    daily['failure_rate_pct'] = round(daily['failed_count'] / daily['transaction_count'] * 100, 2)
    daily = daily.sort_values('transaction_date')
    daily['rolling_mean'] = daily['failed_count'].rolling(7, min_periods=3).mean()
    daily['rolling_std'] = daily['failed_count'].rolling(7, min_periods=3).std()
    daily['is_spike'] = daily['failed_count'] > (daily['rolling_mean'] + 2 * daily['rolling_std'])
    return daily


# ────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ────────────────────────────────────────────────────────────────────────────

def viz1_overall_outcome(overall: dict):
    """Viz 1 — Overall Transaction Success vs Failure (Donut + Bar)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Overall Transaction Success vs Failure', fontsize=16, fontweight='bold', color=PALETTE['primary'])

    # Donut
    ax = axes[0]
    labels = ['Success', 'Failed']
    sizes = [overall['successful_transactions'], overall['failed_transactions']]
    colors = [PALETTE['success'], PALETTE['failure']]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.2f%%',
        startangle=90, pctdistance=0.82,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')
    ax.set_title('Transaction Outcome Distribution', fontsize=13, pad=15)

    # Bar
    ax2 = axes[1]
    bars = ax2.bar(labels, sizes, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val, pct in zip(bars, sizes, [overall['success_rate_pct'], overall['failure_rate_pct']]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.01,
                 f'{val:,}\n({pct}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Transactions', fontsize=12)
    ax2.set_title('Transaction Counts', fontsize=13)
    ax2.set_ylim(0, max(sizes) * 1.15)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'viz1_overall_outcome.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {path}")
    return path


def viz2_failure_by_bank(bank_df: pd.DataFrame):
    """Viz 2 — Failure Rate by Bank (horizontal bar)."""
    sorted_df = bank_df.sort_values('failure_rate_pct', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Transaction Failure Rate by Bank', fontsize=16, fontweight='bold', color=PALETTE['primary'])

    bars = ax.barh(sorted_df['sender_bank'], sorted_df['failure_rate_pct'],
                   color=PALETTE['failure'], alpha=0.85, edgecolor='white', linewidth=1.2)
    for bar, count in zip(bars, sorted_df['transaction_count']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.2f}%  (n={count:,})',
                va='center', fontsize=10, color=PALETTE['primary'])

    ax.set_xlabel('Failure Rate (%)', fontsize=12)
    ax.set_ylabel('Sender Bank', fontsize=12)
    ax.set_xlim(0, sorted_df['failure_rate_pct'].max() * 1.25)
    ax.axvline(bank_df['failure_rate_pct'].mean(), color=PALETTE['accent'],
               linestyle='--', linewidth=1.5, label=f"Avg: {bank_df['failure_rate_pct'].mean():.2f}%")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'viz2_failure_by_bank.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {path}")
    return path


def viz3_failure_by_payment_method(pm_df: pd.DataFrame):
    """Viz 3 — Failure Rate by Payment Method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Transaction Failure Rate by Payment Method', fontsize=16, fontweight='bold', color=PALETTE['primary'])

    # Failure rate
    ax = axes[0]
    colors = [PALETTE['failure'] if r == pm_df['failure_rate_pct'].max() else PALETTE['accent']
              for r in pm_df['failure_rate_pct']]
    bars = ax.bar(pm_df['transaction_type'], pm_df['failure_rate_pct'], color=colors,
                  edgecolor='white', linewidth=1.5, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Failure Rate (%)', fontsize=12)
    ax.set_xlabel('Transaction Type', fontsize=12)
    ax.set_title('Failure Rate', fontsize=13)

    # Volume
    ax2 = axes[1]
    ax2.bar(pm_df['transaction_type'], pm_df['transaction_count'], color=PALETTE['primary'],
            alpha=0.8, edgecolor='white', linewidth=1.5, width=0.5)
    for bar in ax2.patches:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pm_df['transaction_count'].max() * 0.01,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=11)
    ax2.set_ylabel('Transaction Count', fontsize=12)
    ax2.set_xlabel('Transaction Type', fontsize=12)
    ax2.set_title('Transaction Volume', fontsize=13)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'viz3_failure_by_payment_method.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {path}")
    return path


def viz4_failure_by_hour(hourly: pd.DataFrame):
    """Viz 4 — Transaction Failure Rate by Hour (line + volume bars)."""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.suptitle('Transaction Failure Rate by Hour of Day', fontsize=16, fontweight='bold', color=PALETTE['primary'])

    color_vol = PALETTE['accent']
    color_rate = PALETTE['failure']

    ax2 = ax1.twinx()
    ax1.bar(hourly['hour'], hourly['transaction_count'], color=color_vol, alpha=0.35,
            label='Transaction Volume', width=0.8)
    ax2.plot(hourly['hour'], hourly['failure_rate_pct'], color=color_rate, linewidth=2.5,
             marker='o', markersize=5, label='Failure Rate %')
    ax2.fill_between(hourly['hour'], hourly['failure_rate_pct'], alpha=0.12, color=color_rate)

    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Transaction Volume', fontsize=12, color=color_vol)
    ax2.set_ylabel('Failure Rate (%)', fontsize=12, color=color_rate)
    ax1.set_xticks(range(0, 24))
    ax2.set_ylim(0, hourly['failure_rate_pct'].max() * 1.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'viz4_failure_by_hour.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {path}")
    return path


def viz5_amount_distribution(df: pd.DataFrame):
    """Viz 5 — Amount distribution: Success vs Failure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Transaction Amount Distribution: Success vs Failure', fontsize=16,
                 fontweight='bold', color=PALETTE['primary'])

    success_amt = df[df['failure'] == 0]['amount_inr']
    failed_amt = df[df['failure'] == 1]['amount_inr']

    # Histograms
    ax = axes[0]
    ax.hist(success_amt, bins=60, alpha=0.65, color=PALETTE['success'], label='Success', density=True)
    ax.hist(failed_amt, bins=60, alpha=0.65, color=PALETTE['failure'], label='Failed', density=True)
    ax.set_xlabel('Amount (INR)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Amount Histogram', fontsize=13)
    ax.legend(fontsize=10)

    # Boxplot
    ax2 = axes[1]
    data_to_plot = [success_amt.values, failed_amt.values]
    bp = ax2.boxplot(data_to_plot, patch_artist=True, tick_labels=['Success', 'Failed'],
                     showfliers=False, widths=0.5)
    for patch, color in zip(bp['boxes'], [PALETTE['success'], PALETTE['failure']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Amount (INR)', fontsize=12)
    ax2.set_title('Boxplot (outliers hidden for clarity)', fontsize=13)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'viz5_amount_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {path}")
    return path


def viz6_device_failure(dev_df: pd.DataFrame):
    """Viz 6 — Device failure rate."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Transaction Failure Rate by Device Type', fontsize=16, fontweight='bold', color=PALETTE['primary'])

    colors = [PALETTE['failure'] if r == dev_df['failure_rate_pct'].max() else PALETTE['accent']
              for r in dev_df['failure_rate_pct']]
    bars = ax.bar(dev_df['device_type'], dev_df['failure_rate_pct'], color=colors,
                  edgecolor='white', linewidth=1.5, width=0.5)
    for bar, count in zip(bars, dev_df['transaction_count']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{bar.get_height():.2f}%\n(n={count:,})', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Failure Rate (%)', fontsize=12)
    ax.set_xlabel('Device Type', fontsize=12)
    ax.set_ylim(0, dev_df['failure_rate_pct'].max() * 1.25)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'viz6_device_failure.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {path}")
    return path


def viz7_spike_detection(daily: pd.DataFrame):
    """Viz 7 — Daily failure spike detection."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle('Daily Transaction Failure Count & Spike Detection',
                 fontsize=16, fontweight='bold', color=PALETTE['primary'])

    ax.plot(daily['transaction_date'], daily['failed_count'], color=PALETTE['failure'],
            linewidth=1.5, alpha=0.8, label='Daily Failures')
    ax.plot(daily['transaction_date'], daily['rolling_mean'], color=PALETTE['accent'],
            linewidth=2, linestyle='--', label='7-Day Rolling Mean')
    if daily['rolling_std'].notna().any():
        upper = daily['rolling_mean'] + 2 * daily['rolling_std']
        ax.fill_between(daily['transaction_date'], daily['rolling_mean'], upper,
                        alpha=0.15, color=PALETTE['accent'], label='±2 STD Band')

    spikes = daily[daily['is_spike'] == True]
    if len(spikes) > 0:
        ax.scatter(spikes['transaction_date'], spikes['failed_count'],
                   color='orange', zorder=5, s=80, label=f'Spike ({len(spikes)} days)')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Failed Transaction Count', fontsize=12)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'viz7_spike_detection.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {path}")
    return path


def run_full_eda(df: pd.DataFrame) -> dict:
    ensure_dirs()

    overall = compute_overall_metrics(df)
    bank_df = compute_bank_metrics(df)
    pm_df = compute_payment_method_metrics(df)
    dev_df = compute_device_metrics(df)
    loc_df = compute_location_metrics(df)
    hourly = compute_hourly_metrics(df)
    tp_df = compute_time_period_metrics(df)
    amount_stats = compute_amount_metrics(df)
    daily = detect_spikes(df)

    # Save summary tables
    bank_df.to_csv(os.path.join(REPORTS_DIR, 'bank_analysis.csv'), index=False)
    pm_df.to_csv(os.path.join(REPORTS_DIR, 'payment_method_analysis.csv'), index=False)
    dev_df.to_csv(os.path.join(REPORTS_DIR, 'device_analysis.csv'), index=False)
    loc_df.to_csv(os.path.join(REPORTS_DIR, 'location_analysis.csv'), index=False)
    hourly.to_csv(os.path.join(REPORTS_DIR, 'hourly_analysis.csv'), index=False)
    tp_df.to_csv(os.path.join(REPORTS_DIR, 'time_period_analysis.csv'), index=False)
    daily.to_csv(os.path.join(REPORTS_DIR, 'daily_spike_analysis.csv'), index=False)

    with open(os.path.join(REPORTS_DIR, 'amount_stats.json'), 'w') as f:
        json.dump(amount_stats, f, indent=2)

    # Visualizations
    viz1_overall_outcome(overall)
    viz2_failure_by_bank(bank_df)
    viz3_failure_by_payment_method(pm_df)
    viz4_failure_by_hour(hourly)
    viz5_amount_distribution(df)
    viz6_device_failure(dev_df)
    viz7_spike_detection(daily)

    print("\n=== EDA SUMMARY ===")
    print(f"Total transactions: {overall['total_transactions']:,}")
    print(f"Success rate: {overall['success_rate_pct']}%")
    print(f"Failure rate: {overall['failure_rate_pct']}%")
    print(f"\nBank with highest failure rate:\n{bank_df.sort_values('failure_rate_pct', ascending=False).head(3)[['sender_bank','failure_rate_pct','transaction_count']]}")
    print(f"\nPayment method analysis:\n{pm_df[['transaction_type','failure_rate_pct','transaction_count']]}")
    print(f"\nDevice analysis:\n{dev_df}")
    print(f"\nTime period failure rates:\n{tp_df[['time_period','failure_rate_pct','transaction_count']]}")

    spikes = daily[daily['is_spike'] == True]
    print(f"\nSpike days detected: {len(spikes)}")
    if len(spikes) > 0:
        print(spikes[['transaction_date', 'failed_count', 'rolling_mean']].head(5))

    return {
        'overall': overall,
        'bank_df': bank_df,
        'pm_df': pm_df,
        'dev_df': dev_df,
        'loc_df': loc_df,
        'hourly': hourly,
        'tp_df': tp_df,
        'amount_stats': amount_stats,
        'daily': daily,
    }


if __name__ == '__main__':
    df = pd.read_csv(PROCESSED_PATH)
    if 'transaction_date' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['transaction_date'] = df['timestamp'].dt.date
    results = run_full_eda(df)
