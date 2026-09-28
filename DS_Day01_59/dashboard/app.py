"""
app.py — Digital Payments Failure Intelligence Dashboard
DS_Day01_59: Fintech — Where Are Digital Payments Failing?

Run with:  streamlit run dashboard/app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path setup ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

DATA_PATH = os.path.join(BASE, 'data', 'processed', 'cleaned_transactions.csv')
METRICS_PATH = os.path.join(BASE, 'outputs', 'metrics', 'model_metrics.json')
INSIGHTS_PATH = os.path.join(BASE, 'outputs', 'reports', 'insights.json')
ACTION_PATH = os.path.join(BASE, 'outputs', 'reports', 'action_plan.json')
FI_PATH = os.path.join(BASE, 'outputs', 'metrics', 'feature_importance.csv')

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Payments Failure Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Main background: 60% White */
.stApp { background: #F8F9FA; color: #333333; }

/* Sidebar: 20% Dark Green */
[data-testid="stSidebar"] {
    background: #0B3D2E !important;
    border-right: 1px solid rgba(0,0,0,0.1);
}
[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #90EE90 !important; font-weight: 600 !important; }

/* Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    text-align: center;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 15px rgba(0,0,0,0.1);
}
.kpi-value { font-size: 1.5rem; font-weight: 700; margin: 8px 0 4px 0; }
.kpi-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #555555; }
.kpi-sub { font-size: 0.72rem; color: #888888; margin-top: 2px; }

/* Section headers: 10% Light Green accent */
.section-header {
    background: rgba(144,238,144,0.15);
    border-left: 4px solid #90EE90;
    padding: 10px 18px;
    border-radius: 0 8px 8px 0;
    margin: 24px 0 16px 0;
}
.section-header h2 { color: #0B3D2E; margin: 0; font-size: 1.3rem; font-weight: 600; }
.section-header p { color: #555555; margin: 4px 0 0 0; font-size: 0.85rem; }

/* Insight cards */
.insight-card {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.insight-number {
    display: inline-block;
    background: #90EE90;
    color: #0B3D2E;
    width: 28px; height: 28px;
    border-radius: 50%;
    text-align: center;
    line-height: 28px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 10px;
}
.insight-title { color: #0B3D2E; font-weight: 600; font-size: 1rem; }
.insight-text { color: #333333; font-size: 0.88rem; line-height: 1.6; margin-top: 8px; }

/* Action plan */
.action-card {
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.05);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border-left: 4px solid #2ECC71;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.action-card.high { border-left-color: #E74C3C; }
.action-card.medium { border-left-color: #F39C12; }
.action-card.low { border-left-color: #2ECC71; }
.priority-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 8px;
}
.badge-high { background: #E74C3C; color: white; }
.badge-medium { background: #F39C12; color: white; }
.badge-low { background: #2ECC71; color: white; }

/* Metric rows */
.metric-row {
    display: flex; gap: 12px; flex-wrap: wrap;
}

/* Title: 10% Other color (Orange) accent */
.dashboard-title {
    color: #F39C12;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.2;
}
.dashboard-subtitle {
    color: #666666;
    font-size: 1rem;
    margin-top: 6px;
}

/* Charts */
.js-plotly-plot { border-radius: 12px !important; }

/* Divider */
.fancy-divider { border: none; height: 1px; background: rgba(0,0,0,0.1); margin: 24px 0; }

/* Streamlit overrides */
h1, h2, h3, h4 { color: #0B3D2E !important; }
p, li, span { color: #333333 !important; }
.stMetric label { color: #666666 !important; }
.stMetric [data-testid="metric-container"] { background: #FFFFFF; border: 1px solid rgba(0,0,0,0.1); border-radius: 12px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['transaction_date'] = df['timestamp'].dt.date
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'time_period' not in df.columns:
        def tp(h):
            if h < 6: return 'Early Morning'
            elif h < 12: return 'Morning'
            elif h < 18: return 'Afternoon'
            else: return 'Evening'
        df['time_period'] = df['hour'].apply(tp)
    return df


@st.cache_data(show_spinner=False)
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_insights():
    if os.path.exists(INSIGHTS_PATH):
        with open(INSIGHTS_PATH) as f:
            return json.load(f)
    return []


@st.cache_data(show_spinner=False)
def load_action_plan():
    if os.path.exists(ACTION_PATH):
        with open(ACTION_PATH) as f:
            return json.load(f)
    return []


@st.cache_data(show_spinner=False)
def load_feature_importance():
    if os.path.exists(FI_PATH):
        return pd.read_csv(FI_PATH)
    return pd.DataFrame()


# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,1)',
    font=dict(family='Inter', color='#333333', size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
    xaxis=dict(gridcolor='rgba(0,0,0,0.06)', zerolinecolor='rgba(0,0,0,0.06)'),
    yaxis=dict(gridcolor='rgba(0,0,0,0.06)', zerolinecolor='rgba(0,0,0,0.06)'),
)
COLOR_SUCCESS = '#2ECC71'
COLOR_FAIL = '#E74C3C'
COLOR_ACCENT = '#90EE90'
COLOR_WARN = '#F39C12'
DISCRETE_PALETTE = [COLOR_ACCENT, COLOR_FAIL, COLOR_WARN, '#0B3D2E', '#F39C12', '#90EE90']


def apply_layout(fig, title=''):
    fig.update_layout(**PLOT_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color='#0B3D2E'), x=0.02))
    return fig


# ── Helper: compute metrics on filtered df ─────────────────────────────────
def compute_kpis(df):
    total = len(df)
    failed = int(df['failure'].sum())
    success = total - failed
    fr = round(failed / total * 100, 2) if total else 0
    sr = round(100 - fr, 2)
    avg_amt = round(df['amount_inr'].mean(), 0) if total else 0
    top_period = (
        df.groupby('time_period')['failure'].mean()
        .idxmax() if total else 'N/A'
    )
    return total, success, failed, sr, fr, avg_amt, top_period


# ── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    # Load data
    with st.spinner('Loading data...'):
        df_raw = load_data()
        metrics = load_metrics()
        insights = load_insights()
        action_plan = load_action_plan()
        fi_df = load_feature_importance()

    # ── SIDEBAR FILTERS ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 Filters")
        st.markdown("---")

        all_banks = sorted(df_raw['sender_bank'].dropna().unique())
        all_methods = sorted(df_raw['transaction_type'].dropna().unique())
        all_devices = sorted(df_raw['device_type'].dropna().unique())
        all_states = sorted(df_raw['sender_state'].dropna().unique())
        all_periods = ['Early Morning', 'Morning', 'Afternoon', 'Evening']
        all_statuses = sorted(df_raw['transaction_status_clean'].dropna().unique())

        sel_banks = st.multiselect("🏦 Bank", all_banks, default=all_banks, key='banks')
        sel_methods = st.multiselect("💳 Transaction Type", all_methods, default=all_methods, key='methods')
        sel_devices = st.multiselect("📱 Device", all_devices, default=all_devices, key='devices')
        sel_states = st.multiselect("📍 State", all_states, default=all_states, key='states')
        sel_periods = st.multiselect("⏰ Time Period", all_periods, default=all_periods, key='periods')
        sel_status = st.multiselect("📊 Status", all_statuses, default=all_statuses, key='status')

        st.markdown("---")
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.rerun()

        st.markdown("---")
        st.markdown("### 📅 Dataset")
        st.markdown(f"**Total Records:** {len(df_raw):,}")
        st.markdown(f"**Date Range:** {df_raw['transaction_date'].min()} → {df_raw['transaction_date'].max()}")

    # ── Apply filters ────────────────────────────────────────────────────────
    mask = (
        df_raw['sender_bank'].isin(sel_banks) &
        df_raw['transaction_type'].isin(sel_methods) &
        df_raw['device_type'].isin(sel_devices) &
        df_raw['sender_state'].isin(sel_states) &
        df_raw['time_period'].isin(sel_periods) &
        df_raw['transaction_status_clean'].isin(sel_status)
    )
    df = df_raw[mask]

    if len(df) == 0:
        st.warning("⚠️ No data matches the current filters. Please reset filters.")
        return

    total, success, failed, sr, fr, avg_amt, top_period = compute_kpis(df)

    # ── HEADER ───────────────────────────────────────────────────────────────
    st.markdown('<h1 class="dashboard-title">💳 Digital Payments Failure Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Operational monitoring and failure analysis · DS_Day01_59</p>', unsafe_allow_html=True)
    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — EXECUTIVE OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>📊 Executive Overview</h2>
        <p>High-level KPIs and transaction outcome distribution</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_data = [
        (c1, "Total Transactions", f"{total:,}", "#F39C12", "All filtered records"),
        (c2, "Success Rate", f"{sr}%", "#2ECC71", f"{success:,} successful"),
        (c3, "Failure Rate", f"{fr}%", "#E74C3C", f"{failed:,} failed"),
        (c4, "Failed Transactions", f"{failed:,}", "#F39C12", "Count of failures"),
        (c5, "Avg Amount", f"₹{avg_amt:,.0f}", "#0B3D2E", "Mean transaction value"),
        (c6, "Peak Failure Period", top_period, "#F39C12", "Highest failure time"),
    ]
    for col, label, val, color, sub in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview charts
    col_a, col_b = st.columns([1, 2])
    with col_a:
        fig_pie = go.Figure(go.Pie(
            labels=['Success', 'Failed'],
            values=[success, failed],
            hole=0.55,
            marker=dict(colors=[COLOR_SUCCESS, COLOR_FAIL]),
            textinfo='label+percent',
            textfont=dict(size=13),
        ))
        apply_layout(fig_pie, 'Transaction Outcome')
        fig_pie.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # Daily volume line
        daily = df.groupby('transaction_date').agg(
            total=('failure', 'count'),
            failed=('failure', 'sum')
        ).reset_index()
        daily['success'] = daily['total'] - daily['failed']

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=daily['transaction_date'], y=daily['total'],
            name='Total', line=dict(color=COLOR_ACCENT, width=2),
            fill='tozeroy', fillcolor='rgba(100,160,255,0.1)'
        ))
        fig_vol.add_trace(go.Scatter(
            x=daily['transaction_date'], y=daily['failed'],
            name='Failed', line=dict(color=COLOR_FAIL, width=1.5)
        ))
        apply_layout(fig_vol, 'Daily Transaction Volume & Failures')
        fig_vol.update_layout(height=320, hovermode='x unified')
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — BANK PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>🏦 Bank Performance Analysis</h2>
        <p>Failure rates and transaction volumes by sender bank</p>
    </div>
    """, unsafe_allow_html=True)

    bank_df = df.groupby('sender_bank').agg(
        total=('failure', 'count'),
        failed=('failure', 'sum')
    ).reset_index()
    bank_df['failure_rate'] = round(bank_df['failed'] / bank_df['total'] * 100, 2)
    bank_df['success'] = bank_df['total'] - bank_df['failed']
    bank_df = bank_df.sort_values('failure_rate', ascending=True)
    avg_fr = round(bank_df['failed'].sum() / bank_df['total'].sum() * 100, 2)

    col1, col2 = st.columns(2)
    with col1:
        fig_bank = go.Figure()
        fig_bank.add_trace(go.Bar(
            y=bank_df['sender_bank'], x=bank_df['failure_rate'],
            orientation='h',
            marker=dict(
                color=bank_df['failure_rate'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title='Fail%', thickness=10)
            ),
            text=[f"{r:.2f}% (n={n:,})" for r, n in zip(bank_df['failure_rate'], bank_df['total'])],
            textposition='outside', textfont=dict(size=11),
        ))
        fig_bank.add_vline(x=avg_fr, line_dash='dash', line_color=COLOR_ACCENT,
                           annotation_text=f"Avg {avg_fr}%", annotation_position="top right")
        apply_layout(fig_bank, 'Failure Rate by Bank (%)')
        fig_bank.update_layout(height=380)
        st.plotly_chart(fig_bank, use_container_width=True)

    with col2:
        fig_bank_vol = go.Figure(data=[
            go.Bar(name='Success', y=bank_df['sender_bank'], x=bank_df['success'],
                   orientation='h', marker_color=COLOR_SUCCESS, opacity=0.85),
            go.Bar(name='Failed', y=bank_df['sender_bank'], x=bank_df['failed'],
                   orientation='h', marker_color=COLOR_FAIL, opacity=0.85),
        ])
        fig_bank_vol.update_layout(barmode='stack')
        apply_layout(fig_bank_vol, 'Success vs Failure by Bank')
        fig_bank_vol.update_layout(height=380)
        st.plotly_chart(fig_bank_vol, use_container_width=True)

    # Bank table
    with st.expander("📋 Bank Summary Table"):
        st.dataframe(
            bank_df.sort_values('failure_rate', ascending=False)[
                ['sender_bank', 'total', 'success', 'failed', 'failure_rate']
            ].rename(columns={
                'sender_bank': 'Bank', 'total': 'Total', 'success': 'Success',
                'failed': 'Failed', 'failure_rate': 'Failure Rate (%)'
            }),
            use_container_width=True, hide_index=True
        )

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — PAYMENT METHOD
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>💳 Payment Method Analysis</h2>
        <p>Failure rates and volumes by transaction type</p>
    </div>
    """, unsafe_allow_html=True)

    pm_df = df.groupby('transaction_type').agg(
        total=('failure', 'count'), failed=('failure', 'sum')
    ).reset_index()
    pm_df['failure_rate'] = round(pm_df['failed'] / pm_df['total'] * 100, 2)
    pm_df['success'] = pm_df['total'] - pm_df['failed']

    col3, col4 = st.columns(2)
    with col3:
        fig_pm = px.bar(
            pm_df.sort_values('failure_rate', ascending=False),
            x='transaction_type', y='failure_rate',
            color='failure_rate', color_continuous_scale='Reds',
            text=[f'{r:.2f}%' for r in pm_df.sort_values('failure_rate', ascending=False)['failure_rate']],
            labels={'transaction_type': 'Transaction Type', 'failure_rate': 'Failure Rate (%)'},
        )
        fig_pm.update_traces(textposition='outside')
        apply_layout(fig_pm, 'Failure Rate by Transaction Type')
        fig_pm.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_pm, use_container_width=True)

    with col4:
        fig_pm_vol = px.bar(
            pm_df, x='transaction_type', y=['success', 'failed'],
            barmode='stack',
            color_discrete_map={'success': COLOR_SUCCESS, 'failed': COLOR_FAIL},
            labels={'transaction_type': 'Type', 'value': 'Count'},
        )
        apply_layout(fig_pm_vol, 'Volume by Transaction Type')
        fig_pm_vol.update_layout(height=360)
        st.plotly_chart(fig_pm_vol, use_container_width=True)

    # Merchant category
    cat_df = df.groupby('merchant_category').agg(
        total=('failure', 'count'), failed=('failure', 'sum')
    ).reset_index()
    cat_df['failure_rate'] = round(cat_df['failed'] / cat_df['total'] * 100, 2)
    cat_df = cat_df.sort_values('failure_rate', ascending=False)

    fig_cat = px.bar(
        cat_df, x='merchant_category', y='failure_rate',
        color='failure_rate', color_continuous_scale='RdYlGn_r',
        text=[f'{r:.2f}%' for r in cat_df['failure_rate']],
        labels={'merchant_category': 'Merchant Category', 'failure_rate': 'Failure Rate (%)'},
    )
    fig_cat.update_traces(textposition='outside')
    apply_layout(fig_cat, 'Failure Rate by Merchant Category')
    fig_cat.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — TIME ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>⏰ Time Analysis</h2>
        <p>Hourly patterns and time-period breakdown</p>
    </div>
    """, unsafe_allow_html=True)

    hourly = df.groupby('hour').agg(
        total=('failure', 'count'), failed=('failure', 'sum')
    ).reset_index()
    hourly['failure_rate'] = round(hourly['failed'] / hourly['total'] * 100, 2)

    fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hourly.add_trace(go.Bar(
        x=hourly['hour'], y=hourly['total'],
        name='Volume', marker_color='rgba(100,160,255,0.3)',
    ), secondary_y=False)
    fig_hourly.add_trace(go.Scatter(
        x=hourly['hour'], y=hourly['failure_rate'],
        name='Failure Rate %', mode='lines+markers',
        line=dict(color=COLOR_FAIL, width=2.5),
        marker=dict(size=6, color=COLOR_FAIL),
    ), secondary_y=True)
    apply_layout(fig_hourly, 'Hourly Transaction Volume & Failure Rate')
    fig_hourly.update_layout(
        height=380, hovermode='x unified',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
    )
    fig_hourly.update_yaxes(title_text="Volume", secondary_y=False, gridcolor='rgba(255,255,255,0.06)')
    fig_hourly.update_yaxes(title_text="Failure Rate (%)", secondary_y=True, gridcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hourly, use_container_width=True)

    # Time period
    period_order = ['Early Morning', 'Morning', 'Afternoon', 'Evening']
    tp_df = df.groupby('time_period').agg(
        total=('failure', 'count'), failed=('failure', 'sum')
    ).reset_index()
    tp_df['failure_rate'] = round(tp_df['failed'] / tp_df['total'] * 100, 2)
    tp_df['time_period'] = pd.Categorical(tp_df['time_period'], categories=period_order, ordered=True)
    tp_df = tp_df.sort_values('time_period')

    col5, col6 = st.columns(2)
    with col5:
        fig_tp = px.bar(
            tp_df, x='time_period', y='failure_rate',
            color='failure_rate', color_continuous_scale='Reds',
            text=[f'{r:.2f}%' for r in tp_df['failure_rate']],
            labels={'time_period': 'Time Period', 'failure_rate': 'Failure Rate (%)'},
        )
        fig_tp.update_traces(textposition='outside')
        apply_layout(fig_tp, 'Failure Rate by Time Period')
        fig_tp.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_tp, use_container_width=True)

    with col6:
        # Spike detection
        daily_spike = df.groupby('transaction_date').agg(
            total=('failure', 'count'), failed=('failure', 'sum')
        ).reset_index().sort_values('transaction_date')
        daily_spike['rolling_mean'] = daily_spike['failed'].rolling(7, min_periods=3).mean()
        daily_spike['rolling_std'] = daily_spike['failed'].rolling(7, min_periods=3).std()
        daily_spike['upper_band'] = daily_spike['rolling_mean'] + 2 * daily_spike['rolling_std']
        daily_spike['is_spike'] = daily_spike['failed'] > daily_spike['upper_band']
        spikes = daily_spike[daily_spike['is_spike'] == True]

        fig_spike = go.Figure()
        fig_spike.add_trace(go.Scatter(
            x=daily_spike['transaction_date'], y=daily_spike['failed'],
            name='Daily Failures', line=dict(color=COLOR_FAIL, width=1.5)
        ))
        fig_spike.add_trace(go.Scatter(
            x=daily_spike['transaction_date'], y=daily_spike['rolling_mean'],
            name='7-Day Mean', line=dict(color=COLOR_ACCENT, dash='dash', width=2)
        ))
        if len(spikes) > 0:
            fig_spike.add_trace(go.Scatter(
                x=spikes['transaction_date'], y=spikes['failed'],
                name=f'Spike ({len(spikes)})',
                mode='markers',
                marker=dict(color=COLOR_WARN, size=10, symbol='star')
            ))
        apply_layout(fig_spike, f'Failure Spikes ({len(spikes)} detected)')
        fig_spike.update_layout(height=340, hovermode='x unified')
        st.plotly_chart(fig_spike, use_container_width=True)

    # Heatmap: Hour × Day
    hm_data = df.groupby(['day_of_week', 'hour'])['failure'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hm_pivot = hm_data.pivot(index='day_of_week', columns='hour', values='failure')
    hm_pivot = hm_pivot.reindex([d for d in day_order if d in hm_pivot.index])

    fig_hm = go.Figure(go.Heatmap(
        z=hm_pivot.values * 100,
        x=hm_pivot.columns,
        y=hm_pivot.index,
        colorscale='RdYlGn_r',
        colorbar=dict(title='Fail %', thickness=15),
        hoverongaps=False,
    ))
    apply_layout(fig_hm, 'Failure Rate Heatmap: Day × Hour (%)')
    fig_hm.update_layout(height=340)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — DEVICE & LOCATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>📱 Device & Location Analysis</h2>
        <p>Failure rates by device type and geographic state</p>
    </div>
    """, unsafe_allow_html=True)

    dev_df = df.groupby('device_type').agg(
        total=('failure', 'count'), failed=('failure', 'sum')
    ).reset_index()
    dev_df['failure_rate'] = round(dev_df['failed'] / dev_df['total'] * 100, 2)

    loc_df = df.groupby('sender_state').agg(
        total=('failure', 'count'), failed=('failure', 'sum')
    ).reset_index()
    loc_df['failure_rate'] = round(loc_df['failed'] / loc_df['total'] * 100, 2)
    loc_df = loc_df.sort_values('failure_rate', ascending=False)

    col7, col8 = st.columns(2)
    with col7:
        fig_dev = px.bar(
            dev_df.sort_values('failure_rate', ascending=False),
            x='device_type', y='failure_rate',
            color='failure_rate', color_continuous_scale='Reds',
            text=[f'{r:.2f}%\n(n={n:,})' for r, n in zip(
                dev_df.sort_values('failure_rate', ascending=False)['failure_rate'],
                dev_df.sort_values('failure_rate', ascending=False)['total']
            )],
            labels={'device_type': 'Device', 'failure_rate': 'Failure Rate (%)'},
        )
        fig_dev.update_traces(textposition='outside')
        apply_layout(fig_dev, 'Failure Rate by Device Type')
        fig_dev.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_dev, use_container_width=True)

    with col8:
        net_df = df.groupby('network_type').agg(
            total=('failure', 'count'), failed=('failure', 'sum')
        ).reset_index()
        net_df['failure_rate'] = round(net_df['failed'] / net_df['total'] * 100, 2)

        fig_net = px.bar(
            net_df.sort_values('failure_rate', ascending=False),
            x='network_type', y='failure_rate',
            color='failure_rate', color_continuous_scale='RdYlGn_r',
            text=[f'{r:.2f}%' for r in net_df.sort_values('failure_rate', ascending=False)['failure_rate']],
            labels={'network_type': 'Network', 'failure_rate': 'Failure Rate (%)'},
        )
        fig_net.update_traces(textposition='outside')
        apply_layout(fig_net, 'Failure Rate by Network Type')
        fig_net.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_net, use_container_width=True)

    # State heatmap
    fig_state = px.bar(
        loc_df.head(15), x='failure_rate', y='sender_state', orientation='h',
        color='failure_rate', color_continuous_scale='Reds',
        text=[f'{r:.2f}%' for r in loc_df.head(15)['failure_rate']],
        labels={'sender_state': 'State', 'failure_rate': 'Failure Rate (%)'},
    )
    fig_state.update_traces(textposition='outside')
    apply_layout(fig_state, 'Top 15 States by Failure Rate')
    fig_state.update_layout(height=440, coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — AMOUNT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>💰 Transaction Amount Analysis</h2>
        <p>Amount distribution: Success vs Failure</p>
    </div>
    """, unsafe_allow_html=True)

    success_amt = df[df['failure'] == 0]['amount_inr']
    failed_amt = df[df['failure'] == 1]['amount_inr']

    col9, col10 = st.columns(2)
    with col9:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=success_amt, name='Success', nbinsx=60,
            marker_color=COLOR_SUCCESS, opacity=0.65, histnorm='probability density'
        ))
        fig_hist.add_trace(go.Histogram(
            x=failed_amt, name='Failed', nbinsx=60,
            marker_color=COLOR_FAIL, opacity=0.65, histnorm='probability density'
        ))
        fig_hist.update_layout(barmode='overlay')
        apply_layout(fig_hist, 'Amount Distribution (Success vs Failed)')
        fig_hist.update_layout(height=360, xaxis_title='Amount (INR)', yaxis_title='Density')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col10:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=success_amt, name='Success',
            marker_color=COLOR_SUCCESS, boxpoints=False, notched=True
        ))
        fig_box.add_trace(go.Box(
            y=failed_amt, name='Failed',
            marker_color=COLOR_FAIL, boxpoints=False, notched=True
        ))
        apply_layout(fig_box, 'Amount Boxplot (Success vs Failed)')
        fig_box.update_layout(height=360, yaxis_title='Amount (INR)')
        st.plotly_chart(fig_box, use_container_width=True)

    # Stats table
    amt_stats = pd.DataFrame({
        'Metric': ['Count', 'Mean (₹)', 'Median (₹)', 'Std Dev (₹)', 'Q25 (₹)', 'Q75 (₹)', 'Max (₹)'],
        'Successful': [
            f"{len(success_amt):,}", f"₹{success_amt.mean():,.0f}", f"₹{success_amt.median():,.0f}",
            f"₹{success_amt.std():,.0f}", f"₹{success_amt.quantile(0.25):,.0f}",
            f"₹{success_amt.quantile(0.75):,.0f}", f"₹{success_amt.max():,.0f}"
        ],
        'Failed': [
            f"{len(failed_amt):,}", f"₹{failed_amt.mean():,.0f}", f"₹{failed_amt.median():,.0f}",
            f"₹{failed_amt.std():,.0f}", f"₹{failed_amt.quantile(0.25):,.0f}",
            f"₹{failed_amt.quantile(0.75):,.0f}", f"₹{failed_amt.max():,.0f}"
        ]
    })
    st.dataframe(amt_stats, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — ML MODEL
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>🤖 ML Model — Random Forest</h2>
        <p>Model evaluation metrics, confusion matrix, and feature importance</p>
    </div>
    """, unsafe_allow_html=True)

    if metrics:
        # Metric KPIs
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        ml_kpis = [
            (mc1, "Accuracy", f"{metrics.get('accuracy', 'N/A')}", "#0B3D2E"),
            (mc2, "Precision", f"{metrics.get('precision', 'N/A')}", "#F39C12"),
            (mc3, "Recall", f"{metrics.get('recall', 'N/A')}", "#90EE90"),
            (mc4, "F1-Score", f"{metrics.get('f1_score', 'N/A')}", "#0B3D2E"),
            (mc5, "ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}", "#F39C12"),
            (mc6, "PR-AUC", f"{metrics.get('pr_auc', 'N/A')}", "#90EE90"),
        ]
        for col, label, val, color in ml_kpis:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{color};">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col11, col12 = st.columns(2)
        with col11:
            # Confusion matrix
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            cm_arr = np.array(cm)
            fig_cm = go.Figure(go.Heatmap(
                z=cm_arr,
                x=['Predicted Success', 'Predicted Failed'],
                y=['Actual Success', 'Actual Failed'],
                text=[[f'TN={cm_arr[0,0]:,}', f'FP={cm_arr[0,1]:,}'],
                      [f'FN={cm_arr[1,0]:,}', f'TP={cm_arr[1,1]:,}']],
                texttemplate='%{text}<br>%{z:,}',
                textfont=dict(size=14, color='white'),
                colorscale=[[0, 'rgba(46,204,113,0.3)'], [0.5, 'rgba(100,160,255,0.4)'],
                            [1, 'rgba(231,76,60,0.7)']],
                showscale=False,
            ))
            apply_layout(fig_cm, 'Confusion Matrix')
            fig_cm.update_layout(height=360)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col12:
            # Feature importance
            if not fi_df.empty:
                top_fi = fi_df.head(15).sort_values('importance', ascending=True)
                fig_fi = go.Figure(go.Bar(
                    y=top_fi['feature'], x=top_fi['importance'],
                    orientation='h',
                    marker=dict(color=top_fi['importance'], colorscale='Blues', showscale=False),
                    text=[f'{v:.4f}' for v in top_fi['importance']],
                    textposition='outside',
                ))
                apply_layout(fig_fi, 'Top 15 Feature Importances')
                fig_fi.update_layout(height=360)
                st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("ℹ️ Run `python run_pipeline.py` from the project root to train the model and populate this section.")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — KEY INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>💡 Key Insights</h2>
        <p>Evidence-based findings derived from the actual dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if insights:
        for ins in insights:
            st.markdown(f"""
            <div class="insight-card">
                <span class="insight-number">{ins['id']}</span>
                <span class="insight-title">{ins['title']}</span>
                <div class="insight-text">{ins['text']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Compute live insights
        st.info("ℹ️ Run `python run_pipeline.py` to generate pre-computed insights, or see live analysis below.")
        bank_top = bank_df.sort_values('failure_rate', ascending=False).iloc[0] if len(bank_df) else None
        if bank_top is not None:
            st.markdown(f"""
            <div class="insight-card">
                <span class="insight-number">1</span>
                <span class="insight-title">Overall Failure Rate</span>
                <div class="insight-text">
                    The overall failure rate is <strong>{fr}%</strong> ({failed:,} out of {total:,} transactions).
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — ACTION PLAN
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Operational Action Plan</h2>
        <p>Evidence-linked recommendations for payment operations</p>
    </div>
    """, unsafe_allow_html=True)

    if action_plan:
        for item in action_plan:
            priority = item.get('priority', 'LOW').lower()
            badge_class = f"badge-{priority}"
            card_class = f"action-card {priority}"
            st.markdown(f"""
            <div class="{card_class}">
                <strong style="color:#ffffff">{item['category']}</strong>
                <span class="priority-badge {badge_class}">{item['priority']}</span>
                <p style="margin:8px 0 4px 0; color:rgba(255,255,255,0.85)">{item['recommendation']}</p>
                <p style="margin:0; font-size:0.78rem; color:rgba(255,255,255,0.5)">
                    📊 Evidence: {item['evidence']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Run `python run_pipeline.py` to generate the action plan.")

    # Footer
    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:rgba(255,255,255,0.35); font-size:0.78rem; padding-bottom:20px">
        DS_Day01_59 · Fintech — Where Are Digital Payments Failing? · Built with Streamlit & Plotly
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
