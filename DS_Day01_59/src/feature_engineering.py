"""
feature_engineering.py
Phase 6 — Feature Engineering for ML
DS_Day01_59: Fintech — Where Are Digital Payments Failing?
"""

import pandas as pd
import numpy as np
import os

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_transactions.csv')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features.csv')


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML features from the cleaned dataset.
    Rules:
      - DO NOT use Transaction_ID as a feature
      - DO NOT use Transaction_Status as a feature
      - DO NOT leak the target (failure)
    """
    df = df.copy()

    # ── Amount features ─────────────────────────────────────────────────────
    # Log transform to handle skewness; +1 to handle potential zeros
    df['amount_log'] = np.log1p(df['amount_inr'])

    # Amount bins
    df['amount_bin'] = pd.cut(
        df['amount_inr'],
        bins=[0, 500, 2000, 10000, 50000, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    # ── Time features ────────────────────────────────────────────────────────
    # hour and time_period already created in cleaning
    # is_weekend already in dataset; override with proper bool
    df['is_weekend'] = df['is_weekend'].astype(int)

    # Network type binary flags
    df['is_5g'] = (df['network_type'] == '5G').astype(int)

    # ── Feature list for ML ──────────────────────────────────────────────────
    categorical_features = [
        'transaction_type',    # P2P, P2M
        'merchant_category',   # Entertainment, Grocery, etc.
        'sender_bank',         # Which bank sent
        'receiver_bank',       # Which bank received
        'device_type',         # Android, iOS, Web
        'network_type',        # 3G, 4G, 5G, WiFi
        'sender_state',        # State
        'sender_age_group',    # age group
        'receiver_age_group',  # age group
        'time_period',         # Early Morning / Morning / Afternoon / Evening
        'day_of_week',         # Day of week
        'amount_bin',          # Amount category
    ]

    numerical_features = [
        'amount_inr',
        'amount_log',
        'hour',
        'is_weekend',
        'is_5g',
    ]

    target = 'failure'

    # Verify all features exist
    missing = [f for f in categorical_features + numerical_features if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing}")
        categorical_features = [f for f in categorical_features if f in df.columns]
        numerical_features = [f for f in numerical_features if f in df.columns]

    feature_cols = categorical_features + numerical_features + [target]
    df_features = df[feature_cols].copy()

    # Convert amount_bin to string to avoid Categorical issues with OneHotEncoder
    df_features['amount_bin'] = df_features['amount_bin'].astype(str)

    print(f"[FEAT] Feature matrix shape: {df_features.shape}")
    print(f"[FEAT] Categorical features: {categorical_features}")
    print(f"[FEAT] Numerical features: {numerical_features}")
    print(f"[FEAT] Target: {target}")
    print(f"[FEAT] Class balance:\n{df_features[target].value_counts()}")

    return df_features, categorical_features, numerical_features


if __name__ == '__main__':
    df = pd.read_csv(PROCESSED_PATH)
    df_features, cat_cols, num_cols = engineer_features(df)
    df_features.to_csv(FEATURES_PATH, index=False)
    print(f"[FEAT] Saved features to: {FEATURES_PATH}")
