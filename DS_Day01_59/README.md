# DS_Day01_59 — FINTECH: WHERE ARE DIGITAL PAYMENTS FAILING?

> **A complete end-to-end Data Science project analyzing UPI transaction failure patterns, training a Random Forest classifier, and delivering an operational intelligence dashboard.**

---

## 1. Problem Statement

Digital payment platforms process millions of transactions daily. Even a small percentage of failures translates into significant revenue loss and customer dissatisfaction. This project analyzes 250,000 real UPI transactions to identify *where*, *when*, and *why* digital payments fail — and delivers a machine-learning model to predict failure before it happens.

---

## 2. Objective

Answer the following operational questions from transaction data:

1. What is the overall transaction failure rate?
2. Which banks have the highest failure rates?
3. Which payment methods (P2P vs P2M) fail more often?
4. At what hours/time-periods do failures spike?
5. Are there statistically unusual failure spikes?
6. Is transaction amount associated with failure?
7. Does device type influence failure rate?
8. Which operational factors most contribute to failures?
9. Can a Random Forest model predict transaction failure accurately?
10. What practical operational improvements should the company implement?

---

## 3. Dataset Description

| Property | Value |
|---|---|
| **File** | `upi_transactions_2024.csv` |
| **Records** | 250,000 |
| **Columns** | 17 |
| **Period** | January–December 2024 |
| **Missing values** | None |
| **Duplicates** | None |

### Column Schema

| Original Column | Cleaned Name | Type | Description |
|---|---|---|---|
| `transaction id` | `transaction_id` | String | Unique transaction ID |
| `timestamp` | `timestamp` | DateTime | Full datetime of transaction |
| `transaction type` | `transaction_type` | Categorical | P2P (person-to-person) or P2M (person-to-merchant) |
| `merchant_category` | `merchant_category` | Categorical | Category of merchant |
| `amount (INR)` | `amount_inr` | Integer | Transaction amount in Indian Rupees |
| `transaction_status` | `transaction_status` | Categorical | SUCCESS or FAILED |
| `sender_age_group` | `sender_age_group` | Categorical | Age group of sender |
| `receiver_age_group` | `receiver_age_group` | Categorical | Age group of receiver |
| `sender_state` | `sender_state` | Categorical | Indian state of sender |
| `sender_bank` | `sender_bank` | Categorical | Bank of sender (Axis, ICICI, HDFC, SBI, etc.) |
| `receiver_bank` | `receiver_bank` | Categorical | Bank of receiver |
| `device_type` | `device_type` | Categorical | Android, iOS, or Web |
| `network_type` | `network_type` | Categorical | 3G, 4G, 5G, WiFi |
| `fraud_flag` | `fraud_flag` | Binary | Fraud indicator (0/1) |
| `hour_of_day` | `hour` | Integer | Hour of transaction (0–23) |
| `day_of_week` | `day_of_week` | Categorical | Day name |
| `is_weekend` | `is_weekend` | Binary | 1 if weekend |

### Column Mapping Notes

- `transaction id` (space) → `transaction_id`
- `amount (INR)` → `amount_inr`
- `transaction type` (space) → `transaction_type`
- No `Payment_Method`, `Bank`, `Device`, `Location` columns by these names; dataset uses `transaction_type`, `sender_bank`, `device_type`, `sender_state` respectively.

---

## 4. Data Cleaning Process

| Step | Action |
|---|---|
| Column names | Normalised (lowercase, underscores, spaces removed) |
| Missing values | **None found** — no imputation needed |
| Duplicate rows | **None found** |
| Duplicate Transaction IDs | Verified unique |
| Transaction Status | Normalised to `Success`/`Failed`; binary target `failure` created |
| Timestamp | Parsed to datetime; derived `hour`, `time_period`, `day_of_week`, `month`, `transaction_date` |
| Amount | Checked for negatives/zeros — none found; no outliers removed |
| Time periods | Early Morning (00–05), Morning (06–11), Afternoon (12–17), Evening (18–23) |

---

## 5. Features Used in ML Model

### Categorical Features (OneHotEncoded)
- `transaction_type` — P2P or P2M
- `merchant_category` — type of merchant
- `sender_bank` — sending bank
- `receiver_bank` — receiving bank
- `device_type` — Android / iOS / Web
- `network_type` — 3G / 4G / 5G / WiFi
- `sender_state` — geographic state
- `sender_age_group` — age demographics
- `receiver_age_group` — age demographics
- `time_period` — Early Morning / Morning / Afternoon / Evening
- `day_of_week` — day of transaction
- `amount_bin` — binned amount category

### Numerical Features (StandardScaled)
- `amount_inr` — raw transaction amount
- `amount_log` — log-transformed amount (log1p)
- `hour` — hour of day
- `is_weekend` — binary flag
- `is_5g` — binary flag for 5G network

### Excluded from Model
- `transaction_id` (identifier — never used as feature)
- `transaction_status` (target leakage)
- `fraud_flag` (correlated artifact — kept separate)

---

## 6. EDA Summary

Seven visualizations generated:

1. **Overall Transaction Outcome** — Success vs Failure donut + bar
2. **Failure Rate by Bank** — Horizontal bar with average reference line
3. **Failure Rate by Payment Method** — Bar comparison + volume
4. **Failure Rate by Hour** — Dual-axis line + volume
5. **Amount Distribution** — Overlapping histograms + boxplot
6. **Device Failure Rate** — Bar chart per device type
7. **Spike Detection** — Daily failures + 7-day rolling mean + spike markers

---

## 7. Machine Learning Approach

**Algorithm:** Random Forest Classifier

**Rationale:** Random Forest handles mixed categorical/numerical features naturally via one-hot encoding, is robust to overfitting with many estimators, produces interpretable feature importances, and performs well on imbalanced datasets with `class_weight='balanced'`.

### Pipeline Architecture

```
Input Data
    ↓
ColumnTransformer
  ├── StandardScaler (numerical)
  └── OneHotEncoder (categorical)
    ↓
RandomForestClassifier
    ↓
Predictions + Probabilities
```

---

## 8. Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
)
```

- `n_estimators=300` — sufficient for stable feature importance estimates
- `class_weight='balanced'` — adjusts for class imbalance without oversampling
- `random_state=42` — reproducible results
- `n_jobs=-1` — uses all CPU cores for speed

### Train/Test Split

```python
train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

- 80% training (200,000 rows)
- 20% testing (50,000 rows)
- Stratified to preserve class proportions

---

## 9. Evaluation Metrics

All metrics computed on the **held-out test set only** (never training data):

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Of predicted failures, how many were real |
| Recall | Of actual failures, how many were caught |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Discrimination ability across all thresholds |
| PR-AUC | Precision-Recall trade-off (better for imbalanced classes) |
| Confusion Matrix | TN, FP, FN, TP breakdown |

> **Note:** For imbalanced transaction data, Recall and PR-AUC are given priority over Accuracy.

---

## 10. Key Findings

*(Numbers populated from actual model run — see `outputs/reports/insights.json`)*

Key findings cover:
- Overall failure rate and volume
- Bank with highest observed failure rate
- Payment type (P2P vs P2M) failure comparison
- Peak failure time period
- Statistical spike detection results
- Amount distribution differences between success and failure
- Device-specific failure patterns

---

## 11. Operational Recommendations

| Category | Priority | Finding |
|---|---|---|
| Bank Operations | HIGH | Initiate technical review with highest-failure bank |
| Payment Routing | MEDIUM | Implement intelligent retry for high-failure transaction types |
| Peak-Time Capacity | HIGH | Pre-provision infrastructure for peak failure periods |
| Device Compatibility | MEDIUM | Review SDK/UX for highest-failure device type |
| Real-time Monitoring | HIGH | Deploy automated threshold alerts |
| Data Quality | LOW | Separate fraud-flagged from operational failures in reporting |

---

## 12. Dashboard

### Launch

```bash
cd e:\Fintech\DS_Day01_59
streamlit run dashboard/app.py
```

### Features

- 🎛️ **Sidebar Filters**: Bank, Transaction Type, Device, State, Time Period, Status
- 📊 **Section 1 — Executive Overview**: 6 KPI cards, daily volume chart, outcome donut
- 🏦 **Section 2 — Bank Performance**: Failure rate bar, stacked success/fail bar, summary table
- 💳 **Section 3 — Payment Method**: Failure rate + volume by type and merchant category
- ⏰ **Section 4 — Time Analysis**: Hourly dual-axis, time-period breakdown, spike detection, day×hour heatmap
- 📱 **Section 5 — Device & Location**: Device/network/state failure rates
- 💰 **Section 6 — Amount Analysis**: Distribution histogram + boxplot + stats table
- 🤖 **Section 7 — ML Model**: 6 metric KPIs, confusion matrix, feature importance
- 💡 **Section 8 — Key Insights**: 7 evidence-based insights
- 🎯 **Section 9 — Action Plan**: Priority-coded operational recommendations

---

## 13. Installation

```bash
# Clone or navigate to the project directory
cd e:\Fintech\DS_Day01_59

# Install dependencies
pip install -r requirements.txt
```

---

## 14. How to Run

```bash
# Step 1: Run the full pipeline (cleaning + EDA + ML)
cd e:\Fintech\DS_Day01_59
python run_pipeline.py

# Step 2: Launch the dashboard
streamlit run dashboard/app.py
```

All outputs are saved to:
- `data/processed/` — cleaned dataset
- `outputs/figures/` — all visualizations (PNG)
- `outputs/metrics/` — model metrics (JSON) + feature importance (CSV)
- `outputs/reports/` — EDA tables, insights, action plan (JSON/CSV)
- `models/` — trained pipeline (PKL)

---

## 15. Project Structure

```
DS_Day01_59/
│
├── data/
│   ├── raw/
│   │   └── upi_transactions_2024.csv        ← Original dataset (preserved)
│   └── processed/
│       ├── cleaned_transactions.csv          ← Cleaned dataset
│       └── features.csv                      ← Feature-engineered dataset
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py                      ← Phase 2-3: Cleaning & Validation
│   ├── eda.py                                ← Phase 4-5: EDA & Visualizations
│   ├── feature_engineering.py               ← Phase 6: Feature Engineering
│   └── train_model.py                        ← Phase 7-10: ML Training & Evaluation
│
├── models/
│   └── random_forest.pkl                     ← Trained Pipeline
│
├── outputs/
│   ├── figures/                              ← 9+ PNG visualizations
│   ├── metrics/
│   │   ├── model_metrics.json
│   │   ├── feature_importance.csv
│   │   └── classification_report.txt
│   └── reports/
│       ├── data_quality_report.json
│       ├── bank_analysis.csv
│       ├── payment_method_analysis.csv
│       ├── device_analysis.csv
│       ├── location_analysis.csv
│       ├── hourly_analysis.csv
│       ├── time_period_analysis.csv
│       ├── daily_spike_analysis.csv
│       ├── amount_stats.json
│       ├── insights.json
│       └── action_plan.json
│
├── dashboard/
│   └── app.py                                ← Streamlit Dashboard
│
├── run_pipeline.py                           ← Master pipeline runner
├── requirements.txt
└── README.md
```

---

## 16. Limitations

1. **No causal inference**: All associations between banks/devices/times and failure rates are observational. Root causes require separate investigation (server logs, API traces).
2. **Synthetic data characteristics**: The dataset is labeled with `fraud_flag` and pre-computed `hour_of_day`/`is_weekend` — in production, these would be derived in real-time.
3. **No real-time stream**: The pipeline is batch-based; production deployment requires a streaming pipeline.
4. **Random Forest interpretability**: Feature importance shows *association*, not causation. SHAP values would provide deeper instance-level explanations.
5. **No hyperparameter tuning**: GridSearchCV/Optuna tuning was intentionally avoided to maintain interpretability and reproducibility.

---

## 17. Future Improvements

- [ ] SHAP value analysis for instance-level explainability
- [ ] XGBoost/LightGBM comparison models
- [ ] Time-series forecasting for failure rate prediction
- [ ] SMOTE oversampling comparison with `class_weight='balanced'`
- [ ] Real-time streaming pipeline (Kafka + Faust)
- [ ] Model drift monitoring (Evidently AI)
- [ ] Hyperparameter tuning with Optuna
- [ ] Interactive SHAP force plots in the dashboard
- [ ] State-level choropleth map for geographic failure visualization
- [ ] Receiver bank analysis (not just sender bank)
