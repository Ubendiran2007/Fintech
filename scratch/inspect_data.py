import pandas as pd

df = pd.read_csv('e:/Fintech/upi_transactions_2024.csv')

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== LAST 5 ROWS ===")
print(df.tail())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATES ===")
print(df.duplicated().sum())

print("\n=== transaction_status unique ===")
print(df['transaction_status'].unique())

print("\n=== device_type unique ===")
print(df['device_type'].unique())

print("\n=== sender_bank unique ===")
print(df['sender_bank'].unique())

print("\n=== receiver_bank unique ===")
print(df['receiver_bank'].unique())

print("\n=== transaction_type unique ===")
print(df['transaction_type'].unique())

print("\n=== sender_state unique ===")
print(df['sender_state'].unique())

print("\n=== merchant_category unique ===")
print(df['merchant_category'].unique())

print("\n=== network_type unique ===")
print(df['network_type'].unique())

print("\n=== DESCRIBE ===")
print(df.describe(include='all'))

print("\n=== Amount (INR) stats ===")
print(df['amount (INR)'].describe())
print("Negative amounts:", (df['amount (INR)'] < 0).sum())
print("Zero amounts:", (df['amount (INR)'] == 0).sum())
