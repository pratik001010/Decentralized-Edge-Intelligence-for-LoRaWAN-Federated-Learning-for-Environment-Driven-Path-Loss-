import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("2.aggregated_measurements_data.csv", nrows=50000)
df["pressure"] = df["pressure"] * 3.125

print("exp_pl unique count:", df["exp_pl"].nunique())
print("exp_pl unique values:", sorted(df["exp_pl"].dropna().unique()))
print()
print("Columns:", list(df.columns))
print()

targets = ["pressure","co2","temperature","humidity","pm25",
           "distance","c_walls","w_walls","rssi","snr","SF","toa"]
for c in targets:
    if c in df.columns:
        try:
            r = round(float(df[c].corr(df["exp_pl"])), 4)
            print(f"Corr {c:15s} vs exp_pl: {r}")
        except Exception as e:
            print(f"Corr {c:15s} vs exp_pl: ERROR {e}")

print()
print("Per-device exp_pl mean:")
print(df.groupby("device_id")["exp_pl"].mean())
