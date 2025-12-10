from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "wetter.csv"  

print("Attempting to load:", csv_path)
if not csv_path.exists():
    raise SystemExit(f"ERROR: file not found at {csv_path}")

df = pd.read_csv(csv_path)
print("\nColumns in CSV:", list(df.columns))
print("\nFirst 5 rows (raw):")
print(df.head().to_string(index=False))

possible_date_cols = ["DATE", "Date", "date", "Datum", "datum"]
date_col = next((c for c in df.columns if c in possible_date_cols), None)

if date_col is None:
    lowered = {c.lower(): c for c in df.columns}
    for cand in possible_date_cols:
        if cand.lower() in lowered:
            date_col = lowered[cand.lower()]
            break

if date_col is None:
    
    for c in df.columns:
        sample = df[c].astype(str).dropna().iloc[:10] if len(df[c].dropna())>0 else []
        try:
            pd.to_datetime(sample, errors="raise")
            date_col = c
            break
        except Exception:
            continue

if date_col is None:
    raise SystemExit("ERROR: Could not detect a date column. Columns were: " + ", ".join(df.columns))
print(f"\nUsing date column: '{date_col}'")

possible_temp_cols = ["TAVG", "Tavg", "temp", "Temp", "TEMP", "Temperatur", "temperatur", "Temperature", "temperature"]
temp_col = next((c for c in df.columns if c in possible_temp_cols), None)

if temp_col is None:
    lowered = {c.lower(): c for c in df.columns}
    for cand in possible_temp_cols:
        if cand.lower() in lowered:
            temp_col = lowered[cand.lower()]
            break

if temp_col is None:
    
    for c in df.columns:
        if "temp" in c.lower() or "temper" in c.lower():
            temp_col = c
            break

if temp_col is None:
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        temp_col = numeric_cols[0]

if temp_col is None:
    raise SystemExit("ERROR: Could not detect a temperature column. Columns were: " + ", ".join(df.columns))
print(f"Using temperature column: '{temp_col}'")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
if df[date_col].isna().all():
    raise SystemExit(f"ERROR: Failed to parse any dates from column '{date_col}'.")

if df[temp_col].dtype == object:
    df[temp_col] = df[temp_col].astype(str).str.replace(",", ".", regex=False)
df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")

df = df.dropna(subset=[date_col, temp_col])
print("\nAfter cleaning: rows =", len(df))
print(df[[date_col, temp_col]].head().to_string(index=False))

overall_avg = df[temp_col].mean()
print(f"\nOverall average temperature ({temp_col}): {overall_avg:.3f}")

july_mask = df[date_col].dt.month == 7
july_df = df.loc[july_mask, temp_col]
if len(july_df) == 0:
    print("WARNING: No July records found.")
else:
    july_avg = july_df.mean()
    print(f"Average temperature in July: {july_avg:.3f} (n={len(july_df)})")

may_mask = df[date_col].dt.month == 5
may_df = df.loc[may_mask, temp_col]
if len(may_df) == 0:
    print("WARNING: No May records found.")
else:
    may_avg = may_df.mean()
    print(f"Average temperature in May: {may_avg:.3f} (n={len(may_df)})")

if len(july_df) >= 2 and len(may_df) >= 2:
    t_stat, p_val = ttest_ind(july_df, may_df, equal_var=False, nan_policy="omit")
    print("\nT-test July vs May (Welch):")
    print(f" t = {t_stat:.4f}, p = {p_val:.4g}")
    if p_val < 0.05:
        print(" → The difference in mean temperatures between July and May is statistically significant (p < 0.05).")
    else:
        print(" → No statistically significant difference found (p >= 0.05).")
else:
    print("\nNot enough data to run t-test (need >=2 observations in each month).")
