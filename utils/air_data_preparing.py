import numpy as np
import pandas as pd
from pathlib import Path

print("Preparing Air Quality dataset from local raw CSVs...")

# =========================
# paths
# =========================
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "../dataset/raw"
OUT_DIR = BASE_DIR / "../dataset/Airquality"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# find station files
# =========================
csv_files = sorted(RAW_DIR.glob("PRSA_Data_*_20130301-20170228.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No station CSV found in {RAW_DIR}. "
        "Please put files like PRSA_Data_Aotizhongxin_20130301-20170228.csv into Model/Data/raw/."
    )

print(f"Found {len(csv_files)} station CSV files in {RAW_DIR}")

# =========================
# read + align time index
# =========================
dfs = []
for f in csv_files:
    df = pd.read_csv(f)

    # build datetime index
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("time").sort_index()

    # keep only PM2.5
    df = df[["PM2.5"]]
    dfs.append(df)

# merge stations into one table
data = pd.concat(dfs, axis=1)

# rename columns station_0..station_{N-1}
data.columns = [f"station_{i}" for i in range(len(dfs))]

print("Original data shape:", data.shape)

# =========================
# cleaning missing values
# =========================
data = data.interpolate()
data = data.bfill().ffill()

missing = int(data.isna().sum().sum())
print("Missing value after cleaning:", missing)
if missing != 0:
    raise ValueError("Still has missing values after cleaning. Please check raw files.")

# =========================
# build X (T, N, 1)
# =========================
X = data.values.astype(np.float32)  # (T, N)
X = X[..., np.newaxis]              # (T, N, 1)
print("Final input shape:", X.shape)

# =========================
# build adjacency A (N, N) by correlation
# =========================
corr = np.corrcoef(data.values.T)
corr = np.nan_to_num(corr, nan=0.0)

corr[corr < 0] = 0.0
maxv = corr.max()
A = corr / maxv if maxv > 0 else corr

print("Adjacency matrix shape:", A.shape)

# =========================
# normalize (global z-score)
# =========================
mean = float(X.mean())
std = float(X.std() + 1e-8)
X_norm = (X - mean) / std

# =========================
# save npy
# =========================
np.save(OUT_DIR / "air_quality_data.npy", X)
np.save(OUT_DIR / "air_quality_adj.npy", A.astype(np.float32))
np.save(OUT_DIR / "air_quality_data_norm.npy", X_norm.astype(np.float32))
np.save(OUT_DIR / "air_quality_norm_stats.npy", np.array([mean, std], dtype=np.float32))

print("\nSaved files to:", OUT_DIR)
print(" - air_quality_data.npy")
print(" - air_quality_adj.npy")
print(" - air_quality_data_norm.npy")
print(" - air_quality_norm_stats.npy")
print("\nDone.")