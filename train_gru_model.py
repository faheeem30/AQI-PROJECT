import os
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib

from utils.dataset_utils import create_sequences


BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data")

all_series = []

print("Scanning dataset...\n")

# ==============================
# LOAD DATA
# ==============================

for root, dirs, files in os.walk(DATA_PATH):

    for file in files:

        if file.endswith(".csv"):

            path = os.path.join(root, file)

            try:

                df = pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    encoding="latin1"
                )

                df = df.replace(["NA", "--", "None"], np.nan)

                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                numeric = df.select_dtypes(include=[np.number])

                pm25_col = None

                for col in numeric.columns:

                    if "PM2.5" in col or "PM25" in col:

                        pm25_col = col
                        break

                if pm25_col:

                    series = numeric[pm25_col].dropna().values

                    if len(series) > 50:

                        all_series.append(series)

                        print("Loaded:", path)

            except:
                print("Skipped:", path)

# ==============================
# CHECK DATA
# ==============================

if len(all_series) == 0:

    print("❌ No usable data")
    exit()

data = np.concatenate(all_series)

print("\nTotal samples:", len(data))


# ==============================
# SCALE DATA
# ==============================

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data.reshape(-1,1))

os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")


# ==============================
# CREATE SEQUENCES
# ==============================

X, y = create_sequences(data_scaled, 24)

print("Training samples:", X.shape)


# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)


# ==============================
# OPTIMIZED GRU MODEL
# ==============================

model = Sequential()

model.add(GRU(
    32,
    return_sequences=True,
    input_shape=(24,1)
))

model.add(Dropout(0.2))

model.add(GRU(16))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)


early_stop = EarlyStopping(
    patience=3,
    restore_best_weights=True
)


# ==============================
# TRAIN MODEL
# ==============================

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,   # bigger batch = faster
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)


# ==============================
# SAVE MODEL
# ==============================

model.save("models/aqi_gru_model.h5")

print("\n✅ Model saved → models/aqi_gru_model.h5")


# ==============================
# METRICS
# ==============================

y_pred = model.predict(X_val)

y_true = y_val.flatten()
y_pred = y_pred.flatten()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

metrics = {
    "PM2.5": {
        "RMSE": float(rmse),
        "MSE": float(mse),
        "MAE": float(mae),
        "R2": float(r2)
    }
}

os.makedirs("metrics", exist_ok=True)

with open("metrics/gru_model.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved → metrics/gru_model.json")