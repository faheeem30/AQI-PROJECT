import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

model = load_model("models/aqi_gru_model.h5", compile=False)
scaler = joblib.load("models/scaler.pkl")  # Load the pre-fitted scaler used during training

def predict_next(values):
    if len(values) < 24:
        return values[-1]

    values = np.array(values[-24:]).reshape(-1,1)

    # Use the pre-fitted scaler instead of fitting each time
    scaled = scaler.transform(values)

    scaled = scaled.reshape(1,24,1)

    pred_scaled = model.predict(scaled, verbose=0)

    pred = scaler.inverse_transform(pred_scaled)

    return float(pred[0][0])