from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Crowd Advisor API")

# Optional: Enable CORS if frontend served elsewhere; adjust origins accordingly
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

MODEL_PATH = Path("models/lstm_street1.pth")
SCALER_PATH = Path("models/lstm_street1_scaler.pkl")
LOOKBACK_DEFAULT = 10
FEATURE_COLS = ['crowd_count', 'hour', 'day_of_week', 'is_weekend', 'is_holiday']

class LSTMForecaster(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])

# Singleton model manager with lazy loading and device awareness
class ModelManager:
    model = None
    scaler = None
    lookback = LOOKBACK_DEFAULT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def load(cls):
        if cls.model is None or cls.scaler is None:
            if not MODEL_PATH.exists() or not SCALER_PATH.exists():
                return False
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            cls.lookback = checkpoint.get('lookback', LOOKBACK_DEFAULT)
            cls.model = LSTMForecaster(input_dim=len(FEATURE_COLS), hidden_dim=checkpoint.get('hidden_dim', 64))
            cls.model.load_state_dict(checkpoint['model_state_dict'])
            cls.model.to(cls.device)
            cls.model.eval()
            cls.scaler = joblib.load(SCALER_PATH)
        return True

ModelManager.load()

class ForecastQuery(BaseModel):
    when: str
    holiday: Optional[int] = 0

class ForecastResponse(BaseModel):
    when: str
    predicted_count: float
    advisory: str
    method: str

@app.get("/health")
def health():
    model_loaded = ModelManager.model is not None and ModelManager.scaler is not None
    return {"ok": True, "model_loaded": model_loaded, "device": str(ModelManager.device)}

@app.get("/now")
def now_count():
    preds = Path("models/preds/street1/counts.json")
    if not preds.exists():
        raise HTTPException(status_code=404, detail="counts.json not found. Run counting first.")
    df = pd.read_json(preds)
    last = df.iloc[-1]
    return {"frame": last["frame_name"], "count": last["count"], "index": int(last["frame_index"])}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(q: ForecastQuery):
    if not ModelManager.load():
        raise HTTPException(status_code=404, detail="Forecast model/scaler not available. Train LSTM first.")

    ts_csv = Path("counts_ts/street1_counts.csv")
    if not ts_csv.exists():
        raise HTTPException(status_code=404, detail="Time series CSV not found. Build it first.")

    df = pd.read_csv(ts_csv, parse_dates=['timestamp']).sort_values('timestamp')

    # Ensure features exist with fallback defaults
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            if feat == 'is_weekend' and 'day_of_week' in df:
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            elif feat == 'is_holiday':
                df['is_holiday'] = 0
            else:
                df[feat] = 0

    X = df[FEATURE_COLS].astype(np.float32).values

    if len(X) < ModelManager.lookback:
        raise HTTPException(status_code=400, detail=f"Not enough history: need {ModelManager.lookback}, have {len(X)}")

    x_seq = X[-ModelManager.lookback:]
    x_seq_scaled = ModelManager.scaler.transform(x_seq)
    x_tensor = torch.tensor(x_seq_scaled, dtype=torch.float32).unsqueeze(0).to(ModelManager.device)

    with torch.no_grad():
        yhat = ModelManager.model(x_tensor).item()

    yhat = max(0.0, yhat)
    yhat_display = round(yhat, 2)

    q25, q75 = np.quantile(df['crowd_count'], [0.25, 0.75])
    if yhat >= q75:
        level = "High"
    elif yhat >= q25:
        level = "Medium"
    else:
        level = "Low"

    return ForecastResponse(
        when=q.when,
        predicted_count=yhat_display,
        advisory=level,
        method="LSTM Neural Network"
    )

@app.post("/forecast_demo", response_model=ForecastResponse)
def forecast_demo(q: ForecastQuery):
    ts_csv = Path("counts_ts/street1_counts.csv")
    if not ts_csv.exists():
        raise HTTPException(status_code=404, detail="Time series CSV not found.")

    df = pd.read_csv(ts_csv, parse_dates=['timestamp'])
    req_time = datetime.fromisoformat(q.when)
    req_hour = req_time.hour
    req_dow = req_time.weekday()

    # Filter on BOTH hour AND day_of_week for relevance
    similar = df[(df['hour'] == req_hour) & (df['day_of_week'] == req_dow)]

    if len(similar) > 0:
        predicted_count = similar['crowd_count'].mean()
    else:
        predicted_count = df['crowd_count'].mean()

    predicted_count = max(0, round(predicted_count, 2))
    q25, q75 = df['crowd_count'].quantile([0.25, 0.75])
    if predicted_count >= q75:
        level = "High"
    elif predicted_count >= q25:
        level = "Medium"
    else:
        level = "Low"

    return ForecastResponse(
        when=q.when,
        predicted_count=predicted_count,
        advisory=level,
        method="Historical Average (Demo Mode)"
    )
