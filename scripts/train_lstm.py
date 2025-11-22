"""
Train LSTM model for crowd forecasting with robust, week-aware time series split and GPU support.
Usage:
  python scripts/train_lstm.py counts_ts/street1_counts.csv models/lstm_street1.pth [lookback] [epochs]
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import random

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, 0])  # Predict crowd_count (first column)
    return np.array(X), np.array(y)

def smart_time_split(df, lookback):
    """Splits at latest complete week for a strict temporal split."""
    df = df.sort_values('timestamp')
    full_weeks = df['timestamp'].dt.isocalendar().week.unique()
    if len(full_weeks) < 2:
        return df, pd.DataFrame()  # all train if only one week
    test_week = full_weeks[-1]
    train_df = df[df['timestamp'].dt.isocalendar().week != test_week]
    test_df = df[df['timestamp'].dt.isocalendar().week == test_week]
    # Avoid leaking past to future, also for user-expected report focus
    return train_df, test_df

def train_model(csv_path, model_save_path, lookback=10, epochs=50, hidden_dim=64, batch_size=64):
    print("=" * 60)
    print("LSTM Crowd Forecasting - Training (with week-aware split & GPU support)")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}\n")

    # Load data
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    print(f"📊 Loaded {len(df)} records from {csv_path}")

    feature_cols = ['crowd_count', 'hour', 'day_of_week', 'is_weekend', 'is_holiday']
    data = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, lookback=lookback)
    print(f"🔢 Created {len(X)} sequences (lookback={lookback})")
    if len(X) < 5:
        print("⚠️  WARNING: Very few sequences for training! For real deployment, you need hours/days/weeks of footage.")

    # week-aware time series split
    full_seq_df = df.iloc[lookback:].reset_index(drop=True)
    train_df, test_df = smart_time_split(full_seq_df, lookback)
    train_idx = train_df.index.values
    test_idx = test_df.index.values

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    print(f"   Train: {len(X_train)} sequences")
    print(f"   Test: {len(X_test)} sequences (last full available week)")

    # Model, loss, optimizer
    input_dim = X_train.shape[2]
    model = LSTMForecaster(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n🚀 Training for {epochs} epochs...")
    model.train()
    n_batches = int(np.ceil(len(X_train) / batch_size))
    for epoch in range(epochs):
        total_loss = 0.0
        perm = torch.randperm(len(X_train))
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            batch_X = X_train[idx]
            batch_y = y_train[idx]
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / n_batches

        # Evaluate test loss
        if len(X_test)>0 and (epoch+1)%10==0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test).squeeze()
                test_loss = criterion(test_outputs, y_test).item()
            model.train()
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")
        elif (epoch+1)%10==0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).squeeze().cpu().numpy()
        test_pred = model(X_test).squeeze().cpu().numpy() if len(X_test)>0 else []
    train_rmse = np.sqrt(np.mean((train_pred - y_train.cpu().numpy()) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - y_test.cpu().numpy()) ** 2)) if len(X_test)>0 else None
    print("\n📊 Final Results:")
    print(f"   Train RMSE: {train_rmse:.4f}") 
    if test_rmse is not None:
        print(f"   Test RMSE:  {test_rmse:.4f}")

    # Save model and scaler
    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'lookback': lookback,
        'feature_cols': feature_cols
    }, model_path)

    scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print(f"\n✅ Saved model to {model_path}")
    print(f"✅ Saved scaler to {scaler_path}")

    # Optional: Plot results if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(train_df['timestamp'], y_train.cpu().numpy(), label='Train True')
        plt.plot(train_df['timestamp'], train_pred, label='Train Pred', alpha=0.7)
        if len(X_test)>0:
            plt.plot(test_df['timestamp'], y_test.cpu().numpy(), label='Test True')
            plt.plot(test_df['timestamp'], test_pred, label='Test Pred', alpha=0.7)
        plt.legend(); plt.title("Crowd Forecasting LSTM"); plt.tight_layout(); plt.show()
    except ImportError:
        pass
    return model, scaler

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/train_lstm.py <csv_path> <model_save_path> [lookback] [epochs]")
        print("Example: python scripts/train_lstm.py counts_ts/street1_counts.csv models/lstm_street1.pth")
        sys.exit(1)

    csv_path = sys.argv[1]
    model_save_path = sys.argv[2]
    lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    train_model(csv_path, model_save_path, lookback=lookback, epochs=epochs)
