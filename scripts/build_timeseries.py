import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def get_holidays(year, country='IN', cache={}):
    """Fetch/cached holidays from API."""
    if (year, country) in cache:
        return cache[(year, country)]
    try:
        import requests
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            holidays = response.json()
            holiday_dates = {h['date'] for h in holidays}
            cache[(year, country)] = holiday_dates
            return holiday_dates
        return set()
    except Exception as e:
        print(f"Warning: Holiday fetch failed for {year} ({country}): {e}")
        return set()


def generate_synthetic_crowd(start_date, real_mean, real_std, real_min, real_max, days=14, freq="15min"):
    """Generate synthetic crowd data with hospital-aware stats and trends."""
    timestamps = pd.date_range(start=start_date, periods=days*24*4, freq=freq)
    sdf = pd.DataFrame({'timestamp': timestamps})
    sdf['hour'] = sdf['timestamp'].dt.hour
    sdf['minute'] = sdf['timestamp'].dt.minute
    sdf['day_of_week'] = sdf['timestamp'].dt.dayofweek
    sdf['day_name'] = sdf['timestamp'].dt.day_name()
    sdf['is_weekend'] = sdf['day_of_week'].isin([5, 6]).astype(int)
    sdf['day_of_month'] = sdf['timestamp'].dt.day
    sdf['month'] = sdf['timestamp'].dt.month
    sdf['year'] = sdf['timestamp'].dt.year

    # Hospital area pattern
    is_weekday = ~sdf['is_weekend'].astype(bool)
    morning_peak = ((sdf['hour'] >= 7) & (sdf['hour'] <= 10))
    lunch_dip = ((sdf['hour'] >= 12) & (sdf['hour'] <= 13))
    evening_peak = ((sdf['hour'] >= 17) & (sdf['hour'] <= 20))
    night_low = ((sdf['hour'] < 6) | (sdf['hour'] > 21))

    base = real_mean * (0.6 + 0.18 * is_weekday) + sdf['is_weekend'] * real_mean * 0.18
    base += morning_peak * real_mean * (0.32 + 0.08 * is_weekday)
    base -= lunch_dip * real_mean * 0.12
    base += evening_peak * real_mean * (0.24 + 0.08 * sdf['is_weekend'])
    base -= night_low * real_mean * 0.32

    noise = np.random.normal(0, real_std * 0.27, size=len(sdf))

    crowd = base + noise
    sdf['crowd_count'] = np.clip(crowd, real_min, real_max).round(2)
    sdf['is_holiday'] = 0
    sdf['frame_name'] = None
    sdf['frame_index'] = None
    sdf['is_synthetic'] = 1
    return sdf


def build_timeseries(
    counts_json_path, output_csv_path,
    start_datetime=None, interval_sec=1.0,
    with_synthetic=False, syn_days=14
):
    # ----- REAL DATA -----
    with open(counts_json_path, 'r') as f:
        counts_data = json.load(f)
    print(f"📊 Loaded {len(counts_data)} count records")

    if start_datetime is None:
        start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = [start_datetime + timedelta(seconds=i * interval_sec) for i in range(len(counts_data))]
    records = []
    for idx, (entry, timestamp) in enumerate(zip(counts_data, timestamps)):
        records.append({
            'timestamp': timestamp,
            'crowd_count': entry['count'],
            'frame_name': entry['frame_name'],
            'frame_index': entry['frame_index'],
            'is_synthetic': 0
        })
    df = pd.DataFrame(records)

    # ---- TEMPORAL FEATURES ----
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour.astype(int)
    df['minute'] = df['timestamp'].dt.minute.astype(int)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype(int)
    df['day_name'] = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['day_of_month'] = df['timestamp'].dt.day.astype(int)
    df['month'] = df['timestamp'].dt.month.astype(int)
    df['year'] = df['timestamp'].dt.year.astype(int)

    # ---- Holidays ----
    years = df['year'].unique()
    all_holidays = set()
    for year in years:
        all_holidays.update(get_holidays(int(year), country='IN'))
    df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['is_holiday'] = df['date_str'].isin(all_holidays).astype(int)
    df.drop('date_str', axis=1, inplace=True)
    print(f"   Holidays found: {len(all_holidays)}")

    # ---- SYNTHETIC DATA GENERATION AND MERGE ----
    if with_synthetic:
        real_mean = df['crowd_count'].mean()
        real_std = df['crowd_count'].std()
        real_min = df['crowd_count'].min()
        real_max = df['crowd_count'].max()
        syn_start = df['timestamp'].max() + timedelta(minutes=15)
        synthetic_df = generate_synthetic_crowd(
            start_date=syn_start,
            real_mean=real_mean,
            real_std=real_std,
            real_min=real_min,
            real_max=real_max,
            days=syn_days,
            freq="15min"
        )
        full_df = pd.concat([df, synthetic_df], ignore_index=True)
        full_df = full_df.sort_values('timestamp').reset_index(drop=True)
        print(f"🧪 Added {len(synthetic_df)} synthetic records.")
    else:
        full_df = df

    # ---- SAVE CSV ----
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(f"✅ Saved time series to {output_path}")

    print(f"\n📊 Time Series Summary:")
    print(f"   Total records: {len(full_df)} (Real: {(full_df.is_synthetic==0).sum()}, Synthetic: {(full_df.is_synthetic==1).sum()})")
    print(f"   Time range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")
    print(f"   Average count: {full_df['crowd_count'].mean():.2f}")
    print(f"   Min count: {full_df['crowd_count'].min():.2f}")
    print(f"   Max count: {full_df['crowd_count'].max():.2f}")
    print(f"\n📋 Feature columns: {list(full_df.columns)}")
    return full_df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/build_timeseries.py <counts_json> <output_csv> [start_datetime] [interval_sec] [use_synthetic:0|1] [syn_days]")
        print("Example: python scripts/build_timeseries.py models/preds/street1/counts.json counts_ts/street1_counts.csv '2025-11-01 08:00:00' 45 1 21")
        sys.exit(1)

    counts_json = sys.argv[1]
    output_csv = sys.argv[2]
    start_dt = None
    interval_sec = 1.0
    use_synthetic = False
    syn_days = 14

    if len(sys.argv) > 3:
        start_dt = datetime.strptime(sys.argv[3], '%Y-%m-%d %H:%M:%S')
    if len(sys.argv) > 4:
        interval_sec = float(sys.argv[4])
    if len(sys.argv) > 5:
        use_synthetic = bool(int(sys.argv[5]))
    if len(sys.argv) > 6:
        syn_days = int(sys.argv[6])

    build_timeseries(counts_json, output_csv, start_dt, interval_sec, use_synthetic, syn_days)
