import streamlit as st
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Crowd Density Advisor", layout="wide")

# Sidebar info / mode selector
st.sidebar.title("Crowd Density Advisor")
st.sidebar.markdown("""
This dashboard visualizes real-time crowd density detected from CCTV footage using CSRNet,
and makes future crowd count predictions using LSTM forecasting.

Instructions:

1. Run crowd counting pipeline on your extracted frames.
2. Build the time series CSV.
3. Train the LSTM forecasting model.
4. Start the FastAPI server.
5. Use this dashboard to view counts, heatmaps, and get short-term forecasts.

Choose dashboard mode below:
""")

dashboard_mode = st.sidebar.radio(
    "Dashboard Mode",
    ["Simple", "Enhanced"],
    help="Simple mode shows current count and forecast now. Enhanced mode allows date/time filtering."
)

# Location selector (for multiple videos)
video_locations = sorted([p.name for p in Path("models/preds").iterdir() if p.is_dir()])
if not video_locations:
    st.sidebar.warning("No video locations found in models/preds/. Please run crowd counting first.")
selected_location = st.sidebar.selectbox("Select Location", video_locations)

st.title(f"🏙️ Crowd Density Advisor: {selected_location}")

counts_json_path = Path(f"models/preds/{selected_location}/counts.json")
heatmaps_dir = Path(f"models/preds/{selected_location}/heatmaps")
timeseries_csv_path = Path(f"counts_ts/{selected_location}_counts.csv")

# ============================================================================
# Enhanced mode with date/time filters
# ============================================================================
if dashboard_mode == "Enhanced":
    st.sidebar.header("🔮 Prediction Settings")

    prediction_date = st.sidebar.date_input(
        "Select Date",
        value=datetime.now().date(),
        min_value=datetime.now().date(),
        max_value=datetime.now().date() + timedelta(days=7)
    )

    prediction_hour = st.sidebar.slider(
        "Select Hour (24-hour format)",
        min_value=0,
        max_value=23,
        value=datetime.now().hour,
        help="0 = midnight, 12 = noon, 23 = 11 PM"
    )

    day_name = prediction_date.strftime("%A")
    st.sidebar.info(f"📅 **{day_name}**")

    is_holiday = st.sidebar.checkbox("Mark as Holiday", value=False)

    prediction_datetime = datetime.combine(prediction_date, datetime.min.time()).replace(hour=prediction_hour)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Predicting for:**\n\n{prediction_datetime.strftime('%Y-%m-%d %H:%M')}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Current Count")
        if counts_json_path.exists():
            df_counts = pd.read_json(counts_json_path)
            st.metric("Latest Count", f"{df_counts['count'].iloc[-1]:.2f} people")

            heatmap_path = heatmaps_dir / f"{df_counts['frame_name'].iloc[-1].replace('.jpg', '_heatmap.jpg')}"
            if heatmap_path.exists():
                st.image(heatmap_path, caption="Latest Density Heatmap", use_container_width=True)
        else:
            st.info(f"Run counting for {selected_location} to generate counts.json.")

    with col2:
        st.subheader("🔮 Prediction for Selected Time")

        if st.button("🚀 Get Prediction", type="primary", use_container_width=True):
            api_url = "http://localhost:8000/forecast_demo"
            payload = {
                "when": prediction_datetime.isoformat(),
                "holiday": 1 if is_holiday else 0
            }

            try:
                with st.spinner("Getting prediction..."):
                    response = requests.post(api_url, json=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Prediction Complete!")

                    pred_count = result.get("predicted_count", 0)
                    advisory = result.get("advisory", "Unknown")

                    if advisory == "Low":
                        st.success(f"🟢 {advisory} Congestion")
                    elif advisory == "Medium":
                        st.warning(f"🟡 {advisory} Congestion")
                    else:
                        st.error(f"🔴 {advisory} Congestion")

                    st.metric("Predicted Crowd Count", f"{pred_count:.2f} people")

                    with st.expander("📋 View Full Response"):
                        st.json(result)

                    st.markdown("### 💡 Recommendation")
                    if advisory == "Low":
                        st.info("✅ Good time to travel! Low crowd expected.")
                    elif advisory == "Medium":
                        st.warning("⚠️ Moderate crowd expected, plan accordingly.")
                    else:
                        st.error("❌ High crowd expected. Consider alternatives.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the API server is running:\n\n`uvicorn app.api:app --reload --port 8000`")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# Simple mode (basic)
# ============================================================================
else:
    st.subheader("📊 Current Count")
    if counts_json_path.exists():
        df_counts = pd.read_json(counts_json_path)
        st.metric("Latest Count", f"{df_counts['count'].iloc[-1]:.2f} people")

        heatmap_path = heatmaps_dir / f"{df_counts['frame_name'].iloc[-1].replace('.jpg', '_heatmap.jpg')}"
        if heatmap_path.exists():
            st.image(str(heatmap_path), caption="Latest Density Heatmap")

    else:
        st.info(f"Run counting for {selected_location} to generate counts.json.")

    st.subheader("📈 Time Series")
    if timeseries_csv_path.exists():
        df_ts = pd.read_csv(timeseries_csv_path, parse_dates=["timestamp"])
        st.line_chart(df_ts.set_index("timestamp")["crowd_count"])
    else:
        st.info(f"Build the time series CSV for {selected_location} to see the chart.")

    st.subheader("🔮 Forecast")
    if st.button("Forecast Next Step"):
        api_url = "http://localhost:8000/forecast_demo"
        now_iso = datetime.now().isoformat(timespec="minutes")
        try:
            response = requests.post(api_url, json={"when": now_iso, "holiday": 0}, timeout=10)
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error(f"API Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Run: uvicorn app.api:app --reload --port 8000")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================================
# Common Section
# ============================================================================
st.markdown("---")
st.subheader("📈 Historical Time Series")

if timeseries_csv_path.exists():
    df_ts = pd.read_csv(timeseries_csv_path, parse_dates=["timestamp"])
    st.line_chart(df_ts.set_index("timestamp")["crowd_count"], use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Crowd", f"{df_ts['crowd_count'].mean():.1f}")
    col2.metric("Maximum Crowd", f"{df_ts['crowd_count'].max():.1f}")
    col3.metric("Minimum Crowd", f"{df_ts['crowd_count'].min():.1f}")
    col4.metric("Total Records", len(df_ts))
else:
    st.info(f"Build the time series CSV for {selected_location} to see the chart.")

st.markdown("---")
st.caption("🤖 AI-Based Crowd Density Prediction System | Powered by CSRNet + LSTM")
