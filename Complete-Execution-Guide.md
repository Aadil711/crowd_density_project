# 🚀 Complete Step-by-Step Execution Guide
## AI-Based Crowd Density Prediction Project

**For First-Time Users - No Prior Knowledge Required**

---

## 📋 Table of Contents

1. [Prerequisites Check](#prerequisites-check)
2. [Method A: Automated Pipeline (3 Minutes)](#method-a-automated-pipeline)
3. [Method B: Manual Step-by-Step (10 Minutes)](#method-b-manual-step-by-step)
4. [Understanding Your Results](#understanding-your-results)
5. [Using Your Own Video](#using-your-own-video)
6. [Troubleshooting](#troubleshooting)
7. [Cleaning Up Storage](#cleaning-up-storage)

---

## Prerequisites Check

### 1. Verify Python Installation

Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux) and type:

```bash
python --version
```

**Expected:** `Python 3.8.x`, `3.9.x`, `3.10.x`, or `3.11.x`

❌ **If not installed:**
1. Download from [python.org](https://www.python.org/downloads/)
2. During installation, **CHECK "Add Python to PATH"**
3. Restart computer
4. Try `python --version` again

---

### 2. Download the Project

**Option A: Direct Download**
1. Download the project ZIP file
2. Extract to a simple path:
   - Windows: `C:\crowd-project`
   - Mac/Linux: `~/crowd-project`

**Option B: Git Clone**
```bash
git clone <repository-url> crowd-project
cd crowd-project
```

---

### 3. Open Project in VS Code

1. Launch **Visual Studio Code**
2. Click **File → Open Folder**
3. Navigate to and select `crowd-project`
4. Click **Select Folder**

---

## Method A: Automated Pipeline ⚡

**Perfect for first-time demo or quick evaluation**

### Step 1: Open Terminal in VS Code

Press `` Ctrl+` `` (backtick) or go to **View → Terminal**

You should see a terminal panel at the bottom.

---

### Step 2: Create Virtual Environment

Copy and paste this command:

**Windows:**
```powershell
python -m venv venv
```

**Mac/Linux:**
```bash
python3 -m venv venv
```

**Wait:** 10-20 seconds

**Expected output:** A new `venv` folder appears in your project

---

### Step 3: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
```

**Mac/Linux (Bash):**
```bash
source venv/bin/activate
```

**Success indicator:** You'll see `(venv)` at the start of your terminal line:
```
(venv) PS C:\crowd-project>
```

---

### Step 4: Install Dependencies

Copy and paste:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Wait:** 5-8 minutes (downloads ~2GB of packages)

**Expected output:** Lots of "Successfully installed..." messages

**Verify installation:**
```bash
pip list
```

You should see: `torch`, `opencv-python`, `pandas`, `fastapi`, `streamlit`, etc.

---

### Step 5: Download Model Weights

```bash
python scripts/download_weights_v2.py
```

**If this fails,** manually download:
1. Open: [https://huggingface.co/rootstrap-org/crowd-counting](https://huggingface.co/rootstrap-org/crowd-counting)
2. Click **"Files and versions"** tab
3. Download **`weights.pth`**
4. Save to: `models/csrnet_shanghaitech.pth`

---

### Step 6: Run the Automated Pipeline

**Windows:**
```powershell
.\run_pipeline.ps1
```

**Mac/Linux:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**What happens next:**
```
[1/5] Extracting frames...        (~30 seconds)
[2/5] Counting crowd density...   (~1-2 minutes)
[3/5] Building time series...     (~5 seconds)
[4/5] Training LSTM forecaster... (~30 seconds)
[5/5] Starting services...        (~5 seconds)
```

**Two new windows will open:**
- **API Server** (shows logs)
- **Browser** with dashboard at `http://localhost:8501`

---

### Step 7: View Your Dashboard

The browser should open automatically. If not, manually open:
```
http://localhost:8501
```

**You should see:**
- Current count: ~67 people
- Heatmap image (blue/red density visualization)
- Time series chart
- Forecast section with a button

**Click "Forecast next step"** button to see prediction!

---

### Step 8: Stop the Application

When done:
1. Go to the **API Server** window
2. Press `Ctrl+C`
3. Go to the **Dashboard** window
4. Press `Ctrl+C`

Both servers will shut down gracefully.

---

## Method B: Manual Step-by-Step 🔧

**Choose this if you want to understand each phase or customize parameters**

### Phase 1: Setup (One-time, ~15 minutes)

#### 1.1 Create & Activate Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verify:** Terminal shows `(venv)` prefix

---

#### 1.2 Install All Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pandas matplotlib scipy scikit-learn
pip install fastapi uvicorn[standard] streamlit pillow requests joblib
```

**Or use requirements.txt:**
```bash
pip install -r requirements.txt
```

**Verification:**
```bash
python -c "import torch, cv2, pandas, fastapi, streamlit; print('✅ All imports successful')"
```

---

#### 1.3 Download Model Weights

**Option A: Automatic**
```bash
python scripts/download_weights_v2.py
```

**Option B: Manual**
1. Visit: [https://huggingface.co/rootstrap-org/crowd-counting](https://huggingface.co/rootstrap-org/crowd-counting)
2. Download `weights.pth`
3. Place at: `models/csrnet_shanghaitech.pth`

**Verify:**
```bash
# Windows
dir models\csrnet_shanghaitech.pth

# Mac/Linux
ls -lh models/csrnet_shanghaitech.pth
```

Should show a 65MB file.

---

### Phase 2: Process Video (~3 minutes)

#### 2.1 Extract Frames from Video

```bash
python scripts/extract_frames.py raw_video/street1.mp4 anonymized_frames/street1 1
```

**Parameters explained:**
- `raw_video/street1.mp4` - input video file
- `anonymized_frames/street1` - where to save frames
- `1` - sample 1 frame per second (change to 0.5 for every 2 seconds)

**Expected output:**
```
📹 Video: raw_video/street1.mp4
   Total frames: 350
   Original FPS: 25.00
   Sampling rate: 1.0 fps
   Frame skip: every 25 frames

🔄 Extracting frames...
   Saved 10 frames...
Done. Saved 14 frames to anonymized_frames\street1
```

**Verify:**
```bash
# Windows
dir anonymized_frames\street1

# Mac/Linux
ls anonymized_frames/street1/
```

Should see: `frame_000000.jpg`, `frame_000001.jpg`, etc.

---

#### 2.2 Count Crowd Density (AI Model Inference)

```bash
python scripts/count_crowd.py anonymized_frames/street1 models/preds/street1 models/csrnet_shanghaitech.pth
```

**Parameters explained:**
- `anonymized_frames/street1` - folder with extracted frames
- `models/preds/street1` - where to save predictions
- `models/csrnet_shanghaitech.pth` - AI model weights

**Expected output:**
```
Loading CSRNet model...
Loading weights from models/csrnet_shanghaitech.pth
Found 14 frames to process
Processing 14/14: frame_000013.jpg
✅ Processed 14 frames
✅ Saved results to models\preds\street1\counts.json

📊 Summary:
   Average count: 65.23
   Min count: 48.50
   Max count: 89.10
```

**Verify:**
```bash
# Check JSON file
cat models/preds/street1/counts.json    # Mac/Linux
type models\preds\street1\counts.json   # Windows

# Check heatmaps
ls models/preds/street1/heatmaps/
```

You should see 14 heatmap images!

---

#### 2.3 Build Time Series Dataset

```bash
python scripts/build_timeseries.py models/preds/street1/counts.json counts_ts/street1_counts.csv
```

**Optional: Specify exact start time:**
```bash
python scripts/build_timeseries.py models/preds/street1/counts.json counts_ts/street1_counts.csv "2025-10-26 09:00:00" 1
```

Parameters:
- `"2025-10-26 09:00:00"` - when the video recording started
- `1` - FPS used during extraction

**Expected output:**
```
📊 Loaded 14 count records
   Holidays found: 0

✅ Saved time series to counts_ts\street1_counts.csv

📊 Time Series Summary:
   Total records: 14
   Time range: 2025-10-25 00:00:00 to 2025-10-25 00:00:13
   Average count: 65.23
   Min count: 48.50
   Max count: 89.10

📋 Feature columns: ['timestamp', 'crowd_count', 'frame_name', 'frame_index', 
                     'hour', 'minute', 'day_of_week', 'day_name', 'is_weekend', 
                     'day_of_month', 'month', 'year', 'is_holiday']
```

**Verify - Open the CSV:**
- In VS Code: Click `counts_ts/street1_counts.csv`
- Or in Excel/Numbers

You should see a table with timestamps and crowd counts!

---

#### 2.4 Train LSTM Forecasting Model

```bash
python scripts/train_lstm.py counts_ts/street1_counts.csv models/lstm_street1.pth
```

**Optional parameters:**
```bash
python scripts/train_lstm.py counts_ts/street1_counts.csv models/lstm_street1.pth 10 50
```
- `10` - lookback window (how many past timesteps to use)
- `50` - number of training epochs

**Expected output:**
```
============================================================
LSTM Crowd Forecasting - Training
============================================================

📊 Loaded 14 records from counts_ts/street1_counts.csv
📋 Using features: ['crowd_count', 'hour', 'day_of_week', 'is_weekend', 'is_holiday']

🔢 Created 4 sequences (lookback=10)
⚠️  WARNING: Very few sequences for training!
   This is expected for a 14-second video.
   For real deployment, you need hours/days of footage.

   Train: 3 sequences
   Test: 1 sequences

🚀 Training for 50 epochs...
Epoch [10/50] - Train Loss: 0.0234, Test Loss: 0.0189
Epoch [20/50] - Train Loss: 0.0156, Test Loss: 0.0134
Epoch [30/50] - Train Loss: 0.0098, Test Loss: 0.0087
Epoch [40/50] - Train Loss: 0.0067, Test Loss: 0.0056
Epoch [50/50] - Train Loss: 0.0045, Test Loss: 0.0039

📊 Final Results:
   Train RMSE: 0.0671
   Test RMSE: 0.0624

✅ Saved model to models\lstm_street1.pth
✅ Saved scaler to models\lstm_street1_scaler.pkl
```

**Verify:**
```bash
ls models/lstm_street1.pth
ls models/lstm_street1_scaler.pkl
```

Both files should exist!

---

### Phase 3: Run API & Dashboard (~1 minute)

#### 3.1 Start API Server

**Open Terminal 1** (in VS Code, you already have one open):

```bash
uvicorn app.api:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Leave this terminal running!** Don't close it.

---

#### 3.2 Test API (Optional)

Open browser and visit:
```
http://localhost:8000/docs
```

You should see **FastAPI** interactive documentation with these endpoints:
- `GET /health` - Check if API is running
- `GET /now` - Get current crowd count
- `POST /forecast` - Get future prediction

**Try the /health endpoint:**
1. Click **GET /health**
2. Click **Try it out**
3. Click **Execute**
4. Should return: `{"ok": true, "model_loaded": true}`

---

#### 3.3 Start Dashboard

**Open Terminal 2** in VS Code:
- Click the **+** button in the terminal panel
- Or press `` Ctrl+Shift+` ``

**Activate venv again in the new terminal:**
```bash
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Mac/Linux
```

**Run dashboard:**
```bash
streamlit run app/dashboard.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Browser should open automatically!**

---

#### 3.4 Interact with Dashboard

The browser shows three sections:

**1. Current count**
- Shows: Latest count: 67.04
- Shows: Heatmap image (blue density visualization)

**2. Time series**
- Line chart showing crowd changes over time

**3. Forecast**
- Button: "Forecast next step"
- Click it!

**Expected forecast response:**
```json
{
  "when": "2025-10-25T19:50:00",
  "predicted_count": 1.867638,
  "advisory": "Low"
}
```

This means:
- **when**: Time you requested prediction for
- **predicted_count**: AI's prediction for crowd density
- **advisory**: Low/Medium/High (travel recommendation)

---

#### 3.5 Stop Servers

When you're done:

**Terminal 1 (API):**
- Press `Ctrl+C`
- Type `Y` if prompted

**Terminal 2 (Dashboard):**
- Press `Ctrl+C`
- Type `Y` if prompted

Both servers shut down cleanly.

---

## Understanding Your Results 📊

### Files Generated

After running the pipeline, you'll have:

```
crowd-project/
├── anonymized_frames/street1/
│   ├── frame_000000.jpg           [14 images, ~5MB total]
│   ├── frame_000001.jpg
│   └── ...
│
├── models/preds/street1/
│   ├── counts.json                [Crowd counts, ~2KB]
│   └── heatmaps/
│       ├── frame_000000_heatmap.jpg  [14 heatmaps, ~3MB]
│       └── ...
│
├── counts_ts/
│   └── street1_counts.csv         [Time series data, ~1KB]
│
└── models/
    ├── lstm_street1.pth           [Trained model, ~50KB]
    └── lstm_street1_scaler.pkl    [Feature scaler, ~1KB]
```

---

### What Each File Means

**1. anonymized_frames/** - Extracted video frames
- Raw input for the AI model
- Can delete after counting to save space
- Keep a few for documentation/demo

**2. counts.json** - Crowd density per frame
```json
[
  {
    "frame": "anonymized_frames/street1/frame_000000.jpg",
    "frame_name": "frame_000000.jpg",
    "count": 67.04,
    "frame_index": 0
  },
  ...
]
```

**3. heatmaps/** - Visual density maps
- Blue = low density, Red = high density
- Great for presentations and validation
- Can delete most (keep 2-3 samples)

**4. street1_counts.csv** - Time series dataset
```csv
timestamp,crowd_count,frame_name,hour,day_of_week,is_weekend,is_holiday
2025-10-25 00:00:00,67.04,frame_000000.jpg,0,4,0,0
2025-10-25 00:00:01,65.23,frame_000001.jpg,0,4,0,0
...
```

**5. lstm_street1.pth** - Trained forecasting model
- Used by API to make predictions
- Keep this file!

---

## Using Your Own Video 🎥

### Step 1: Prepare Your Video

**Requirements:**
- Format: .mp4, .avi, or .mov
- Length: Any (but longer videos take more time)
- Content: CCTV-style footage of people/crowds

**Place video in:**
```
raw_video/my_location.mp4
```

---

### Step 2: Know Your Video Details

**Important:** You need to know:
1. **When the video was recorded** (date and time)
2. **What FPS to sample** (0.5 = every 2 seconds, 1 = every 1 second)

Example:
- Video recorded: Oct 26, 2025 at 9:00 AM
- Want to sample: Every 2 seconds (0.5 fps)

---

### Step 3: Run Pipeline with Your Video

**Automated (easiest):**
```powershell
# Windows
.\run_pipeline.ps1 -VideoPath "raw_video/my_location.mp4" -FPS 0.5 -StartTime "2025-10-26 09:00:00"
```

**Manual step-by-step:**

```bash
# 1. Extract frames (change 0.5 to your desired FPS)
python scripts/extract_frames.py raw_video/my_location.mp4 anonymized_frames/my_location 0.5

# 2. Count crowds
python scripts/count_crowd.py anonymized_frames/my_location models/preds/my_location models/csrnet_shanghaitech.pth

# 3. Build time series (use your actual start time!)
python scripts/build_timeseries.py models/preds/my_location/counts.json counts_ts/my_location_counts.csv "2025-10-26 09:00:00" 0.5

# 4. Train model
python scripts/train_lstm.py counts_ts/my_location_counts.csv models/lstm_my_location.pth

# 5. Update API to use new model (edit app/api.py):
# Change: MODEL_PATH = Path("models/lstm_my_location.pth")

# 6. Start services
uvicorn app.api:app --reload --port 8000
streamlit run app/dashboard.py
```

---

### For Very Long Videos

**Problem:** A 2-hour video at 1 fps = 7,200 frames = slow processing

**Solutions:**

**Option A: Lower FPS**
```bash
python scripts/extract_frames.py raw_video/long.mp4 anonymized_frames/long 0.2
```
This samples 1 frame every 5 seconds (0.2 fps)

**Option B: Split Video**
Use a video editor to split into 10-minute chunks:
- `long_part1.mp4`, `long_part2.mp4`, etc.
- Process each separately
- Combine the resulting CSVs

**Option C: Process Chunks**
```bash
# Process hour 1
ffmpeg -i raw_video/long.mp4 -ss 00:00:00 -t 01:00:00 raw_video/long_hour1.mp4
python scripts/extract_frames.py raw_video/long_hour1.mp4 anonymized_frames/long_hour1 0.5

# Process hour 2
ffmpeg -i raw_video/long.mp4 -ss 01:00:00 -t 01:00:00 raw_video/long_hour2.mp4
python scripts/extract_frames.py raw_video/long_hour2.mp4 anonymized_frames/long_hour2 0.5

# Then process each separately and combine CSVs
```

---

## Troubleshooting 🔧

### Error: "Python not found"

**Solution:**
1. Install Python from [python.org](https://www.python.org/downloads/)
2. During installation, CHECK "Add Python to PATH"
3. Restart computer
4. Verify: `python --version`

---

### Error: "Module not found" or Import errors

**Solution:**
```bash
# Make sure venv is activated (you see (venv) in terminal)
.\venv\Scripts\activate

# Reinstall all dependencies
pip install -r requirements.txt

# Verify specific module
python -c "import torch; print(torch.__version__)"
```

---

### Error: "FileNotFoundError: weights.pth"

**Solution:**
Manually download weights:
1. Go to: [https://huggingface.co/rootstrap-org/crowd-counting](https://huggingface.co/rootstrap-org/crowd-counting)
2. Click "Files and versions"
3. Download `weights.pth`
4. Save exactly to: `models/csrnet_shanghaitech.pth`

---

### Error: "Connection refused" when clicking Forecast

**Solution:**
API server isn't running. In a separate terminal:
```bash
.\venv\Scripts\activate
uvicorn app.api:app --reload --port 8000
```

Leave it running, then try dashboard again.

---

### Error: "Port 8000 already in use"

**Solution:**
Something else is using port 8000.

**Option A: Kill it**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Mac/Linux
lsof -ti:8000 | xargs kill -9
```

**Option B: Use different port**
```bash
uvicorn app.api:app --reload --port 8001
```
Then edit `app/dashboard.py` line 23:
```python
api = "http://localhost:8001/forecast"
```

---

### Dashboard shows blank/loading forever

**Solution:**
1. Check both terminals are running (API + Dashboard)
2. Refresh browser (Ctrl+F5)
3. Check browser console for errors (F12)
4. Verify files exist:
   - `models/preds/street1/counts.json`
   - `counts_ts/street1_counts.csv`

---

### Very slow processing

**Cause:** Large video or high FPS

**Solution:**
- Use lower FPS: `0.2` or `0.5` instead of `1`
- Process video in smaller chunks
- For real deployment, consider GPU acceleration

---

## Cleaning Up Storage 🧹

After successful run, you can delete large files to save space.

### What to Keep (Essential)

```
✅ Keep:
├── counts_ts/street1_counts.csv        (~1KB)
├── models/lstm_street1.pth             (~50KB)
├── models/lstm_street1_scaler.pkl      (~1KB)
├── models/preds/street1/counts.json    (~2KB)
├── models/csrnet_shanghaitech.pth      (~65MB)
└── 2-3 sample heatmaps for demo        (~500KB)
```

### What to Delete (Optional)

```
❌ Can Delete:
├── anonymized_frames/street1/          (~5MB+)
│   └── All frame images
├── models/preds/street1/heatmaps/      (~3MB+)
│   └── Most heatmaps (keep 2-3 samples)
└── raw_video/street1.mp4 (after processing, archive elsewhere)
```

### Cleanup Commands

**Windows:**
```powershell
# Delete frames
Remove-Item -Recurse -Force anonymized_frames\street1

# Delete most heatmaps (keep first 3)
Get-ChildItem models\preds\street1\heatmaps -Filter "*.jpg" | Select-Object -Skip 3 | Remove-Item
```

**Mac/Linux:**
```bash
# Delete frames
rm -rf anonymized_frames/street1

# Delete most heatmaps (keep first 3)
ls models/preds/street1/heatmaps/*.jpg | tail -n +4 | xargs rm
```

**Result:** Free up 5-20MB+ per video location

---

## Export Reports for Submission 📤

Generate comprehensive analysis reports:

```bash
python scripts/export_results.py counts_ts/street1_counts.csv reports/
```

**Generated files:**
```
reports/
├── summary_20251025_201530.json          [Statistics summary]
├── hourly_stats_20251025_201530.csv      [Peak hours analysis]
├── daily_stats_20251025_201530.csv       [Day patterns]
├── full_data_20251025_201530.csv         [Complete dataset]
└── report_20251025_201530.md             [Markdown report]
```

**Open `report_*.md` in VS Code** - Beautiful formatted report you can convert to PDF!

---

## Quick Command Reference 📋

**Setup (one-time):**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_weights_v2.py
```

**Full Pipeline (automated):**
```bash
.\run_pipeline.ps1
```

**Manual Pipeline:**
```bash
python scripts/extract_frames.py raw_video/street1.mp4 anonymized_frames/street1 1
python scripts/count_crowd.py anonymized_frames/street1 models/preds/street1 models/csrnet_shanghaitech.pth
python scripts/build_timeseries.py models/preds/street1/counts.json counts_ts/street1_counts.csv
python scripts/train_lstm.py counts_ts/street1_counts.csv models/lstm_street1.pth
uvicorn app.api:app --reload --port 8000
streamlit run app/dashboard.py
```

**Export reports:**
```bash
python scripts/export_results.py counts_ts/street1_counts.csv reports/
```

---

## Success Checklist ✅

Before demo/presentation, verify:

- [ ] Virtual environment activated (`(venv)` in terminal)
- [ ] All dependencies installed (`pip list` shows packages)
- [ ] Weights downloaded (`models/csrnet_shanghaitech.pth` exists, 65MB)
- [ ] Sample video present (`raw_video/street1.mp4`)
- [ ] Frames extracted successfully (check `anonymized_frames/street1/`)
- [ ] Counts generated (`models/preds/street1/counts.json` exists)
- [ ] Heatmaps created (`models/preds/street1/heatmaps/*.jpg`)
- [ ] CSV built (`counts_ts/street1_counts.csv` exists)
- [ ] Model trained (`models/lstm_street1.pth` exists)
- [ ] API starts without errors (`uvicorn` command works)
- [ ] Dashboard loads in browser (`http://localhost:8501`)
- [ ] Forecast button returns predictions

**If all checked ✅, you're ready to present!**

---

## Time Estimates ⏱️

**Setup (First Time):**
- Install Python: 5-10 min
- Download project: 1 min
- Create venv: 1 min
- Install dependencies: 5-8 min
- Download weights: 1-2 min
- **Total: 15-20 minutes**

**Running Pipeline:**
- Extract frames: 30 sec
- Count crowds: 1-2 min
- Build time series: 5 sec
- Train LSTM: 30 sec
- Start services: 10 sec
- **Total: 3-5 minutes**

**With 1-hour video:**
- Extract (0.5 fps): 2-3 min
- Count: 10-15 min
- Rest: 2 min
- **Total: 15-20 minutes**

---

## Need Help? 💬

**Check logs:**
- API errors: See Terminal 1 (uvicorn)
- Dashboard errors: See Terminal 2 (streamlit)
- Python errors: Read full stack trace

**Common issues:**
1. Module not found → Reinstall with `pip install -r requirements.txt`
2. Connection error → Make sure API is running first
3. File not found → Check paths, run from project root
4. Port in use → Use different port or kill process

**Still stuck?**
- Review troubleshooting section above
- Check that ALL setup steps completed successfully
- Try restarting VS Code and terminals

---

**🎉 Congratulations! You now know how to run the complete AI Crowd Density Prediction system from scratch!**

---

*Last updated: October 2025*