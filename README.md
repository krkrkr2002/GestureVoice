# GestureVoice (Local Webcam → Gesture → Voice)

This repo already contains a trained model (`backend/model.h5`). You can run **fully locally** (no paid/online “credits”) to:

- open your **webcam**
- detect your **hand gesture**
- predict the label
- **speak** the output out loud (offline TTS)

## Requirements (important)

- **Python 3.10.x** (recommended: `python-3.10.13` as in `runtime.txt`)
  - If you use Python 3.12, `tensorflow` typically won’t install on Windows.

## Run the “one go” webcam + voice app (Windows / PowerShell)

From the repo root (`GestureVoice-main/`):

```powershell
cd .\backend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python .\live_camera_tts.py
```

- Press **`q`** to quit the camera window.

## Run the backend API (optional)

```powershell
cd .\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then your API is on:
- `http://localhost:8000/`
- `http://localhost:8000/docs`

## Run the frontend (optional)

```powershell
cd .\frontend
npm install
npm run dev
```

Open `http://localhost:3000`

