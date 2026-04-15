# Scam-Job-Recruitment

Detection of fake recruitment emails using a hybrid deep learning pipeline.

## Supported upload formats

- `.txt`
- `.pdf`
- `.docx`

## Model download behavior

The app expects the model at `model/fake_recruitment_detector.pth`.

- If the file exists locally, it is used directly.
- If missing, the backend auto-downloads it from Google Drive on first run using `gdown`.
- You can override the download URL using environment variable `MODEL_URL`.

Default Drive source:
`https://drive.google.com/file/d/1cE0BzOZsffaK5kvCCK8Tv_qi-ZN-MrVZ/view?usp=drive_link`

## Local run

1. Open terminal in project root.
2. Create and activate virtual env.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start backend:
   `python backend/app.py`
5. Open:
   `http://127.0.0.1:5000`

## What to upload to GitHub

Upload:
- `backend/`
- `frontend/`
- `model/*.py` (code only)
- `requirements.txt`
- `requirements-train.txt` (optional, for trainers)
- `.python-version` (needed for Render / consistent Python)
- `.gitignore`
- `README.md`

Do not upload:
- `.venv/`
- `model/fake_recruitment_detector.pth` (large binary)
- `backend/analysis_reports.db`
- `reports/` contents
- `__pycache__/`

## Deploy on Render (step-by-step)

Render’s default Python is **3.14.x**, which often breaks ML wheels (e.g. `scikit-learn`). This repo pins **3.11.11** via `.python-version` so installs use pre-built wheels.

1. Push this repo to GitHub (include `.python-version` and `requirements.txt`).
2. In [Render](https://render.com): **New** → **Web Service** → connect the repo.
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `python backend/app.py` (uses Render’s `PORT` automatically)
5. **Environment variables** (recommended):
   - `MODEL_URL` = `https://drive.google.com/uc?id=1cE0BzOZsffaK5kvCCK8Tv_qi-ZN-MrVZ`
   - If Render still picks Python 3.14, set `PYTHON_VERSION` = `3.11.11` ([docs](https://render.com/docs/python-version)).
6. Deploy. First boot may be slow (PyTorch + model download).

### Training locally (optional)

Training script needs extra packages:

`pip install -r requirements-train.txt`
