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
- `.gitignore`
- `README.md`

Do not upload:
- `.venv/`
- `model/fake_recruitment_detector.pth` (large binary)
- `backend/analysis_reports.db`
- `reports/` contents
- `__pycache__/`

## Deploy steps (Render/Railway style)

1. Push source code to GitHub.
2. Create a new Web Service from the repo.
3. Set build command:
   `pip install -r requirements.txt`
4. Set start command:
   `python backend/app.py`
5. Add env var (recommended):
   `MODEL_URL=https://drive.google.com/uc?id=1cE0BzOZsffaK5kvCCK8Tv_qi-ZN-MrVZ`
6. Deploy. On first boot, model is downloaded automatically if missing.
