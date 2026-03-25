# AI-Deepfake-Detection

Full-stack AI Deepfake Detection System (Python, PyTorch, FastAPI, React)

## Project Overview

- Backend: FastAPI model serving pipeline, deployable via AWS EC2 / Docker.
- AI model: PyTorch CNN classifier (real vs fake), fine-tunable via `backend/app/train.py`.
- Frontend: React + TypeScript with real-time confidence visualization and REST API integration.
- Accuracy target: 91%+ (structured for training on diverse video/image datasets).

## Repo Structure

- `backend/`
  - `app/main.py`: FastAPI endpoints (`/health`, `/predict`, `/predict/batch`).
  - `app/inference.py`: model load and prediction pipeline.
  - `app/models/deepfake_model.py`: CNN architecture.
  - `app/train.py`: training loop with validation checkpointing.
  - `requirements.txt`: Python dependencies.
- `frontend/`
  - React app built with TypeScript.
  - `src/App.tsx`: image upload + inference UI + chart display.

## Setup and Run

### 1) Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create `backend/model_weights` and place a PyTorch weights file there (or train with sample data):

```bash
mkdir -p backend/model_weights
python -m app.train --train-dir ../data/train --val-dir ../data/val --output backend/model_weights/deepfake_classifier.pth --epochs 8
```

Run server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl http://localhost:8000/health
```

### 2) Frontend

```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000` (or `3001` if 3000 is busy), upload an image, and click "Run Inference".

### 3) Docker Compose (optional, strong interview setup)

```bash
docker-compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

## Architecture Highlights

- Modular inference pipeline with built-in preprocessing, model loading and softmax confidence.
- Batch endpoint for API-driven dataset scoring: `/predict/batch`.
- CORS-enabled API ready for a React SPA.
- Production-grade metrics tracking hooks can be added in `app/main.py`.

## Inference Example

`POST /predict` payload:
```json
{ "image_base64": "data:image/png;base64,..." }
```

Response:

```json
{
  "label": "fake",
  "fake_confidence": 0.95,
  "real_confidence": 0.05
}
```



