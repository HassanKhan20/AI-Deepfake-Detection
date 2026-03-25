from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from app.utils import load_image_from_base64
from app.inference import predict


app = FastAPI(title='AI Deepfake Detection API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def _ensure_image(data: str):
    try:
        return load_image_from_base64(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f'Invalid image data: {exc}')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictResponse)
def predict_image(req: PredictRequest):
    img = _ensure_image(req.image_base64)
    return predict(img)


@app.post('/predict/batch', response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    if not req.image_base64_list:
        raise HTTPException(status_code=400, detail='Empty list')

    results = []
    for image_b64 in req.image_base64_list:
        img = _ensure_image(image_b64)
        results.append(predict(img))

    return {'results': results}
