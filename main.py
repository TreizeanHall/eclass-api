from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from ml.train_test_split import train_model
from ml.predict import predict_one

import os

app = FastAPI(title="Email Classifier API")

API_KEY = os.getenv("API_KEY", "")  # set this when running docker

def require_api_key(authorization: str | None = Header(default=None)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

class PredictRequest(BaseModel):
    subject: str | None = ""
    body: str | None = ""
    top_k: int = Field(default=3, ge=1, le=10)


class TrainRequest(BaseModel):
    lookback_days: int = Field(default=30, ge=1, le=365)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train", dependencies=[Depends(require_api_key)])
def train(req: TrainRequest):
    return train_model()

@app.post("/predict")
def predict(req: PredictRequest):
    return predict_one(subject=req.subject or "", description=req.body or "")



