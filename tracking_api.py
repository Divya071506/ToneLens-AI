
import os

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="ToneLens AI Tracking API")

# Allow your Hugging Face Space's browser origin.
# This does not make your local API publicly reachable.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://diviii15-tonelens.hf.space"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Use the MLflow server you already run locally.
MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("ToneLens-AI-Experiment")


class Prediction(BaseModel):
    sentiment: str
    confidence: float = Field(ge=0.0, le=1.0)


@app.get("/")
def home():
    return {
        "status": "running",
        "service": "ToneLens AI Tracking API"
    }


@app.post("/log-prediction")
def log_prediction(data: Prediction):
    sentiment = data.sentiment.strip().upper()

    if sentiment not in {"POSITIVE", "NEGATIVE"}:
        raise HTTPException(
            status_code=400,
            detail="Sentiment must be POSITIVE or NEGATIVE"
        )

    try:
        with mlflow.start_run(
            run_name="ToneLens-Prediction"
        ):
            mlflow.log_param(
                "model",
                "distilbert-sst2"
            )
            mlflow.log_metric(
                "prediction_confidence",
                data.confidence
            )
            mlflow.set_tag(
                "predicted_sentiment",
                sentiment
            )
            mlflow.set_tag(
                "event_type",
                "user_prediction"
            )

        return {
            "logged": True,
            "sentiment": sentiment,
            "confidence": data.confidence
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"MLflow logging failed: {error}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
