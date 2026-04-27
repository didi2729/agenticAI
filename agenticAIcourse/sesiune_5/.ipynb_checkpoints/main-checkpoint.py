import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
 

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["regressor"] = joblib.load("model.joblib")
    yield
    ml_model.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.sepal_length,
        features.sepal_width
    ]]
    prediction = ml_model["regressor"].predict(input_data)
    return {"prediction": prediction[0], "model_version": "1.0"}
