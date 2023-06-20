from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split

app = FastAPI()
data = pd.read_csv("C:/Users/didit/Downloads/MiningOper/mineplant.csv")

class InputData(BaseModel):
    Iron_Feed: float
    Silica_Feed: float
    Iron_Concentrate: float

rf_model = joblib.load("rf_model.pkl")

# Prepare the data for prediction
X = data.drop(['% Silica Concentrate'], axis=1)
y = data['% Silica Concentrate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.Iron_Feed, data.Silica_Feed, data.Iron_Concentrate]])
    prediction = rf_model.predict(features)
    return {"prediction": prediction[0]}

@app.get("/predict")
def get_predict(Iron_Feed: float, Silica_Feed: float, Iron_Concentrate: float):
    features = np.array([[Iron_Feed, Silica_Feed, Iron_Concentrate]])
    prediction = rf_model.predict(features)
    return {"prediction": prediction[0]}
