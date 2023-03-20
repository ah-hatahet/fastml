from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd


app = FastAPI()

class ScoringItem(BaseModel):
    SepalLength: float
    SepalWidth: float
    PetalLength: float
    PetalWidth: float

with open('model.pkl','rb') as f:
    model = pickle.load(f)
    

@app.post('/')


async def scoring_ençdpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()],columns = item.dict().keys())
    yhat = model.predict(df)

    return {'prediction':float(yhat)}