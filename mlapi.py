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

with open('models/KNN.pkl','rb') as f:
    knn = pickle.load(f)

with open('models/RF.pkl','rb') as f:
    rf = pickle.load(f)
with open('models/SVM.pkl','rb') as f:
    svm = pickle.load(f)
with open('models/LR.pkl','rb') as f:
    lr = pickle.load(f)

@app.post('/knn_endpoint')
@app.post('/rf_endpoint')
@app.post('/svm_endpoint')
@app.post('/lr_endpoint')

async def knn_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()],columns = item.dict().keys())
    yhat = knn.predict(df)

    return {'prediction':float(yhat)}

async def rf_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()],columns = item.dict().keys())
    yhat = rf.predict(df)

    return {'prediction':float(yhat)}

async def svm_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()],columns = item.dict().keys())
    yhat = svm.predict(df)

    return {'prediction':float(yhat)}

async def lr_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()],columns = item.dict().keys())
    yhat = lr.predict(df)

    return {'prediction':float(yhat)}