# serve.py
import torch
from src.mtl import MultiTaskModel
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
device = torch.device('cpu')
model = MultiTaskModel(backbone_pth='models/backbone', device=device)
model.load('best_model')
model.eval()

class In(BaseModel):
    sentence: str

@app.post("/predict")
def predict(payload: In):
    with torch.no_grad():
        outA, outB = model.predict(payload.sentence)
        outA = outA.cpu().numpy().tolist()
        outB = outB.cpu().numpy().tolist()
        mappingA = {0: "Politics", 1: "Sports", 2: "Technology", 3: "Entertainment"}
        mappingB = {0: "Positive", 1: "Neutral", 2: "Negative"}
        outA = [mappingA[i] for i in outA]
        outB = [mappingB[i] for i in outB]
        
    return {"taskA": outA, "taskB": outB}
