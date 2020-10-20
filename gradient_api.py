from typing import List, Optional
from fastapi import Depends, FastAPI
from pydantic import BaseModel
import uvicorn
import os
from aitextgen import aitextgen
import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
#app = FastAPI()
app = FastAPI(openapi_prefix='/model-serving/'+os.getenv("HOSTNAME").split('-')[0])
ai = aitextgen(model="/models/pytorch_model_124M.bin", config="/models/config_124M.json", to_gpu=True)

def get_model():
    return ai

class GPT2Request(BaseModel):
    prompt: str
    samples: int
    batch_size: int = Optional
    max_length: int = Optional
    temperature: int = Optional


class GPT2Response(BaseModel):
    generated: List[str]

class HeartbeatResult(BaseModel):
    is_alive: bool


@app.post("/predict", response_model=List[str])
def predict(request: GPT2Request, model = Depends(get_model)):
    response = ai.generate(n=request.samples,
            batch_size=request.samples,
            prompt=request.prompt,
            max_length=256,
            temperature=1.0,
            top_p=0.9,
            return_as_list=True)
    
    # Bold the prompt if printing to console
    print(response)
    return GPT2Response(
        generated=response
    )

@app.get("/", response_model=HeartbeatResult)
def get_heartbeat()-> HeartbeatResult:
    heartbeat = HeartbeatResult(is_alive=True)
    return heartbeat

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")