import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
import uvicorn
from support_env import SupportEnv
from models import SupportAction, SupportObservation

app = FastAPI(title="Support Triage OpenEnv")
env = SupportEnv()

@app.post("/reset")
def reset() -> SupportObservation:
    return env.reset()

from pydantic import BaseModel

class StepResponse(BaseModel):
    observation: SupportObservation
    reward: float
    done: bool
    info: dict

@app.post("/step", response_model=StepResponse)
def step(action: SupportAction) -> StepResponse:
    obs, reward, done, info = env.step(action)
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info
    )

@app.get("/state")
def state() -> dict:
    return env.state()

@app.get("/")
def health():
    return {"status": "ok"}

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == '__main__':
    main()
