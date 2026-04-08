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

@app.post("/step")
def step(action: SupportAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/")
def health():
    return {"status": "ok"}

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == '__main__':
    main()
