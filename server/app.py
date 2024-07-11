from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import json
from datetime import datetime

from model.main import MiniZapWithEDA


app = FastAPI()
mini_zap = MiniZapWithEDA()

class TriggerPayload(BaseModel):
    value: float
    source: str = "api"

@app.post("/trigger")
async def trigger_flow(payload: TriggerPayload):
    try:
        full_payload = {
            "timestamp": datetime.now().isoformat(),
            "value": payload.value,
            "source": payload.source
        }
        result = mini_zap.run_flow(full_payload)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_results():
    try:
        results = mini_zap.es.search(index="mini_zap_flow_results", body={"query": {"match_all": {}}}, size=10)
        return [hit['_source'] for hit in results['hits']['hits']]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plot")
async def get_plot():
    return FileResponse("value_over_time.png")

@app.get("/")
async def read_root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)