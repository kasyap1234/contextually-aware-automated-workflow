from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List
import json
from datetime import datetime
import numpy as np

from model.main import MiniZapWithEDA

app = FastAPI()
mini_zap = MiniZapWithEDA()

class TriggerPayload(BaseModel):
    value: int
    source: str = "api"

@app.post("/trigger")
async def trigger_flow(payloads: List[TriggerPayload]):
    try:
        full_payloads = [
            {
                "timestamp": datetime.now().isoformat(),
                "value": payload.value,
                "source": payload.source
            }
            for payload in payloads
        ]
        result = mini_zap.run_flow(full_payloads)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_results():
    try:
        df = mini_zap.process_data()
        
        
        df = df.replace([np.inf, -np.inf, np.nan], None)
        
        results = df.to_dict(orient='records')
        return results[:10] 
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
