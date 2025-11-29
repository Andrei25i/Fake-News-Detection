from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import NewsRequest, NewsResponse
from services import load_models, predict
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(title="Fake News Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=False, 
    allow_methods=["*"],    
    allow_headers=["*"], 
)

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
def read_root():
    return {"status": "Server Operational"}

@app.post("/predict", response_model=NewsResponse)
async def predict_endpoint(request: NewsRequest):
    try:
        predictions = await predict(
            request.title,
            request.text
        )

        if not predictions:
            raise HTTPException(status_code=500, detail="No models available for prediction")

        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    print(f"Server runs on http://{HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)