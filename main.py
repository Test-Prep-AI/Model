# main.py
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="My FastAPI Application",
    description="A simple RESTful API built with FastAPI",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "test-prep-ai API"}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
