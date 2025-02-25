# test_import.py
try:
    from fastapi import FastAPI
    print("FastAPI import successful")
except ImportError as e:
    print(f"FastAPI import failed: {e}")

try:
    import uvicorn
    print("Uvicorn import successful")
except ImportError as e:
    print(f"Uvicorn import failed: {e}")

try:
    from pydantic import BaseModel
    print("Pydantic import successful")
except ImportError as e:
    print(f"Pydantic import failed: {e}")
