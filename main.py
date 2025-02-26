from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from openai import OpenAI
import boto3
import uvicorn
import os

load_dotenv()
app = FastAPI(
    title="LLM API Server for test-prep-api",
    description="LLM AI model server for test-prep-ai",
    version="0.1.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포할떄는 도메인으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# aws s3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION"),
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

bucket = os.environ.get("S3_BUCKET_NAME")

@app.get("/input-pdf/{filename}")
async def process_pdf(filename: str):
    try:

        s3_key = filename
        presigned_url = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=1800
        )
        print(presigned_url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
