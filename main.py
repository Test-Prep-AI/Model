from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import boto3
import uvicorn
import os
import re
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from test_model import PDFQuestionGenerator

load_dotenv()
app = FastAPI(
    title="LLM API Server for test-prep-api",
    description="LLM AI model server for test-prep-ai",
    version="0.1.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포할때는 도메인으로 변경
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

bucket = os.environ.get("S3_BUCKET_NAME")

# request 모델
class QuestionRequest(BaseModel):
    fileName: str
    types: List[int]
    level: str
    message: str = ""  # Optional message 디폴트는 " "

# 객관식 문제 파싱 
def parse_multiple_choice_questions(text):
    questions = []
    pattern = r"(\d+)\.\s+(.+?)\s+-\s+a\.\s+(.+?)\s+-\s+b\.\s+(.+?)\s+-\s+c\.\s+(.+?)\s+-\s+d\.\s+(.+?)\s+정답:\s+(.+?)\s+\(해설:\s+(.+?)\)"
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        question_number = match.group(1)
        question_text = match.group(2).strip()
        option_a = match.group(3).strip()
        option_b = match.group(4).strip()
        option_c = match.group(5).strip()
        option_d = match.group(6).strip()
        answer = match.group(7).strip()
        explanation = match.group(8).strip()
        
        question = {
            "number": int(question_number),
            "text": question_text,
            "options": {
                "a": option_a,
                "b": option_b,
                "c": option_c,
                "d": option_d
            },
            "answer": answer,
            "explanation": explanation
        }
        questions.append(question)
    
    return questions

# 단답형 문제 파싱
def parse_short_answer_questions(text):
    questions = []
    pattern = r"(\d+)\.\s+(.+?)\s+정답:\s+(.+?)\s+\((.+?)\)"
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        question_number = match.group(1)
        question_text = match.group(2).strip()
        answer = match.group(3).strip()
        explanation = match.group(4).strip()
        
        question = {
            "number": int(question_number),
            "text": question_text,
            "answer": answer,
            "explanation": explanation
        }
        questions.append(question)
    
    return questions

# 서술형 문제 파싱
def parse_essay_questions(text):
    questions = []
    pattern = r"문제\s+(\d+):\s+(.+?)\s+답안:\s+(.+?)(?=문제\s+\d+:|$)"
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        question_number = match.group(1)
        question_text = match.group(2).strip()
        answer = match.group(3).strip()
        
        question = {
            "number": int(question_number),
            "text": question_text,
            "answer": answer
        }
        questions.append(question)
    
    return questions

@app.get("/")
def root():
    return {"hello test-prep-ai"}

@app.post("/problems/input-pdf")
async def process_pdf(request: QuestionRequest):
    try:
        # PDF 읽기 위한 S3 Presigned URL 생성
        s3_key = request.fileName
        presigned_url = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=1800
        )
        
        # model 객체 생성
        generator = PDFQuestionGenerator()
        
        all_questions = []
        all_referenced_pages = set()
        
        # 문제 type array 매핑
        question_types = ["객관식", "단답형", "서술형"]
        parsing_functions = {
            "객관식": parse_multiple_choice_questions,
            "단답형": parse_short_answer_questions,
            "서술형": parse_essay_questions
        }
        
        # 문제 type 별로 문제 생성
        for idx, count in enumerate(request.types):
            if count > 0:
                question_type = question_types[idx]
                
                # 문제 생성
                result, pages = generator.generate_questions(
                    presigned_url,  # s3 pdf url
                    count,  # 생성할 문제 수
                    request.level,  # 문제 난이도
                    question_type,  # 문제 유형 (객관식, 단답형, 서술형)
                    request.message  # 옵셔널 메시지
                )
                
                
                # 문제 유형별 파싱 함수 호출
                parsed_questions = parsing_functions[question_type](result)
                
                # append all
                all_questions.append({
                    "type": question_type,
                    "questions": parsed_questions
                })
                
                # 참고 한 페이지 수
                for page in pages:
                    all_referenced_pages.add(page)
        

        topic = generator.generate_overall_topic(presigned_url)
    
        # response 생성
        response = {
            "questions": all_questions,
            "referencedPages": sorted(list(all_referenced_pages)),
            "topic": topic
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/input-pdf/{filename}")
async def get_pdf_url(filename: str):
    try:
        s3_key = filename
        presigned_url = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": s3_key}, ExpiresIn=1800
        )
        return JSONResponse(content={"presignedUrl": presigned_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)