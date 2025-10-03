from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time
from hyperclova_local_client import HyperCLOVALocalClient

# FastAPI 앱 생성
app = FastAPI(
    title="HyperCLOVA X SEED Chat API",
    description="사용자 질문에 답변하는 간단한 챗봇 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class ChatRequest(BaseModel):
    question: str
    model_name: Optional[str] = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

class ChatResponse(BaseModel):
    answer: str
    model_name: str
    response_time: float

# 전역 클라이언트
client = HyperCLOVALocalClient()

@app.get("/")
async def root():
    """서버 상태 확인"""
    return {"message": "HyperCLOVA X SEED Chat API", "status": "running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """사용자 질문에 답변"""
    try:
        # 모델이 로드되지 않았다면 자동 로드
        if not client.model:
            client.load_model()
        
        start_time = time.time()
        # 짧고 간결한 답변만 생성
        answer = client.generate_text(request.question, max_length=30)
        response_time = time.time() - start_time
        
        if answer is None:
            raise HTTPException(status_code=500, detail="답변 생성에 실패했습니다")
        
        return ChatResponse(
            answer=answer,
            model_name=request.model_name,
            response_time=response_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

if __name__ == "__main__":
    print("HyperCLOVA X SEED Chat API 서버 시작")
    print("API 문서: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )