from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from config import settings
from direct_pinecone_service import get_vectorstore_service
from hyperclova_client import get_hyperclova_client
from mangum import Mangum
from database import db_instance
from routers import auth

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# 보안 미들웨어 설정
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
    )

# CORS middleware 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth.router)

# MongoDB 연결 이벤트
@app.on_event("startup")
async def startup_db_client():
    """앱 시작 시 MongoDB 연결"""
    await db_instance.connect_db()
    logger.info("MongoDB Atlas 연결 완료")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": f"{settings.APP_NAME} is running",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": settings.APP_VERSION
    }

@app.get("/metrics")
async def metrics():
    """메트릭 엔드포인트 (프로덕션에서는 Prometheus 등 사용)"""
    if not settings.ENABLE_METRICS:
        return {"message": "Metrics disabled"}
    
    return {
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "log_level": settings.LOG_LEVEL
    }


# 요청/응답 모델
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    query: str
    k: int = 3  # 검색할 문서 수
    include_sources: bool = True  # 출처 포함 여부


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    answer: str
    sources: list = []


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    수업계획서 기반 챗봇 엔드포인트
    
    1. 질문 의도 분류 (수업 관련 vs 일상 대화)
    2. 수업 관련: PINECONE 검색 + HyperCLOVA 직접 답변
    3. 일상 대화: HyperCLOVA 직접 답변
    """
    try:
        logger.info(f"채팅 요청: {request.query}")
        
        # HyperCLOVA 클라이언트 초기화
        hyperclova = get_hyperclova_client()
        
        # 1. 질문 의도 분류
        intent = hyperclova.classify_intent(request.query)
        logger.info(f"질문 의도: {intent}")
        
        # 2. 일상 대화인 경우 바로 답변
        if intent == 'casual_chat':
            logger.info("일상 대화로 분류 - 직접 답변 생성")
            answer = hyperclova.generate_casual_answer(request.query)
            
            return ChatResponse(
                answer=answer,
                sources=[]
            )
        
        # 3. PINECONE 벡터 검색 (교수님 목록 요청 또는 수업 관련 질문)
        logger.info(f"{intent} 분류 - 벡터 검색 수행")
        
        vectorstore = get_vectorstore_service()
        search_results = vectorstore.similarity_search(
            query=request.query,
            k=request.k
        )
        
        if not search_results:
            return ChatResponse(
                answer="죄송합니다. 관련 수업 정보를 찾을 수 없습니다. 다른 방식으로 질문해주시겠어요?",
                sources=[]
            )
        
        logger.info(f"검색된 문서 수: {len(search_results)}")
        
        # 4. HyperCLOVA가 직접 질문을 이해하고 답변 생성
        answer = hyperclova.generate_answer(
            query=request.query,
            context_docs=search_results
        )
        
        # 응답 구성
        sources = []
        if request.include_sources:
            for result in search_results:
                sources.append({
                    "course_name": result["metadata"].get("course_name", ""),
                    "professor": result["metadata"].get("professor", ""),
                    "section": result["metadata"].get("section", ""),
                    "content_preview": result["page_content"][:200] + "..."
                })
        
        logger.info("답변 생성 완료")
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )

# Lambda 핸들러 생성
def lambda_handler(event, context):
    """Lambda 핸들러 래퍼"""
    try:
        logger.info(f"Lambda handler invoked")
        logger.info(f"Event path: {event.get('path')}")
        logger.info(f"Event method: {event.get('httpMethod')}")
        logger.info(f"Settings ALLOWED_ORIGINS: {settings.ALLOWED_ORIGINS}")
        
        # Mangum 핸들러 호출
        mangum_handler = Mangum(app, lifespan="auto")
        response = mangum_handler(event, context)
        logger.info(f"Lambda handler response: {response}")
        return response
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}", exc_info=True)
        # Mangum 형식에 맞게 반환
        import json
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Internal server error", "detail": str(e)})
        }

if __name__ == "__main__":
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS if settings.ENVIRONMENT == "production" else 1,
        log_level=settings.LOG_LEVEL.lower()
    )
