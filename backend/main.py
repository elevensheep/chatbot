from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import logging
from config import settings

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


# ============== RAG 관련 엔드포인트 ==============

# 요청/응답 모델
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    query: str = Field(..., description="사용자 질문", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(3, description="검색할 문서 개수", ge=1, le=10)
    return_sources: Optional[bool] = Field(False, description="출처 정보 포함 여부")
    max_tokens: Optional[int] = Field(500, description="최대 생성 토큰 수", ge=50, le=2000)
    temperature: Optional[float] = Field(0.7, description="생성 다양성", ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str = Field(..., description="검색 쿼리", min_length=1, max_length=500)
    top_k: Optional[int] = Field(5, description="검색할 문서 개수", ge=1, le=20)


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    answer: str = Field(..., description="AI 답변")
    query: str = Field(..., description="원본 질문")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="참고 문서")


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    results: List[Dict[str, Any]] = Field(..., description="검색 결과")
    query: str = Field(..., description="검색 쿼리")


# RAG 파이프라인 초기화 (지연 로딩)
_rag_pipeline = None
_vector_search = None

def get_vector_search():
    """VectorSearch 인스턴스를 반환합니다 (API 키 불필요)."""
    global _vector_search
    if _vector_search is None:
        try:
            from vector_search import get_vector_search as get_vs
            _vector_search = get_vs()
            logger.info("✅ VectorSearch 초기화 완료")
        except Exception as e:
            logger.error(f"❌ VectorSearch 초기화 실패: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"벡터 검색 시스템 초기화 실패: {str(e)}"
            )
    return _vector_search

def get_rag():
    """RAG 파이프라인 인스턴스를 반환합니다."""
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            from rag_pipeline import get_rag_pipeline
            _rag_pipeline = get_rag_pipeline()
            logger.info("✅ RAG 파이프라인 초기화 완료")
        except Exception as e:
            logger.error(f"❌ RAG 파이프라인 초기화 실패: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG 시스템 초기화 실패: {str(e)}"
            )
    return _rag_pipeline


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    질의응답 엔드포인트 (RAG 기반)
    
    사용자 질문에 대해 수업계획서 정보를 검색하고 HyperCLOVA X로 답변을 생성합니다.
    """
    try:
        rag = get_rag()
        
        # RAG 파이프라인 실행
        result = rag.answer(
            query=request.query,
            top_k=request.top_k,
            return_sources=request.return_sources,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"❌ 채팅 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    벡터 검색 엔드포인트
    
    질문과 유사한 수업계획서 정보를 검색합니다.
    (HyperCLOVA API 키 불필요)
    """
    try:
        # VectorSearch만 사용 (API 키 불필요)
        vs = get_vector_search()
        
        # 벡터 검색만 수행
        results = vs.search(request.query, top_k=request.top_k)
        
        return SearchResponse(
            results=results,
            query=request.query
        )
    
    except Exception as e:
        logger.error(f"❌ 검색 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}"
        )


@app.get("/rag/status")
async def rag_status():
    """RAG 시스템 상태 확인"""
    try:
        rag = get_rag()
        return {
            "status": "ready",
            "message": "RAG 시스템이 정상 작동 중입니다.",
            "vector_db_loaded": rag.vector_search is not None,
            "llm_loaded": rag.hyperclova_client is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "vector_db_loaded": False,
            "llm_loaded": False
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
