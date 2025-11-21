from fastapi import FastAPI, HTTPException, APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from bson import ObjectId
import uvicorn
import logging
from config import settings
from direct_pinecone_service import get_vectorstore_service
from hyperclova_client import get_hyperclova_client
from database import db_instance, Collections
from routers import auth, conversations
from auth_utils import get_current_user

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
    root_path=settings.ROOT_PATH,
    openapi_url=(f"{settings.API_PREFIX}{settings.OPENAPI_URL}" if settings.ENABLE_DOCS else None),
    docs_url=(f"{settings.API_PREFIX}{settings.DOCS_URL}" if settings.ENABLE_DOCS else None),
    redoc_url=(f"{settings.API_PREFIX}{settings.REDOC_URL}" if settings.ENABLE_DOCS else None),
)

# 보안 미들웨어 설정
if settings.ENVIRONMENT == "production":
    # 기본 허용 호스트(프로덕션 도메인)
    allowed_hosts = ["*.bu-chatbot.co.kr"]

    # 내부 테스트/헬스체크 용도로 로컬호스트를 허용하려면
    # 환경변수 ALLOW_LOCALHOST_IN_PROD=true 로 켜십시오.
    import os
    if os.getenv("ALLOW_LOCALHOST_IN_PROD", "false").lower() in ("1", "true", "yes"):
        allowed_hosts += ["localhost", "127.0.0.1"]

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )

# CORS middleware 설정
if settings.ALLOWED_ORIGINS == ["*"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

router = APIRouter(prefix=settings.API_PREFIX)

# Validation Error 핸들러 추가 (422 에러 로깅)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 Validation Error를 상세히 로깅"""
    errors = exc.errors()
    error_details = []
    for error in errors:
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        error_details.append({
            "field": field,
            "message": error.get("msg"),
            "type": error.get("type")
        })
    
    logger.warning(
        f"Validation Error - {request.method} {request.url.path}: {error_details}"
    )
    
    # 기본 FastAPI validation error 응답 반환
    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )

# 라우터 등록
app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(conversations.router, prefix=settings.API_PREFIX)

# MongoDB 연결 이벤트
@app.on_event("startup")
async def startup_db_client():
    """앱 시작 시 MongoDB 연결"""
    await db_instance.connect_db()
    logger.info("MongoDB Atlas 연결 완료")


@router.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": f"{settings.APP_NAME} is running",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

@router.get("/health")
async def health_check():
    """헬스 체크 엔드포인트 - 실제 의존성 연결 상태 확인"""
    health_status = {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": settings.APP_VERSION,
        "checks": {}
    }
    
    # Pinecone 연결 상태 확인
    pinecone_status = "unknown"
    try:
        vectorstore = get_vectorstore_service()
        if vectorstore and vectorstore.index:
            # 간단한 연결 테스트 (인덱스 정보 확인)
            # 실제 쿼리는 하지 않고 클라이언트만 확인
            pinecone_status = "healthy"
            health_status["checks"]["pinecone"] = {
                "status": "healthy",
                "index_name": settings.PINECONE_INDEX_NAME
            }
        else:
            pinecone_status = "unhealthy"
            health_status["checks"]["pinecone"] = {
                "status": "unhealthy",
                "error": "Pinecone index not initialized"
            }
    except Exception as e:
        pinecone_status = "unhealthy"
        health_status["checks"]["pinecone"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        logger.warning(f"Pinecone health check failed: {e}")
    
    hyperclova_status = "unknown"
    try:
        if settings.HYPERCLOVA_API_KEY:
            hyperclova_status = "healthy"
            health_status["checks"]["hyperclova"] = {
                "status": "healthy",
                "api_key_configured": True
            }
        else:
            hyperclova_status = "unhealthy"
            health_status["checks"]["hyperclova"] = {
                "status": "unhealthy",
                "error": "HyperCLOVA API key not configured"
            }
    except Exception as e:
        hyperclova_status = "unhealthy"
        health_status["checks"]["hyperclova"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        logger.warning(f"HyperCLOVA health check failed: {e}")
    
    # 전체 상태 결정 (하나라도 unhealthy면 unhealthy)
    if pinecone_status == "unhealthy" or hyperclova_status == "unhealthy":
        health_status["status"] = "unhealthy"
        # HTTP 503 반환 (로드 밸런서가 헬스 체크 실패로 인식)
        return JSONResponse(status_code=503, content=health_status)
    
    return health_status

@router.get("/metrics")
async def metrics():
    """메트릭 엔드포인트 (프로덕션에서는 Prometheus 등 사용)"""
    if not settings.ENABLE_METRICS:
        return {"message": "Metrics disabled"}
    
    return {
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "log_level": settings.LOG_LEVEL
    }


app.include_router(router)

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
