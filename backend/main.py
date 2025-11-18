from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import os
from config import settings
from direct_pinecone_service import get_vectorstore_service
from hyperclova_client import get_hyperclova_client
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
    root_path=settings.ROOT_PATH,
    openapi_url=(f"{settings.API_PREFIX}{settings.OPENAPI_URL}" if settings.ENABLE_DOCS else None),
    docs_url=(f"{settings.API_PREFIX}{settings.DOCS_URL}" if settings.ENABLE_DOCS else None),
    redoc_url=(f"{settings.API_PREFIX}{settings.REDOC_URL}" if settings.ENABLE_DOCS else None),
)

# CORS middleware 설정 (가장 먼저 추가 - 역순 실행이므로 마지막에 추가)
# 개발 환경 또는 로컬 호스트에서는 모든 오리진 허용
logger.info(f"CORS 설정 - ENVIRONMENT: {settings.ENVIRONMENT}, ALLOWED_ORIGINS: {settings.ALLOWED_ORIGINS}")

# 로컬 개발 환경 감지: localhost 오리진이 있거나 개발 환경인 경우
is_local_dev = (
    settings.ENVIRONMENT == "development" or
    settings.ALLOWED_ORIGINS == ["*"] or
    any("localhost" in origin or "127.0.0.1" in origin for origin in settings.ALLOWED_ORIGINS) or
    os.getenv("ALLOW_LOCALHOST", "false").lower() == "true" or
    os.getenv("ALLOW_LOCALHOST_IN_PROD", "false").lower() == "true"
)

# 예외 핸들러 추가 (에러 로깅 및 CORS 헤더 포함)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """요청 검증 에러 핸들러"""
    logger.error(f"요청 검증 실패: {exc.errors()}, 요청 본문: {await request.body()}")
    origin = request.headers.get("Origin", "*")
    allowed_origin = "*" if (is_local_dev or origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1")) else origin
    
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
        headers={
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Credentials": "true",
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP 예외 핸들러"""
    origin = request.headers.get("Origin", "*")
    allowed_origin = "*" if (is_local_dev or origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1")) else origin
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Credentials": "true",
        }
    )

# 로컬 개발 환경이거나 프로덕션에서 localhost 허용이 켜져 있으면 모든 오리진 허용
if is_local_dev or os.getenv("ALLOW_LOCALHOST_IN_PROD", "false").lower() == "true":
    logger.info("로컬 개발 환경 CORS 설정 적용: 모든 오리진 허용")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
        ],
        expose_headers=["*"],
    )
else:
    # 프로덕션 환경: localhost도 추가 (로컬 테스트용)
    production_origins = settings.ALLOWED_ORIGINS.copy()
    if "http://localhost:3000" not in production_origins:
        production_origins.append("http://localhost:3000")
    if "http://127.0.0.1:3000" not in production_origins:
        production_origins.append("http://127.0.0.1:3000")
    
    logger.info(f"프로덕션 CORS 설정 적용: {production_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=production_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
        ],
        expose_headers=["*"],
    )

# 보안 미들웨어 설정 (CORS 이후에 추가 - 역순 실행이므로 먼저 실행됨)
# 로컬 개발 환경에서는 TrustedHostMiddleware 비활성화
logger.info(f"is_local_dev: {is_local_dev}, ENVIRONMENT: {settings.ENVIRONMENT}")

# 프로덕션 환경에서도 localhost는 항상 허용 (로컬 테스트용)
if settings.ENVIRONMENT == "production" and not is_local_dev:
    # 기본 허용 호스트(프로덕션 도메인 + localhost)
    allowed_hosts = ["*.bu-chatbot.co.kr", "localhost", "127.0.0.1", "0.0.0.0"]
    
    # 환경변수로 추가 허용
    if os.getenv("ALLOW_LOCALHOST_IN_PROD", "false").lower() in ("1", "true", "yes"):
        logger.info(f"프로덕션 환경에서 localhost 허용: {allowed_hosts}")

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )
    logger.info(f"TrustedHostMiddleware 활성화: {allowed_hosts}")
else:
    logger.info("로컬 개발 환경: TrustedHostMiddleware 비활성화")

# OPTIONS 요청을 명시적으로 처리하는 미들웨어 (CORS 미들웨어보다 먼저 실행되도록)
class CORSOptionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            origin = request.headers.get("Origin", "*")
            
            # 로컬 개발 환경이면 모든 오리진 허용
            if is_local_dev or origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1"):
                allowed_origin = "*"
            else:
                # 프로덕션 환경: 요청한 오리진이 허용 목록에 있으면 허용
                production_origins = settings.ALLOWED_ORIGINS + ["http://localhost:3000", "http://127.0.0.1:3000"]
                if origin in production_origins:
                    allowed_origin = origin
                else:
                    allowed_origin = "*"
            
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": allowed_origin,
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD",
                    "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, Origin, Access-Control-Request-Method, Access-Control-Request-Headers",
                    "Access-Control-Max-Age": "3600",
                }
            )
        
        response = await call_next(request)
        return response

# OPTIONS 미들웨어 추가 (가장 먼저 실행되도록 마지막에 추가 - 역순 실행)
app.add_middleware(CORSOptionsMiddleware)

router = APIRouter(prefix=settings.API_PREFIX)

# 라우터 등록
app.include_router(auth.router, prefix=settings.API_PREFIX)

# MongoDB 연결 이벤트
@app.on_event("startup")
async def startup_db_client():
    """앱 시작 시 MongoDB 연결 및 모델 사전 로딩"""
    # MongoDB 연결
    await db_instance.connect_db()
    logger.info("MongoDB Atlas 연결 완료")
    
    # 모델 사전 로딩 (첫 요청 대기 시간 단축)
    try:
        logger.info("모델 사전 로딩 시작...")
        hyperclova = get_hyperclova_client()
        await hyperclova._load_model()
        logger.info("모델 사전 로딩 완료")
    except Exception as e:
        logger.error(f"모델 사전 로딩 실패 (첫 요청 시 로딩됨): {e}")
        # 모델 로딩 실패해도 서버는 계속 실행 (첫 요청 시 재시도)


@router.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": f"{settings.APP_NAME} is running",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

@router.get("/conversations")
async def get_conversations():
    """대화 목록 조회 (임시 구현)"""
    # TODO: 실제 데이터베이스에서 대화 목록 조회
    # 프론트엔드에서 배열을 기대하므로 배열을 반환
    return []

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
        # Hugging Face 모델이 로드되었는지 확인
        hyperclova = get_hyperclova_client()
        if hyperclova._model_loaded:
            hyperclova_status = "healthy"
            health_status["checks"]["hyperclova"] = {
                "status": "healthy",
                "model_name": settings.HF_MODEL_NAME,
                "device": settings.HF_DEVICE
            }
        else:
            # 모델이 아직 로드되지 않았으면 로드 시도
            try:
                await hyperclova._load_model()
                hyperclova_status = "healthy"
                health_status["checks"]["hyperclova"] = {
                    "status": "healthy",
                    "model_name": settings.HF_MODEL_NAME,
                    "device": settings.HF_DEVICE
                }
            except Exception as load_error:
                hyperclova_status = "unhealthy"
                health_status["checks"]["hyperclova"] = {
                    "status": "unhealthy",
                    "error": f"Model loading failed: {str(load_error)}",
                    "model_name": settings.HF_MODEL_NAME
                }
    except Exception as e:
        hyperclova_status = "unhealthy"
        health_status["checks"]["hyperclova"] = {
            "status": "unhealthy",
            "error": str(e),
            "model_name": settings.HF_MODEL_NAME
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


@router.post("/chat", response_model=ChatResponse)
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
        
        # 1. 질문 의도 분류 (비동기)
        intent = await hyperclova.classify_intent(request.query)
        logger.info(f"질문 의도: {intent}")
        
        # 2. 일상 대화인 경우 바로 답변 (비동기)
        if intent == 'casual_chat':
            logger.info("일상 대화로 분류 - 직접 답변 생성")
            answer = await hyperclova.generate_casual_answer(request.query)
            
            return ChatResponse(
                answer=answer,
                sources=[]
            )
        
        # 3. PINECONE 벡터 검색 (비동기)
        logger.info(f"{intent} 분류 - 벡터 검색 수행")
        
        vectorstore = get_vectorstore_service()
        search_results = await vectorstore.similarity_search(
            query=request.query,
            k=request.k
        )
        
        if not search_results:
            return ChatResponse(
                answer="죄송합니다. 관련 수업 정보를 찾을 수 없습니다. 다른 방식으로 질문해주시겠어요?",
                sources=[]
            )
        
        logger.info(f"검색된 문서 수: {len(search_results)}")
        
        # 4. HyperCLOVA가 직접 질문을 이해하고 답변 생성 (비동기)
        answer = await hyperclova.generate_answer(
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
