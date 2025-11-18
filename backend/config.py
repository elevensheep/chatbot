# ...existing code...
"""
애플리케이션 설정 관리
"""
import os
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (backend/.env 또는 프로젝트 루트의 .env)
backend_env = Path(__file__).parent / ".env"
root_env = Path(__file__).parent.parent / ".env"

if backend_env.exists():
    try:
        load_dotenv(backend_env, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            load_dotenv(backend_env, encoding='utf-16')
        except:
            pass  # .env 파일을 읽을 수 없으면 환경변수에서 직접 읽음
elif root_env.exists():
    try:
        load_dotenv(root_env, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            load_dotenv(root_env, encoding='utf-16')
        except:
            pass  # .env 파일을 읽을 수 없으면 환경변수에서 직접 읽음



class Settings:
    """애플리케이션 설정 클래스"""
    
    # 기본 설정
    APP_NAME: str = "Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    # API 경로 프리픽스 및 프록시 루트 경로
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")
    ROOT_PATH: str = os.getenv("ROOT_PATH", "")
    # 문서화(Swagger / ReDoc) 설정
    ENABLE_DOCS: bool = os.getenv("ENABLE_DOCS", "true").lower() == "true"
    OPENAPI_URL: str = os.getenv("OPENAPI_URL", "/openapi.json")
    DOCS_URL: str = os.getenv("DOCS_URL", "/docs")
    REDOC_URL: str = os.getenv("REDOC_URL", "/redoc")
    # 원격 주입 제거(ECS는 Task env/Secrets 사용)
    
    # 서버 설정
    # 우선순위: PORT/HOST → BACKEND_PORT/BACKEND_HOST → 기본값
    HOST: str = os.getenv("HOST", os.getenv("BACKEND_HOST", "0.0.0.0"))
    PORT: int = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "5000")))
    WORKERS: int = int(os.getenv("BACKEND_WORKERS", "1"))
    
    # 보안 설정
    # JWT 설정과 호환: JWT_SECRET_KEY가 있으면 SECRET_KEY 기본값으로 사용
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-key-change-me")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", JWT_SECRET_KEY)
    
    # CORS 설정 - 환경별 기본값
    DEFAULT_DEV_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
    DEFAULT_PROD_ORIGINS: str = "https://bu-chatbot.co.kr"
    ALLOWED_ORIGINS: List[str] = []
    ALLOW_ALL_ORIGINS_IN_DEV: bool = os.getenv("ALLOW_ALL_ORIGINS_IN_DEV", "false").lower() == "true"
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
   # MongoDB Atlas 설정
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "chatbot_db")

    # JWT 인증 설정
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24시간

    # SMTP 이메일 설정
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    
    # 외부 API 설정 - HyperCLOVA X (레거시, 사용 안 함)
    HYPERCLOVA_API_KEY: str = os.getenv("HYPERCLOVA_API_KEY", "")
    HYPERCLOVA_API_GATEWAY_KEY: Optional[str] = os.getenv("HYPERCLOVA_API_GATEWAY_KEY")
    HYPERCLOVA_REQUEST_ID: Optional[str] = os.getenv("HYPERCLOVA_REQUEST_ID")
    
    # Hugging Face 모델 설정 (HyperCLOVAX SEED)
    HF_MODEL_NAME: str = os.getenv("SEED_MODEL_PATH", os.getenv("HF_MODEL_NAME", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"))
    HF_DEVICE: str = os.getenv("SEED_DEVICE", os.getenv("HF_DEVICE", "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"))
    HF_USE_FP16: bool = os.getenv("HF_USE_FP16", "false").lower() == "true"
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # Hugging Face 토큰 (private 모델 접근용)
    HF_LOAD_IN_8BIT: bool = os.getenv("SEED_LOAD_IN_8BIT", os.getenv("HF_LOAD_IN_8BIT", "false")).lower() == "true"
    HF_LOAD_IN_4BIT: bool = os.getenv("SEED_LOAD_IN_4BIT", os.getenv("HF_LOAD_IN_4BIT", "false")).lower() == "true"
    
    # PINECONE 벡터 스토어 설정
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "chatbot-courses")
    
    def __init__(self):
        """설정 초기화"""
        # ALLOWED_ORIGINS를 환경 및 설정에 따라 리스트로 변환
        if self.ENVIRONMENT == "production":
            raw = os.getenv("ALLOWED_ORIGINS", self.DEFAULT_PROD_ORIGINS)
        else:
            raw = os.getenv("ALLOWED_ORIGINS", self.DEFAULT_DEV_ORIGINS)
        
        if isinstance(raw, str):
            self.ALLOWED_ORIGINS = [origin.strip() for origin in raw.split(",") if origin.strip()]
        
        # 개발에서 모든 오리진 허용 옵션 처리
        if not self.ENVIRONMENT == "production" and self.ALLOW_ALL_ORIGINS_IN_DEV:
            self.ALLOWED_ORIGINS = ["*"]
    
    # 모니터링 설정
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    ENABLE_HEALTH_CHECK: bool = os.getenv("ENABLE_HEALTH_CHECK", "true").lower() == "true"


# 전역 설정 인스턴스 (Parameter Store 값 포함)
settings = Settings()
# ...existing code...