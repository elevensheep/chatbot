# ...existing code...
"""
애플리케이션 설정 관리
"""
import os
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (backend/.env 또는 프로젝트 루트의 .env)
backend_env = Path(__file__).parent.parent / ".env"
root_env = Path(__file__).parent.parent.parent / ".env"

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
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", JWT_SECRET_KEY)
    
    # CORS 설정 - 환경별 기본값
    DEFAULT_DEV_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
    DEFAULT_PROD_ORIGINS: str = "http://bu-chatbot.co.kr,https://bu-chatbot.co.kr"
    ALLOWED_ORIGINS: List[str] = []
    ALLOW_ALL_ORIGINS_IN_DEV: bool = os.getenv("ALLOW_ALL_ORIGINS_IN_DEV", "false").lower() == "true"
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # MongoDB Atlas 설정
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "chatbot_db")

    # SMTP 이메일 설정
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    
    # 프론트엔드 URL 설정 (이메일 인증 링크용)
    # 기본값은 __init__에서 환경에 따라 설정됨
    FRONTEND_URL: str = ""
    
    # 외부 API 설정 - HyperCLOVA X
    HYPERCLOVA_API_KEY: str = os.getenv("HYPERCLOVA_API_KEY", "")
    HYPERCLOVA_API_GATEWAY_KEY: Optional[str] = os.getenv("HYPERCLOVA_API_GATEWAY_KEY")
    HYPERCLOVA_REQUEST_ID: Optional[str] = os.getenv("HYPERCLOVA_REQUEST_ID")
    
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
        
        # FRONTEND_URL 설정 (다른 용도로 사용될 수 있으므로 기본값은 localhost)
        # 이메일 인증 링크는 email_utils.py에서 production 환경일 때만 도메인 사용
        env_frontend_url = os.getenv("FRONTEND_URL")
        if env_frontend_url:
            self.FRONTEND_URL = env_frontend_url
        else:
            # 기본값은 localhost (이메일 링크는 email_utils.py에서 별도 처리)
            self.FRONTEND_URL = "http://localhost:3000"
    
    # 모니터링 설정
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    ENABLE_HEALTH_CHECK: bool = os.getenv("ENABLE_HEALTH_CHECK", "true").lower() == "true"


# 전역 설정 인스턴스 (Parameter Store 값 포함)
settings = Settings()
# ...existing code...