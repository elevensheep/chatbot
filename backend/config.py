"""
애플리케이션 설정 관리
"""
import os
import boto3
from typing import Optional
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


def get_parameter_store_value(parameter_name: str, default_value: str = "") -> str:
    """
    AWS Systems Manager Parameter Store에서 값을 가져옵니다.
    
    Args:
        parameter_name: Parameter Store의 파라미터 이름
        default_value: 값을 가져올 수 없을 때 사용할 기본값
        
    Returns:
        Parameter Store에서 가져온 값 또는 기본값
    """
    try:
        # Lambda 환경에서는 boto3 클라이언트를 자동으로 생성
        ssm_client = boto3.client('ssm')
        
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=True  # SecureString 파라미터의 경우 복호화
        )
        
        return response['Parameter']['Value']
    except Exception as e:
        print(f"Warning: Could not retrieve parameter {parameter_name}: {e}")
        return default_value


class Settings:
    """애플리케이션 설정 클래스"""
    
    # 기본 설정
    APP_NAME: str = "Chatbot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # 서버 설정
    HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BACKEND_PORT", "5000"))
    WORKERS: int = int(os.getenv("BACKEND_WORKERS", "1"))
    
    # 보안 설정
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    
    # CORS 설정 - 기본 개발 origins
    DEFAULT_ORIGINS: str = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS)  # 환경변수로 우선 설정 가능
    
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

    # 외부 API 설정 - HyperCLOVA X
    HYPERCLOVA_API_KEY: str = os.getenv("HYPERCLOVA_API_KEY", "")
    HYPERCLOVA_API_GATEWAY_KEY: Optional[str] = os.getenv("HYPERCLOVA_API_GATEWAY_KEY")
    HYPERCLOVA_REQUEST_ID: Optional[str] = os.getenv("HYPERCLOVA_REQUEST_ID")
    
    # PINECONE 벡터 스토어 설정
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "chatbot-courses")
    
    def __init__(self):
        """설정 초기화 시 Parameter Store에서 값 가져오기"""
        try:
            # Parameter Store 파라미터 이름들
            pinecone_api_key_param = os.getenv("PINECONE_API_KEY_PARAM")
            pinecone_index_name_param = os.getenv("PINECONE_INDEX_NAME_PARAM")
            hyperclova_api_key_param = os.getenv("HYPERCLOVA_API_KEY_PARAM")
            
            # Parameter Store에서 값 가져오기 (환경변수가 없으면 기본값 사용)
            if pinecone_api_key_param:
                try:
                    self.PINECONE_API_KEY = get_parameter_store_value(
                        pinecone_api_key_param, 
                        self.PINECONE_API_KEY
                    )
                except Exception as e:
                    print(f"Warning: Could not get PINECONE_API_KEY from Parameter Store: {e}")
            
            if pinecone_index_name_param:
                try:
                    self.PINECONE_INDEX_NAME = get_parameter_store_value(
                        pinecone_index_name_param, 
                        self.PINECONE_INDEX_NAME
                    )
                except Exception as e:
                    print(f"Warning: Could not get PINECONE_INDEX_NAME from Parameter Store: {e}")
            
            if hyperclova_api_key_param:
                try:
                    self.HYPERCLOVA_API_KEY = get_parameter_store_value(
                        hyperclova_api_key_param, 
                        self.HYPERCLOVA_API_KEY
                    )
                except Exception as e:
                    print(f"Warning: Could not get HYPERCLOVA_API_KEY from Parameter Store: {e}")
            
            # ALLOWED_ORIGINS를 리스트로 변환
            if isinstance(self.ALLOWED_ORIGINS, str):
                self.ALLOWED_ORIGINS = [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
                
        except Exception as e:
            print(f"Warning: Settings initialization error: {e}")
            # 기본값 사용 (문자열을 리스트로 변환)
            if isinstance(self.ALLOWED_ORIGINS, str):
                self.ALLOWED_ORIGINS = self.ALLOWED_ORIGINS.split(",")
    
    # 모니터링 설정
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    ENABLE_HEALTH_CHECK: bool = os.getenv("ENABLE_HEALTH_CHECK", "true").lower() == "true"


# 전역 설정 인스턴스 (Parameter Store 값 포함)
settings = Settings()
