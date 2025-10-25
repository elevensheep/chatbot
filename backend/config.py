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
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000").split(",")
    
    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # 데이터베이스 설정
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # 외부 API 설정 - HyperCLOVA X
    HYPERCLOVA_API_KEY: str = os.getenv("HYPERCLOVA_API_KEY", "")
    HYPERCLOVA_API_GATEWAY_KEY: Optional[str] = os.getenv("HYPERCLOVA_API_GATEWAY_KEY")
    HYPERCLOVA_REQUEST_ID: Optional[str] = os.getenv("HYPERCLOVA_REQUEST_ID")
    
    # PINECONE 벡터 스토어 설정
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "chatbot-courses")
    
    def __init__(self):
        """설정 초기화 시 Parameter Store에서 값 가져오기"""
        # Parameter Store 파라미터 이름들
        pinecone_api_key_param = os.getenv("PINECONE_API_KEY_PARAM")
        pinecone_index_name_param = os.getenv("PINECONE_INDEX_NAME_PARAM")
        hyperclova_api_key_param = os.getenv("HYPERCLOVA_API_KEY_PARAM")
        
        # Parameter Store에서 값 가져오기 (환경변수가 없으면 기본값 사용)
        if pinecone_api_key_param:
            self.PINECONE_API_KEY = get_parameter_store_value(
                pinecone_api_key_param, 
                self.PINECONE_API_KEY
            )
        
        if pinecone_index_name_param:
            self.PINECONE_INDEX_NAME = get_parameter_store_value(
                pinecone_index_name_param, 
                self.PINECONE_INDEX_NAME
            )
        
        if hyperclova_api_key_param:
            self.HYPERCLOVA_API_KEY = get_parameter_store_value(
                hyperclova_api_key_param, 
                self.HYPERCLOVA_API_KEY
            )
    
    # 모니터링 설정
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    ENABLE_HEALTH_CHECK: bool = os.getenv("ENABLE_HEALTH_CHECK", "true").lower() == "true"


# 전역 설정 인스턴스 (Parameter Store 값 포함)
settings = Settings()
