"""
JWT 인증 및 비밀번호 해싱 유틸리티
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from config import settings
import logging
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

security = HTTPBearer()


def hash_password(password: str) -> str:
    """
    비밀번호 해싱 (bcrypt 직접 사용)

    Args:
        password: 평문 비밀번호

    Returns:
        해시된 비밀번호
    """
    # bcrypt는 72바이트 제한이 있으므로 자동으로 잘라줍니다
    password_bytes = password.encode('utf-8')

    # bcrypt는 72바이트까지만 처리 가능
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]

    # salt 생성 및 해싱
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)

    # bytes를 string으로 변환
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    비밀번호 검증 (bcrypt 직접 사용)

    Args:
        plain_password: 평문 비밀번호
        hashed_password: 해시된 비밀번호

    Returns:
        비밀번호 일치 여부
    """
    password_bytes = plain_password.encode('utf-8')

    # bcrypt는 72바이트까지만 처리 가능
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]

    hashed_bytes = hashed_password.encode('utf-8')

    try:
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"비밀번호 검증 실패: {e}")
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    JWT Access Token 생성

    Args:
        data: 토큰에 포함할 데이터 (예: {"sub": "user_id"})
        expires_delta: 만료 시간 (Optional)

    Returns:
        JWT 토큰 문자열
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    JWT Access Token 디코딩

    Args:
        token: JWT 토큰 문자열

    Returns:
        디코딩된 페이로드 딕셔너리 또는 None (실패 시)
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.error(f"JWT 디코딩 오류: {e}")
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    JWT 토큰에서 현재 사용자 ID 추출

    Args:
        credentials: HTTP Authorization 헤더

    Returns:
        사용자 ID (user_id)

    Raises:
        HTTPException: 토큰이 유효하지 않거나 만료된 경우
    """
    token = credentials.credentials
    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않거나 만료된 토큰입니다"
        )

    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="토큰에 사용자 정보가 없습니다"
        )

    return user_id
