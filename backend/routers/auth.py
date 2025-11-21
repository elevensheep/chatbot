"""
인증 관련 API 라우터
"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timedelta
from bson import ObjectId
import logging

from models.user import (
    UserSignup, UserLogin, UserResponse,
    EmailVerificationRequest, EmailVerificationConfirm,
    PasswordResetRequest, PasswordResetConfirm
)
from database import Collections, db_instance
from auth_utils import hash_password, verify_password, create_access_token
from config import settings
from email_utils import generate_verification_token, generate_password_reset_code, save_verification_token, verify_token, send_verification_email

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post(
    "/send-verification-email",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="이메일 인증 요청",
    description="회원가입 전 이메일 인증 코드를 발송합니다."
)
async def send_verification_email_endpoint(request: EmailVerificationRequest):
    """
    이메일 인증 메일 발송 엔드포인트

    - **email**: 백석대학교 이메일 (@bu.ac.kr)
    """
    try:
        # 1. 이미 인증된 이메일인지 확인
        users_collection = db_instance.get_collection(Collections.USERS)
        existing_user = await users_collection.find_one({"email": request.email})

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "success": False,
                    "error": {
                        "code": "CONFLICT",
                        "message": "이미 등록된 이메일입니다",
                        "details": f"이메일 '{request.email}'은(는) 이미 사용 중입니다"
                    }
                }
            )

        # 2. 인증 토큰 생성
        token = generate_verification_token()

        # 3. 토큰 저장
        await save_verification_token(request.email, token, expires_minutes=30)

        # 4. 이메일 발송
        await send_verification_email(request.email, token)

        logger.info(f"이메일 인증 요청: {request.email}")

        return {
            "success": True,
            "message": "인증 이메일이 발송되었습니다. 이메일을 확인해주세요.",
            "data": {
                "email": request.email,
                "expires_in_minutes": 30
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"이메일 인증 요청 처리 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "이메일 발송 중 오류가 발생했습니다",
                    "details": str(e)
                }
            }
        )


@router.post(
    "/verify-email",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="이메일 인증 확인",
    description="이메일로 받은 인증 토큰을 확인합니다."
)
async def verify_email_endpoint(request: EmailVerificationConfirm):
    """
    이메일 인증 확인 엔드포인트

    - **token**: 이메일로 받은 인증 토큰
    """
    try:
        # 1. 토큰 검증
        email = await verify_token(request.token)

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": {
                        "code": "INVALID_TOKEN",
                        "message": "유효하지 않거나 만료된 인증 토큰입니다",
                        "details": "인증 링크를 다시 확인하거나 새로운 인증 이메일을 요청해주세요"
                    }
                }
            )

        logger.info(f"이메일 인증 완료: {email}")

        return {
            "success": True,
            "message": "이메일 인증이 완료되었습니다. 회원가입을 진행해주세요.",
            "data": {
                "email": email,
                "verified": True
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"이메일 인증 확인 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "이메일 인증 처리 중 오류가 발생했습니다",
                    "details": str(e)
                }
            }
        )


@router.post(
    "/signup",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="회원가입",
    description="새로운 사용자를 등록합니다."
)
async def signup(user_data: UserSignup):
    """
    회원가입 엔드포인트

    - **email**: 이메일 (@bu.ac.kr)
    - **password**: 비밀번호 (최소 8자, 영문+숫자+특수문자)
    """
    try:
        # 1. 데이터베이스 연결
        users_collection = db_instance.get_collection(Collections.USERS)
        verification_collection = db_instance.get_collection(Collections.EMAIL_VERIFICATIONS)

        # 1.5. 이메일 인증 여부 확인
        verification = await verification_collection.find_one({
            "email": user_data.email,
            "verified": True
        })

        if not verification:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": {
                        "code": "EMAIL_NOT_VERIFIED",
                        "message": "이메일 인증이 필요합니다",
                        "details": "회원가입 전에 이메일 인증을 완료해주세요"
                    }
                }
            )

        # 2. 중복 확인 (이메일)
        existing_email = await users_collection.find_one({"email": user_data.email})
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "success": False,
                    "error": {
                        "code": "CONFLICT",
                        "message": "이미 존재하는 이메일입니다",
                        "details": f"이메일 '{user_data.email}'은(는) 이미 등록되어 있습니다"
                    }
                }
            )

        # 3. 비밀번호 해싱
        password_hash = hash_password(user_data.password)

        # 4. 사용자 문서 생성
        now = datetime.utcnow()
        user_doc = {
            "email": user_data.email,
            "password_hash": password_hash,
            "role": "user",
            "email_verified": True,  # 이메일 인증 완료 상태
            "created_at": now,
            "updated_at": now
        }

        # 5. MongoDB에 삽입
        result = await users_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)

        logger.info(f"새로운 사용자 등록: {user_data.email}")

        # 6. 응답 생성 (API 명세서 형식)
        return {
            "success": True,
            "data": {
                "user_id": user_id,
                "email": user_data.email,
                "created_at": now.isoformat() + "Z"
            },
            "message": "회원가입이 완료되었습니다"
        }

    except HTTPException:
        # HTTPException은 그대로 전달
        raise

    except Exception as e:
        logger.error(f"회원가입 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "서버 내부 오류가 발생했습니다",
                    "details": str(e)
                }
            }
        )


@router.post(
    "/login",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="로그인",
    description="사용자 인증 후 JWT 토큰을 발급합니다."
)
async def login(credentials: UserLogin):
    """
    로그인 엔드포인트

    - **email**: 이메일
    - **password**: 비밀번호

    **Returns**: JWT Access Token 및 사용자 정보
    """
    try:
        # 1. 데이터베이스 연결
        users_collection = db_instance.get_collection(Collections.USERS)

        # 2. 이메일로 사용자 조회
        user = await users_collection.find_one({"email": credentials.email})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "success": False,
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "이메일 또는 비밀번호가 올바르지 않습니다",
                        "details": "입력하신 정보를 다시 확인해주세요"
                    }
                }
            )

        # 3. 비밀번호 검증
        if not verify_password(credentials.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "success": False,
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "이메일 또는 비밀번호가 올바르지 않습니다",
                        "details": "입력하신 정보를 다시 확인해주세요"
                    }
                }
            )

        # 4. JWT 토큰 생성
        access_token = create_access_token(
            data={"sub": str(user["_id"]), "email": user["email"]}
        )

        logger.info(f"사용자 로그인 성공: {credentials.email}")

        # 5. 응답 생성 (API 명세서 형식)
        return {
            "success": True,
            "data": {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # 초 단위
                "user": {
                    "user_id": str(user["_id"]),
                    "email": user["email"]
                }
            },
            "message": "로그인 성공"
        }

    except HTTPException:
        # HTTPException은 그대로 전달
        raise

    except Exception as e:
        logger.error(f"로그인 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "서버 내부 오류가 발생했습니다",
                    "details": str(e)
                }
            }
        )


@router.post(
    "/password-reset/request",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="비밀번호 재설정 요청",
    description="등록된 이메일로 비밀번호 재설정 인증 코드를 발송합니다."
)
async def request_password_reset(request: PasswordResetRequest):
    """
    비밀번호 재설정 요청 엔드포인트

    - **email**: 등록된 이메일 (@bu.ac.kr)
    """
    try:
        users_collection = db_instance.get_collection(Collections.USERS)

        # 1. 이메일이 등록되어 있는지 확인
        user = await users_collection.find_one({"email": request.email})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "success": False,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "등록되지 않은 이메일입니다",
                        "details": f"이메일 '{request.email}'로 등록된 계정을 찾을 수 없습니다"
                    }
                }
            )

        # 2. 인증 코드 생성 (6자리 숫자)
        code = generate_password_reset_code()

        # 3. 코드 저장 (email_verifications 컬렉션 재사용, type 구분)
        verification_collection = db_instance.get_collection(Collections.EMAIL_VERIFICATIONS)

        # 기존 비밀번호 재설정 코드 삭제
        await verification_collection.delete_many({
            "email": request.email,
            "type": "password_reset"
        })

        # 새 코드 저장
        verification_doc = {
            "email": request.email,
            "token": code,  # 6자리 숫자 코드
            "type": "password_reset",
            "verified": False,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(minutes=30)
        }
        await verification_collection.insert_one(verification_doc)

        # 4. 이메일 발송
        await send_verification_email(request.email, code, is_password_reset=True)

        logger.info(f"비밀번호 재설정 요청: {request.email}")

        return {
            "success": True,
            "message": "비밀번호 재설정 인증 코드가 이메일로 발송되었습니다.",
            "data": {
                "email": request.email,
                "expires_in_minutes": 30
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"비밀번호 재설정 요청 처리 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "비밀번호 재설정 요청 중 오류가 발생했습니다",
                    "details": str(e)
                }
            }
        )


@router.post(
    "/password-reset/confirm",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="비밀번호 재설정 확인",
    description="인증 코드를 확인하고 새 비밀번호로 변경합니다."
)
async def confirm_password_reset(request: PasswordResetConfirm):
    """
    비밀번호 재설정 확인 엔드포인트

    - **email**: 이메일
    - **token**: 이메일로 받은 인증 코드
    - **new_password**: 새 비밀번호
    """
    try:
        users_collection = db_instance.get_collection(Collections.USERS)
        verification_collection = db_instance.get_collection(Collections.EMAIL_VERIFICATIONS)

        # 1. 토큰 검증
        verification = await verification_collection.find_one({
            "email": request.email,
            "token": request.token,
            "type": "password_reset"
        })

        if not verification:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": {
                        "code": "INVALID_TOKEN",
                        "message": "유효하지 않은 인증 코드입니다",
                        "details": "인증 코드를 다시 확인하거나 새로운 요청을 해주세요"
                    }
                }
            )

        # 2. 만료 확인
        if verification["expires_at"] < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": {
                        "code": "EXPIRED_TOKEN",
                        "message": "인증 코드가 만료되었습니다",
                        "details": "새로운 비밀번호 재설정 요청을 해주세요"
                    }
                }
            )

        # 3. 비밀번호 변경
        new_password_hash = hash_password(request.new_password)

        result = await users_collection.update_one(
            {"email": request.email},
            {
                "$set": {
                    "password_hash": new_password_hash,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "success": False,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "사용자를 찾을 수 없습니다",
                        "details": "계정 정보를 확인해주세요"
                    }
                }
            )

        # 4. 사용된 토큰 삭제
        await verification_collection.delete_one({"_id": verification["_id"]})

        logger.info(f"비밀번호 재설정 완료: {request.email}")

        return {
            "success": True,
            "message": "비밀번호가 성공적으로 변경되었습니다. 새 비밀번호로 로그인해주세요.",
            "data": {
                "email": request.email
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"비밀번호 재설정 확인 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "비밀번호 재설정 처리 중 오류가 발생했습니다",
                    "details": str(e)
                }
            }
        )
