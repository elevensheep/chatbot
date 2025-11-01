"""
사용자 관련 Pydantic 모델
"""
from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict
from typing import Optional
from datetime import datetime
import re


class UserSignup(BaseModel):
    """회원가입 요청 모델"""
    student_id: str = Field(..., min_length=8, max_length=10, description="학번")
    name: str = Field(..., min_length=2, max_length=20, description="이름")
    email: EmailStr = Field(..., description="이메일")
    password: str = Field(..., min_length=8, description="비밀번호")
    department: str = Field(..., min_length=2, max_length=50, description="학과")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": "202412345",
                "name": "홍길동",
                "email": "student@example.com",
                "password": "SecurePassword123!",
                "department": "컴퓨터공학과"
            }
        }
    )

    @field_validator('student_id')
    @classmethod
    def validate_student_id(cls, v):
        """학번 유효성 검증 (숫자만 허용)"""
        if not v.isdigit():
            raise ValueError('학번은 숫자만 입력 가능합니다')
        if len(v) < 8 or len(v) > 10:
            raise ValueError('학번은 8-10자리여야 합니다')
        return v

    @field_validator('email')
    @classmethod
    def validate_email_domain(cls, v):
        """이메일 도메인 검증 (@bu.ac.kr만 허용)"""
        if not v.endswith('@bu.ac.kr'):
            raise ValueError('백석대학교 이메일(@bu.ac.kr)만 사용 가능합니다')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """비밀번호 유효성 검증 (영문+숫자+특수문자)"""
        # 영문 포함 확인
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError('비밀번호는 영문을 포함해야 합니다')

        # 숫자 포함 확인
        if not re.search(r'\d', v):
            raise ValueError('비밀번호는 숫자를 포함해야 합니다')

        # 특수문자 포함 확인
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('비밀번호는 특수문자를 포함해야 합니다')

        return v


class UserLogin(BaseModel):
    """로그인 요청 모델"""
    student_id: str = Field(..., description="학번")
    password: str = Field(..., description="비밀번호")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": "202412345",
                "password": "SecurePassword123!"
            }
        }
    )


class UserResponse(BaseModel):
    """사용자 응답 모델 (비밀번호 제외)"""
    user_id: str = Field(..., description="사용자 ID")
    student_id: str = Field(..., description="학번")
    name: str = Field(..., description="이름")
    email: str = Field(..., description="이메일")
    department: str = Field(..., description="학과")
    created_at: datetime = Field(..., description="생성 시간")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "student_id": "202412345",
                "name": "홍길동",
                "email": "student@example.com",
                "department": "컴퓨터공학과",
                "created_at": "2025-10-13T12:00:00Z"
            }
        }
    )


class EmailVerificationRequest(BaseModel):
    """이메일 인증 요청 모델"""
    email: EmailStr = Field(..., description="인증할 이메일")

    @field_validator('email')
    @classmethod
    def validate_email_domain(cls, v):
        """이메일 도메인 검증 (@bu.ac.kr만 허용)"""
        if not v.endswith('@bu.ac.kr'):
            raise ValueError('백석대학교 이메일(@bu.ac.kr)만 사용 가능합니다')
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "student@bu.ac.kr"
            }
        }
    )


class EmailVerificationConfirm(BaseModel):
    """이메일 인증 확인 모델"""
    token: str = Field(..., description="인증 토큰")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "abc123xyz..."
            }
        }
    )


class UserInDB(BaseModel):
    """데이터베이스에 저장되는 사용자 모델"""
    student_id: str
    name: str
    email: str
    password_hash: str
    department: str
    role: str = "user"  # user 또는 admin
    email_verified: bool = False  # 이메일 인증 여부
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": "202412345",
                "name": "홍길동",
                "email": "student@example.com",
                "password_hash": "$2b$12$...",
                "department": "컴퓨터공학과",
                "role": "user",
                "created_at": "2025-10-13T12:00:00Z",
                "updated_at": "2025-10-13T12:00:00Z"
            }
        }
    )
