"""
대화방 및 메시지 관련 Pydantic 모델
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


class ConversationCreate(BaseModel):
    """대화방 생성 요청 모델"""
    title: Optional[str] = Field(default="새 대화", description="대화방 제목")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "데이터베이스 평가 방법"
            }
        }
    )


class ConversationResponse(BaseModel):
    """대화방 응답 모델"""
    id: str = Field(..., description="대화방 ID")
    user_id: str = Field(..., description="소유자 ID")
    title: str = Field(..., description="대화방 제목")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="마지막 업데이트 시간")
    message_count: int = Field(default=0, description="메시지 개수")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "user_id": "507f191e810c19729de860ea",
                "title": "데이터베이스 평가 방법",
                "created_at": "2025-11-14T12:00:00Z",
                "updated_at": "2025-11-14T12:05:00Z",
                "message_count": 4
            }
        }
    )


class ConversationUpdate(BaseModel):
    """대화방 제목 수정 요청 모델"""
    title: str = Field(..., min_length=1, max_length=100, description="새 제목")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "데이터베이스 평가 및 과제 일정"
            }
        }
    )


class MessageCreate(BaseModel):
    """메시지 생성 요청 모델 (내부 사용)"""
    conversation_id: str = Field(..., description="대화방 ID")
    role: str = Field(..., description="메시지 역할 (user | assistant)")
    content: str = Field(..., description="메시지 내용")
    sources: List[dict] = Field(default=[], description="출처 (assistant만)")
    order: int = Field(..., description="메시지 순서")


class MessageResponse(BaseModel):
    """메시지 응답 모델"""
    id: str = Field(..., description="메시지 ID")
    conversation_id: str = Field(..., description="대화방 ID")
    role: str = Field(..., description="메시지 역할 (user | assistant)")
    content: str = Field(..., description="메시지 내용")
    sources: List[dict] = Field(default=[], description="출처 정보")
    created_at: datetime = Field(..., description="생성 시간")
    order: int = Field(..., description="메시지 순서")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "conversation_id": "507f191e810c19729de860ea",
                "role": "user",
                "content": "데이터베이스 평가 방법이 뭐야?",
                "sources": [],
                "created_at": "2025-11-14T12:00:00Z",
                "order": 0
            }
        }
    )


class ConversationInDB(BaseModel):
    """데이터베이스에 저장되는 대화방 모델"""
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "507f191e810c19729de860ea",
                "title": "새 대화",
                "created_at": "2025-11-14T12:00:00Z",
                "updated_at": "2025-11-14T12:00:00Z",
                "message_count": 0
            }
        }
    )


class MessageInDB(BaseModel):
    """데이터베이스에 저장되는 메시지 모델"""
    conversation_id: str
    role: str
    content: str
    sources: List[dict] = []
    created_at: datetime
    order: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_id": "507f191e810c19729de860ea",
                "role": "user",
                "content": "데이터베이스 평가 방법이 뭐야?",
                "sources": [],
                "created_at": "2025-11-14T12:00:00Z",
                "order": 0
            }
        }
    )
