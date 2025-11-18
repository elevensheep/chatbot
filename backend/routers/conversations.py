"""
대화방 관리 API 라우터
"""
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime
from bson import ObjectId
from typing import List, Optional
from pydantic import BaseModel
import logging

from models.conversation import (
    ConversationCreate,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse
)
from database import Collections, db_instance
from auth_utils import get_current_user
from direct_pinecone_service import get_vectorstore_service
from hyperclova_client import get_hyperclova_client
from services.title_generator import auto_generate_title

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="새 대화방 생성",
    description="새로운 대화방을 생성합니다."
)
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user_id: str = Depends(get_current_user)
):
    """
    새 대화방 생성

    - **title**: 대화방 제목 (기본값: "새 대화")
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        # 대화방 문서 생성
        now = datetime.utcnow()
        conversation_doc = {
            "user_id": current_user_id,
            "title": conversation_data.title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0
        }

        # MongoDB에 삽입
        result = await conversations_collection.insert_one(conversation_doc)
        conversation_id = str(result.inserted_id)

        logger.info(f"새 대화방 생성: {conversation_id} (사용자: {current_user_id})")

        # 응답 생성
        return ConversationResponse(
            id=conversation_id,
            user_id=current_user_id,
            title=conversation_data.title,
            created_at=now,
            updated_at=now,
            message_count=0
        )

    except Exception as e:
        logger.error(f"대화방 생성 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화방 생성 중 오류가 발생했습니다"
        )


@router.get(
    "",
    response_model=List[ConversationResponse],
    summary="내 대화방 목록 조회",
    description="현재 사용자의 모든 대화방을 조회합니다 (최신순)."
)
async def get_my_conversations(
    current_user_id: str = Depends(get_current_user)
):
    """
    내 대화방 목록 조회

    - 최신순 정렬 (updated_at 기준)
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        # 사용자의 대화방 조회 (최신순)
        cursor = conversations_collection.find(
            {"user_id": current_user_id}
        ).sort("updated_at", -1)  # -1: 내림차순 (최신순)

        conversations = await cursor.to_list(length=None)

        logger.info(f"대화방 목록 조회: {len(conversations)}개 (사용자: {current_user_id})")

        # 응답 변환
        return [
            ConversationResponse(
                id=str(conv["_id"]),
                user_id=conv["user_id"],
                title=conv["title"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"],
                message_count=conv.get("message_count", 0)
            )
            for conv in conversations
        ]

    except Exception as e:
        logger.error(f"대화방 목록 조회 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화방 목록 조회 중 오류가 발생했습니다"
        )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="특정 대화방 조회",
    description="특정 대화방의 정보를 조회합니다."
)
async def get_conversation(
    conversation_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """
    특정 대화방 조회

    - **conversation_id**: 대화방 ID
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        # ObjectId 변환
        try:
            obj_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 대화방 ID입니다"
            )

        # 대화방 조회
        conversation = await conversations_collection.find_one({"_id": obj_id})

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화방을 찾을 수 없습니다"
            )

        # 소유권 확인
        if conversation["user_id"] != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="접근 권한이 없습니다"
            )

        # 응답 생성
        return ConversationResponse(
            id=str(conversation["_id"]),
            user_id=conversation["user_id"],
            title=conversation["title"],
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            message_count=conversation.get("message_count", 0)
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"대화방 조회 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화방 조회 중 오류가 발생했습니다"
        )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="대화방 삭제",
    description="특정 대화방과 관련된 모든 메시지를 삭제합니다."
)
async def delete_conversation(
    conversation_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """
    대화방 삭제

    - **conversation_id**: 대화방 ID
    - 대화방과 관련된 모든 메시지도 함께 삭제됩니다
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)
        messages_collection = db_instance.get_collection(Collections.MESSAGES)

        # ObjectId 변환
        try:
            obj_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 대화방 ID입니다"
            )

        # 대화방 조회
        conversation = await conversations_collection.find_one({"_id": obj_id})

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화방을 찾을 수 없습니다"
            )

        # 소유권 확인
        if conversation["user_id"] != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="접근 권한이 없습니다"
            )

        # 1. 관련 메시지 삭제
        delete_messages_result = await messages_collection.delete_many(
            {"conversation_id": conversation_id}
        )

        # 2. 대화방 삭제
        await conversations_collection.delete_one({"_id": obj_id})

        logger.info(
            f"대화방 삭제 완료: {conversation_id} "
            f"(메시지 {delete_messages_result.deleted_count}개 삭제)"
        )

        return None

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"대화방 삭제 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화방 삭제 중 오류가 발생했습니다"
        )


@router.patch(
    "/{conversation_id}/title",
    response_model=ConversationResponse,
    summary="대화방 제목 수정",
    description="특정 대화방의 제목을 수정합니다."
)
async def update_conversation_title(
    conversation_id: str,
    update_data: ConversationUpdate,
    current_user_id: str = Depends(get_current_user)
):
    """
    대화방 제목 수정

    - **conversation_id**: 대화방 ID
    - **title**: 새 제목
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        # ObjectId 변환
        try:
            obj_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 대화방 ID입니다"
            )

        # 대화방 조회
        conversation = await conversations_collection.find_one({"_id": obj_id})

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화방을 찾을 수 없습니다"
            )

        # 소유권 확인
        if conversation["user_id"] != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="접근 권한이 없습니다"
            )

        # 제목 업데이트
        now = datetime.utcnow()
        await conversations_collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "title": update_data.title,
                    "updated_at": now
                }
            }
        )

        logger.info(f"대화방 제목 수정: {conversation_id} -> {update_data.title}")

        # 응답 생성
        return ConversationResponse(
            id=conversation_id,
            user_id=current_user_id,
            title=update_data.title,
            created_at=conversation["created_at"],
            updated_at=now,
            message_count=conversation.get("message_count", 0)
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"대화방 제목 수정 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화방 제목 수정 중 오류가 발생했습니다"
        )


@router.get(
    "/{conversation_id}/messages",
    response_model=List[MessageResponse],
    summary="대화방 메시지 조회",
    description="특정 대화방의 모든 메시지를 조회합니다 (순서대로)."
)
async def get_conversation_messages(
    conversation_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """
    대화방 메시지 조회

    - **conversation_id**: 대화방 ID
    - 메시지 순서대로 정렬
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)
        messages_collection = db_instance.get_collection(Collections.MESSAGES)

        # ObjectId 변환
        try:
            obj_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 대화방 ID입니다"
            )

        # 대화방 조회 및 소유권 확인
        conversation = await conversations_collection.find_one({"_id": obj_id})

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화방을 찾을 수 없습니다"
            )

        if conversation["user_id"] != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="접근 권한이 없습니다"
            )

        # 메시지 조회 (순서대로)
        cursor = messages_collection.find(
            {"conversation_id": conversation_id}
        ).sort("order", 1)  # 1: 오름차순 (순서대로)

        messages = await cursor.to_list(length=None)

        logger.info(f"메시지 조회: {len(messages)}개 (대화방: {conversation_id})")

        # 응답 변환
        return [
            MessageResponse(
                id=str(msg["_id"]),
                conversation_id=msg["conversation_id"],
                role=msg["role"],
                content=msg["content"],
                sources=msg.get("sources", []),
                created_at=msg["created_at"],
                order=msg["order"]
            )
            for msg in messages
        ]

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"메시지 조회 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="메시지 조회 중 오류가 발생했습니다"
        )


# Chat API 요청/응답 모델
class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    conversation_id: str
    query: str
    k: int = 3  # 검색할 문서 수
    include_sources: bool = True  # 출처 포함 여부


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    message_id: str
    answer: str
    sources: list = []


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    current_user_id: str = Depends(get_current_user)
):
    """
    대화방에서 채팅하기

    1. 질문 의도 분류 (수업 관련 vs 일상 대화)
    2. 수업 관련: PINECONE 검색 + HyperCLOVA 직접 답변
    3. 일상 대화: HyperCLOVA 직접 답변
    4. 메시지를 DB에 저장
    5. 1번째 또는 5번째 메시지인 경우 대화방 제목 자동 생성
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)
        messages_collection = db_instance.get_collection(Collections.MESSAGES)

        # 1. 대화방 확인 및 소유권 검증
        try:
            obj_id = ObjectId(request.conversation_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 대화방 ID입니다"
            )

        conversation = await conversations_collection.find_one({"_id": obj_id})

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화방을 찾을 수 없습니다"
            )

        if conversation["user_id"] != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="접근 권한이 없습니다"
            )

        logger.info(f"채팅 요청: {request.query} (대화방: {request.conversation_id})")

        # 2. 현재 메시지 순서 계산
        current_message_count = conversation.get("message_count", 0)
        user_message_order = current_message_count
        bot_message_order = current_message_count + 1

        # 3. 사용자 메시지 저장
        now = datetime.utcnow()
        user_message_doc = {
            "conversation_id": request.conversation_id,
            "role": "user",
            "content": request.query,
            "sources": [],
            "created_at": now,
            "order": user_message_order
        }

        user_msg_result = await messages_collection.insert_one(user_message_doc)

        # 4. AI 응답 생성
        hyperclova = get_hyperclova_client()

        # 질문 의도 분류 (비동기)
        intent = await hyperclova.classify_intent(request.query)
        logger.info(f"질문 의도: {intent}")

        answer = ""
        sources = []

        # 일상 대화인 경우 바로 답변 (비동기)
        if intent == 'casual_chat':
            logger.info("일상 대화로 분류 - 직접 답변 생성")
            answer = await hyperclova.generate_casual_answer(request.query)
        else:
            # PINECONE 벡터 검색 (비동기)
            logger.info(f"{intent} 분류 - 벡터 검색 수행")

            vectorstore = get_vectorstore_service()
            search_results = await vectorstore.similarity_search(
                query=request.query,
                k=request.k
            )

            if not search_results:
                answer = "죄송합니다. 관련 수업 정보를 찾을 수 없습니다. 다른 방식으로 질문해주시겠어요?"
            else:
                logger.info(f"검색된 문서 수: {len(search_results)}")

                # HyperCLOVA가 직접 질문을 이해하고 답변 생성 (비동기)
                answer = await hyperclova.generate_answer(
                    query=request.query,
                    context_docs=search_results
                )

                # 출처 정보 수집
                if request.include_sources:
                    for result in search_results:
                        sources.append({
                            "course_name": result["metadata"].get("course_name", ""),
                            "professor": result["metadata"].get("professor", ""),
                            "section": result["metadata"].get("section", ""),
                            "content_preview": result["page_content"][:200] + "..."
                        })

        # 5. 봇 메시지 저장
        bot_message_doc = {
            "conversation_id": request.conversation_id,
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "created_at": datetime.utcnow(),
            "order": bot_message_order
        }

        bot_msg_result = await messages_collection.insert_one(bot_message_doc)

        # 6. 대화방 업데이트 (메시지 수 증가, 업데이트 시간 갱신)
        new_message_count = bot_message_order + 1
        await conversations_collection.update_one(
            {"_id": obj_id},
            {
                "$set": {
                    "message_count": new_message_count,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        # 7. 대화방 제목 자동 생성 (1번째 또는 5번째 메시지)
        # message_count가 2 (user + bot 1개씩) 또는 10 (user + bot 5개씩)일 때 실행
        if new_message_count == 2 or new_message_count == 10:
            await auto_generate_title(
                conversation_id=request.conversation_id,
                message_count=new_message_count,
                user_query=request.query
            )

        logger.info("채팅 응답 완료")

        return ChatResponse(
            message_id=str(bot_msg_result.inserted_id),
            answer=answer,
            sources=sources
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )
