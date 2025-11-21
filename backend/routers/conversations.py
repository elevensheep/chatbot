"""
대화 관리 API 라우터
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
import logging

from database import Collections, db_instance
from auth_utils import get_current_user
from hyperclova_client import get_hyperclova_client
from direct_pinecone_service import get_vectorstore_service
from services.title_generator import auto_generate_title

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


# ==================== Pydantic 모델 ====================

class ConversationCreate(BaseModel):
    """대화 생성 요청 모델"""
    title: Optional[str] = "새 대화"


class ConversationResponse(BaseModel):
    """대화 응답 모델"""
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationsListResponse(BaseModel):
    """대화 목록 응답 모델"""
    conversations: List[ConversationResponse]


class MessageResponse(BaseModel):
    """메시지 응답 모델"""
    id: str
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[dict]] = []
    order: int
    created_at: str


class MessagesListResponse(BaseModel):
    """메시지 목록 응답 모델"""
    messages: List[MessageResponse]


class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    conversation_id: Optional[str] = None  # None이면 자동으로 새 대화방 생성
    query: str
    k: int = 3
    include_sources: bool = True


class MessageRequest(BaseModel):
    """메시지 추가 요청 모델 (RESTful용)"""
    query: str
    k: int = 3
    include_sources: bool = True


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    answer: str
    sources: List[dict] = []
    message_id: str


# ==================== 내부 헬퍼 함수 ====================

async def _process_chat_message(
    conversation_id: str,
    query: str,
    k: int,
    include_sources: bool,
    current_user_id: str
) -> dict:
    """
    채팅 메시지 처리 공통 로직

    Returns:
        {"answer": str, "sources": list, "message_id": str, "conversation_id": str}
    """
    conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)
    messages_collection = db_instance.get_collection(Collections.MESSAGES)

    # ObjectId 변환
    try:
        conv_object_id = ObjectId(conversation_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="유효하지 않은 대화 ID입니다"
        )

    # 1. 대화방 확인 및 소유권 검증
    conversation = await conversations_collection.find_one({
        "_id": conv_object_id,
        "user_id": ObjectId(current_user_id)
    })

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="대화를 찾을 수 없습니다"
        )

    logger.info(f"채팅 요청: {query} (대화: {conversation_id})")

    # 2. 최근 메시지 3개 조회 (대화 흐름 유지용)
    recent_messages = await messages_collection.find(
        {"conversation_id": conv_object_id}
    ).sort("order", -1).limit(3).to_list(length=3)
    
    # 최근 메시지를 시간순으로 정렬 (오래된 것부터)
    recent_messages.reverse()
    
    # 메시지 히스토리 구성 (role과 content만)
    message_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in recent_messages
    ]
    
    logger.info(f"최근 메시지 {len(message_history)}개를 컨텍스트로 포함")

    # 3. 현재 메시지 순서 계산
    message_count = await messages_collection.count_documents(
        {"conversation_id": conv_object_id}
    )
    user_message_order = message_count
    bot_message_order = message_count + 1

    # 4. 사용자 메시지 저장
    now = datetime.utcnow()
    user_message_doc = {
        "conversation_id": conv_object_id,
        "role": "user",
        "content": query,
        "order": user_message_order,
        "created_at": now
    }
    await messages_collection.insert_one(user_message_doc)

    # 5. AI 응답 생성
    hyperclova = get_hyperclova_client()

    # 5-1. 질문 의도 분류
    intent = await hyperclova.classify_intent(query)
    logger.info(f"질문 의도: {intent}")

    # 5-2. 일상 대화인 경우 바로 답변
    if intent == 'casual_chat':
        logger.info("일상 대화로 분류 - 직접 답변 생성")
        answer = await hyperclova.generate_casual_answer(query, message_history)
        sources = []
    else:
        # 5-3. 수업 관련: Pinecone 벡터 검색
        logger.info(f"{intent} 분류 - 벡터 검색 수행")

        vectorstore = get_vectorstore_service()
        search_results = await vectorstore.similarity_search(
            query=query,
            k=k
        )

        if not search_results:
            answer = "죄송합니다. 관련 수업 정보를 찾을 수 없습니다. 다른 방식으로 질문해주시겠어요?"
            sources = []
        else:
            logger.info(f"검색된 문서 수: {len(search_results)}")

            # 5-4. HyperCLOVA 답변 생성 (최근 메시지 히스토리 포함)
            answer = await hyperclova.generate_answer(
                query=query,
                context_docs=search_results,
                message_history=message_history
            )

            # 5-5. 출처 구성
            sources = []
            if include_sources:
                for result in search_results:
                    sources.append({
                        "course_name": result["metadata"].get("course_name", ""),
                        "professor": result["metadata"].get("professor", ""),
                        "section": result["metadata"].get("section", ""),
                        "content_preview": result["page_content"][:200] + "..."
                    })

    # 6. 봇 메시지 저장
    bot_message_doc = {
        "conversation_id": conv_object_id,
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "order": bot_message_order,
        "created_at": datetime.utcnow()
    }
    bot_message_result = await messages_collection.insert_one(bot_message_doc)
    bot_message_id = str(bot_message_result.inserted_id)

    # 7. 대화방 updated_at 갱신
    await conversations_collection.update_one(
        {"_id": conv_object_id},
        {"$set": {"updated_at": datetime.utcnow()}}
    )

    # 8. 자동 제목 생성 (1번째 또는 5번째 대화)
    new_message_count = bot_message_order + 1  # 사용자 + 봇 메시지 포함
    if new_message_count == 2 or new_message_count == 10:
        await auto_generate_title(
            conversation_id=conversation_id,
            message_count=new_message_count,
            user_query=query
        )

    logger.info("답변 생성 완료")

    return {
        "answer": answer,
        "sources": sources,
        "message_id": bot_message_id,
        "conversation_id": conversation_id
    }


# ==================== API 엔드포인트 ====================

@router.post(
    "",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="새 대화 생성",
    description="새로운 대화방을 생성합니다."
)
async def create_conversation(
    request: ConversationCreate,
    current_user_id: str = Depends(get_current_user)
):
    """
    새 대화 생성

    - **title**: 대화 제목 (기본값: "새 대화")
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        now = datetime.utcnow()
        conversation_doc = {
            "user_id": ObjectId(current_user_id),
            "title": request.title,
            "created_at": now,
            "updated_at": now
        }

        result = await conversations_collection.insert_one(conversation_doc)
        conversation_id = str(result.inserted_id)

        logger.info(f"새 대화 생성: {conversation_id} (사용자: {current_user_id})")

        return {
            "success": True,
            "data": {
                "conversation_id": conversation_id,
                "title": request.title,
                "created_at": now.isoformat() + "Z"
            },
            "message": "대화가 생성되었습니다"
        }

    except Exception as e:
        logger.error(f"대화 생성 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화 생성 중 오류가 발생했습니다"
        )


@router.get(
    "",
    response_model=dict,
    summary="대화 목록 조회",
    description="사용자의 모든 대화를 최신순으로 조회합니다."
)
async def get_conversations(
    current_user_id: str = Depends(get_current_user)
):
    """
    내 대화 목록 조회 (최신순)
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        conversations = await conversations_collection.find(
            {"user_id": ObjectId(current_user_id)}
        ).sort("updated_at", -1).to_list(length=100)

        conversations_list = [
            {
                "id": str(conv["_id"]),
                "title": conv["title"],
                "created_at": conv["created_at"].isoformat() + "Z",
                "updated_at": conv["updated_at"].isoformat() + "Z"
            }
            for conv in conversations
        ]

        logger.info(f"대화 목록 조회: {len(conversations_list)}개 (사용자: {current_user_id})")

        return {
            "success": True,
            "data": {
                "conversations": conversations_list,
                "total": len(conversations_list)
            }
        }

    except Exception as e:
        logger.error(f"대화 목록 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화 목록 조회 중 오류가 발생했습니다"
        )


@router.get(
    "/{conversation_id}",
    response_model=dict,
    summary="특정 대화 조회",
    description="특정 대화의 상세 정보를 조회합니다."
)
async def get_conversation(
    conversation_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """
    특정 대화 조회
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        # ObjectId 변환
        try:
            conv_object_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 대화 ID입니다"
            )

        conversation = await conversations_collection.find_one({
            "_id": conv_object_id,
            "user_id": ObjectId(current_user_id)
        })

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화를 찾을 수 없습니다"
            )

        return {
            "success": True,
            "data": {
                "id": str(conversation["_id"]),
                "title": conversation["title"],
                "created_at": conversation["created_at"].isoformat() + "Z",
                "updated_at": conversation["updated_at"].isoformat() + "Z"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"대화 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화 조회 중 오류가 발생했습니다"
        )


@router.delete(
    "/{conversation_id}",
    response_model=dict,
    summary="대화 삭제",
    description="대화 및 모든 메시지를 삭제합니다."
)
async def delete_conversation(
    conversation_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """
    대화 및 모든 메시지 삭제
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)
        messages_collection = db_instance.get_collection(Collections.MESSAGES)

        # ObjectId 변환
        try:
            conv_object_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 대화 ID입니다"
            )

        # 소유권 확인
        conversation = await conversations_collection.find_one({
            "_id": conv_object_id,
            "user_id": ObjectId(current_user_id)
        })

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화를 찾을 수 없습니다"
            )

        # 메시지 삭제
        await messages_collection.delete_many({"conversation_id": conv_object_id})

        # 대화 삭제
        await conversations_collection.delete_one({"_id": conv_object_id})

        logger.info(f"대화 삭제: {conversation_id} (사용자: {current_user_id})")

        return {
            "success": True,
            "message": "대화가 삭제되었습니다"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"대화 삭제 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="대화 삭제 중 오류가 발생했습니다"
        )


@router.get(
    "/{conversation_id}/messages",
    response_model=dict,
    summary="메시지 목록 조회",
    description="특정 대화의 모든 메시지를 순서대로 조회합니다."
)
async def get_messages(
    conversation_id: str,
    current_user_id: str = Depends(get_current_user)
):
    """
    대화의 모든 메시지 조회 (순서대로)
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)
        messages_collection = db_instance.get_collection(Collections.MESSAGES)

        # ObjectId 변환
        try:
            conv_object_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="유효하지 않은 대화 ID입니다"
            )

        # 소유권 확인
        conversation = await conversations_collection.find_one({
            "_id": conv_object_id,
            "user_id": ObjectId(current_user_id)
        })

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="대화를 찾을 수 없습니다"
            )

        # 메시지 조회
        messages = await messages_collection.find(
            {"conversation_id": conv_object_id}
        ).sort("order", 1).to_list(length=1000)

        messages_list = [
            {
                "id": str(msg["_id"]),
                "conversation_id": conversation_id,
                "role": msg["role"],
                "content": msg["content"],
                "sources": msg.get("sources", []),
                "order": msg["order"],
                "created_at": msg["created_at"].isoformat() + "Z"
            }
            for msg in messages
        ]

        logger.info(f"메시지 목록 조회: {len(messages_list)}개 (대화: {conversation_id})")

        return {
            "success": True,
            "data": {
                "messages": messages_list,
                "total": len(messages_list)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"메시지 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="메시지 조회 중 오류가 발생했습니다"
        )


@router.post(
    "/{conversation_id}/messages",
    response_model=ChatResponse,
    summary="대화방에 메시지 추가 (RESTful)",
    description="특정 대화방에 메시지를 추가하고 AI 응답을 받습니다."
)
async def add_message_to_conversation(
    conversation_id: str,
    request: MessageRequest,
    current_user_id: str = Depends(get_current_user)
):
    """
    RESTful 스타일 메시지 추가 API

    Path parameter로 conversation_id를 받습니다.
    """
    try:
        result = await _process_chat_message(
            conversation_id=conversation_id,
            query=request.query,
            k=request.k,
            include_sources=request.include_sources,
            current_user_id=current_user_id
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            message_id=result["message_id"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"메시지 추가 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )


@router.post(
    "/chat",
    response_model=dict,
    summary="채팅하기 (자동 대화방 생성)",
    description="conversation_id가 없으면 자동으로 새 대화방을 생성합니다."
)
async def chat_with_auto_create(
    request: ChatRequest,
    current_user_id: str = Depends(get_current_user)
):
    """
    채팅 API (자동 대화방 생성)

    - conversation_id가 없으면 자동으로 새 대화방 생성
    - conversation_id가 있으면 해당 대화방에 메시지 추가
    - 로그인 필수
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        # conversation_id가 없으면 자동으로 새 대화방 생성
        if not request.conversation_id:
            logger.info(f"새 대화방 자동 생성 (사용자: {current_user_id})")

            now = datetime.utcnow()
            conversation_doc = {
                "user_id": ObjectId(current_user_id),
                "title": "새 대화",
                "created_at": now,
                "updated_at": now
            }
            result = await conversations_collection.insert_one(conversation_doc)
            conversation_id = str(result.inserted_id)

            logger.info(f"새 대화방 생성 완료: {conversation_id}")
        else:
            conversation_id = request.conversation_id

        # 메시지 처리
        result = await _process_chat_message(
            conversation_id=conversation_id,
            query=request.query,
            k=request.k,
            include_sources=request.include_sources,
            current_user_id=current_user_id
        )

        return {
            "success": True,
            "data": {
                "conversation_id": conversation_id,
                "message_id": result["message_id"],
                "answer": result["answer"],
                "sources": result["sources"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )
