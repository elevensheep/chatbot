"""
대화방 제목 자동 생성 서비스
"""
import logging
from datetime import datetime
from bson import ObjectId

from database import Collections, db_instance
from hyperclova_client import get_hyperclova_client

logger = logging.getLogger(__name__)


async def auto_generate_title(conversation_id: str, message_count: int, user_query: str):
    """
    메시지 개수에 따라 대화방 제목 자동 생성

    Args:
        conversation_id: 대화방 ID
        message_count: 현재 메시지 개수
        user_query: 사용자 질문

    Logic:
        - 첫 메시지 (message_count == 1): 질문 일부를 제목으로
        - 5번째 메시지 (message_count == 5): HyperCLOVA로 대화 요약
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        # ObjectId 변환
        conversation_obj_id = ObjectId(conversation_id)

        # 대화방 조회
        conversation = await conversations_collection.find_one({"_id": conversation_obj_id})

        if not conversation:
            logger.warning(f"대화방을 찾을 수 없습니다: {conversation_id}")
            return

        # 1. 첫 번째 메시지: 질문 일부를 제목으로
        if message_count == 1:
            # 질문이 30자 이상이면 자르기
            if len(user_query) > 30:
                title = user_query[:30] + "..."
            else:
                title = user_query

            await conversations_collection.update_one(
                {"_id": conversation_obj_id},
                {"$set": {"title": title}}
            )

            logger.info(f"첫 메시지 기반 제목 생성: {title}")

        # 2. 5번째 메시지: HyperCLOVA로 요약
        elif message_count == 5:
            # "새 대화" 상태에서만 요약 실행 (이미 제목이 있으면 스킵)
            if conversation.get("title") == "새 대화":
                # 백그라운드에서 요약 (비동기)
                await summarize_conversation_title(conversation_id)

    except Exception as e:
        logger.error(f"제목 자동 생성 중 오류: {e}", exc_info=True)


async def summarize_conversation_title(conversation_id: str):
    """
    HyperCLOVA를 사용하여 대화 내용을 요약한 제목 생성

    Args:
        conversation_id: 대화방 ID
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)
        messages_collection = db_instance.get_collection(Collections.MESSAGES)

        # 최근 5개 메시지 가져오기
        cursor = messages_collection.find(
            {"conversation_id": conversation_id}
        ).sort("order", 1).limit(5)

        messages = await cursor.to_list(5)

        if len(messages) < 2:
            logger.info("메시지가 충분하지 않아 요약을 건너뜁니다")
            return

        # 대화 컨텍스트 구성
        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])

        # HyperCLOVA로 요약 요청
        hyperclova = get_hyperclova_client()

        prompt = f"""다음 대화를 15자 이내로 요약한 제목을 만들어주세요.
대화의 핵심 주제를 간결하게 표현해야 합니다.

대화:
{context}

제목 (15자 이내):"""

        # 요약 생성 (HyperCLOVA의 일반 답변 생성 함수 사용)
        summary = await hyperclova.generate_casual_answer(prompt)

        # 제목 정제 (따옴표, 개행 제거)
        title = summary.strip().replace('"', '').replace("'", '').replace('\n', ' ')

        # 15자 제한
        if len(title) > 15:
            title = title[:15] + "..."

        # 대화방 제목 업데이트
        conversation_obj_id = ObjectId(conversation_id)
        await conversations_collection.update_one(
            {"_id": conversation_obj_id},
            {"$set": {"title": title}}
        )

        logger.info(f"HyperCLOVA 요약 제목 생성 완료: {title}")

    except Exception as e:
        logger.error(f"대화 요약 중 오류: {e}", exc_info=True)
