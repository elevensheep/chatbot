"""
대화 제목 자동 생성 서비스
"""
import logging
from bson import ObjectId
from database import Collections, db_instance
from hyperclova_client import get_hyperclova_client

logger = logging.getLogger(__name__)


async def auto_generate_title(
    conversation_id: str,
    message_count: int,
    user_query: str
) -> None:
    """
    대화 제목 자동 생성

    - 1번째 대화 (message_count == 2): 첫 사용자 질문의 앞 30자
    - 5번째 대화 (message_count == 10): HyperCLOVA로 대화 요약

    Args:
        conversation_id: 대화방 ID
        message_count: 현재 메시지 개수 (사용자 + 봇 메시지 포함)
        user_query: 사용자 질문 (첫 대화 시 제목으로 사용)
    """
    try:
        conversations_collection = db_instance.get_collection(Collections.CONVERSATIONS)

        # 1번째 대화: 첫 질문 30자
        if message_count == 2:
            title = user_query[:30] + ("..." if len(user_query) > 30 else "")
            await conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"title": title}}
            )
            logger.info(f"대화 제목 생성 (1번째): {title}")
            return

        # 5번째 대화: HyperCLOVA 요약
        if message_count == 10:
            # 모든 메시지 가져오기
            messages_collection = db_instance.get_collection(Collections.MESSAGES)
            messages = await messages_collection.find(
                {"conversation_id": ObjectId(conversation_id)}
            ).sort("order", 1).to_list(length=10)

            # 대화 내용 조합
            conversation_text = "\n".join([
                f"{'사용자' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
                for msg in messages
            ])

            # HyperCLOVA로 제목 생성
            hyperclova = get_hyperclova_client()
            prompt = f"""다음 대화 내용을 보고 간결한 제목을 30자 이내로 생성해주세요.
제목만 출력하고 다른 설명은 하지 마세요.

대화 내용:
{conversation_text}

제목:"""

            title = await hyperclova.generate_casual_answer(prompt)
            title = title.strip()[:30]

            await conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"title": title}}
            )
            logger.info(f"대화 제목 생성 (5번째): {title}")
            return

    except Exception as e:
        logger.error(f"제목 생성 중 오류 발생: {e}", exc_info=True)
        # 제목 생성 실패는 치명적이지 않으므로 예외를 던지지 않음
