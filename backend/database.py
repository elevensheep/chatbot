"""
MongoDB Atlas (클라우드) 데이터베이스 연결 설정
"""
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from config import settings

logger = logging.getLogger(__name__)

class Database:
    """MongoDB Atlas 데이터베이스 클래스"""

    client: AsyncIOMotorClient = None

    @classmethod
    async def connect_db(cls):
        """MongoDB Atlas 연결"""
        try:
            # MongoDB Atlas URI 가져오기
            mongo_uri = settings.MONGODB_URI

            if not mongo_uri:
                raise ValueError(
                    "MONGODB_URI 환경변수가 설정되지 않았습니다. "
                    "MongoDB Atlas 연결 문자열을 .env 파일에 추가해주세요.\n"
                    "예시: MONGODB_URI=mongodb+srv://<username>:<password>@cluster.xxxxx.mongodb.net/"
                )

            # MongoDB Atlas 클라이언트 생성
            cls.client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                retryWrites=True,
                w='majority'  # Write concern for Atlas
            )

            # 연결 테스트
            await cls.client.admin.command('ping')
            logger.info(f"MongoDB Atlas 연결 성공 (Database: {settings.MONGODB_DATABASE})")

        except ConnectionFailure as e:
            logger.error(f"MongoDB Atlas 연결 실패: {e}")
            logger.error("MongoDB Atlas 연결 문자열이 올바른지 확인해주세요.")
            raise
        except Exception as e:
            logger.error(f"MongoDB Atlas 초기화 중 오류 발생: {e}")
            raise

    @classmethod
    async def close_db(cls):
        """MongoDB Atlas 연결 종료"""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB Atlas 연결 종료")

    @classmethod
    def get_database(cls):
        """데이터베이스 인스턴스 반환"""
        if cls.client is None:
            raise Exception("Database not initialized. Call connect_db() first.")
        return cls.client[settings.MONGODB_DATABASE]

    @classmethod
    def get_collection(cls, collection_name: str):
        """컬렉션 인스턴스 반환"""
        db = cls.get_database()
        return db[collection_name]


# 전역 데이터베이스 인스턴스
db_instance = Database()


def get_db():
    """데이터베이스 인스턴스 반환 (의존성 주입용)"""
    return db_instance.get_database()


# 컬렉션 이름 상수
class Collections:
    """MongoDB 컬렉션 이름"""
    USERS = "users"
    COURSES = "courses"
    CHAT_HISTORY = "chat_history"
    EMAIL_VERIFICATIONS = "email_verifications"
    CONVERSATIONS = "conversations"  # 대화방
    MESSAGES = "messages"  # 메시지
