"""
이메일 발송 유틸리티
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import secrets
import logging
from typing import Optional
from config import settings
from database import Collections, db_instance

logger = logging.getLogger(__name__)


def generate_verification_token() -> str:
    """이메일 인증 토큰 생성 (32자리 랜덤 문자열)"""
    return secrets.token_urlsafe(32)


async def save_verification_token(email: str, token: str, expires_minutes: int = 30):
    """
    이메일 인증 토큰을 데이터베이스에 저장

    Args:
        email: 이메일 주소
        token: 인증 토큰
        expires_minutes: 토큰 만료 시간 (분)
    """
    try:
        verification_collection = db_instance.get_collection(Collections.EMAIL_VERIFICATIONS)

        # 기존 토큰 삭제
        deleted = await verification_collection.delete_many({"email": email})
        if deleted.deleted_count > 0:
            logger.info(f"기존 토큰 {deleted.deleted_count}개 삭제: {email}")

        # 새 토큰 저장
        expires_at = datetime.utcnow() + timedelta(minutes=expires_minutes)
        verification_doc = {
            "email": email,
            "token": token,
            "expires_at": expires_at,
            "created_at": datetime.utcnow(),
            "verified": False
        }

        result = await verification_collection.insert_one(verification_doc)
        logger.info(f"✓ 이메일 인증 토큰 저장 완료: {email}")
        logger.info(f"  토큰: {token[:20]}... (길이: {len(token)})")
        logger.info(f"  만료시간: {expires_at} (UTC)")
        logger.info(f"  Document ID: {result.inserted_id}")

    except Exception as e:
        logger.error(f"토큰 저장 중 오류: {e}", exc_info=True)
        raise


async def verify_token(token: str) -> Optional[str]:
    """
    토큰을 검증하고 이메일 주소를 반환

    Args:
        token: 인증 토큰

    Returns:
        이메일 주소 또는 None
    """
    try:
        verification_collection = db_instance.get_collection(Collections.EMAIL_VERIFICATIONS)

        logger.info(f"토큰 검증 시도: {token[:20]}... (길이: {len(token)})")

        # 토큰 조회 (verified 상태 관계없이)
        verification = await verification_collection.find_one({"token": token})

        if not verification:
            logger.warning(f"토큰을 찾을 수 없음: {token[:20]}...")
            # 디버깅: 최근 토큰 확인
            recent_tokens = await verification_collection.find().sort("created_at", -1).limit(3).to_list(3)
            logger.info(f"최근 생성된 토큰 {len(recent_tokens)}개:")
            for rt in recent_tokens:
                logger.info(f"  - {rt['email']}: {rt['token'][:20]}... (verified: {rt.get('verified', False)})")
            return None

        logger.info(f"토큰 찾음: {verification['email']}, verified={verification.get('verified', False)}")

        # 이미 검증된 토큰인지 확인
        if verification.get("verified", False):
            logger.info(f"이미 인증된 토큰입니다: {verification['email']}")
            # 이미 인증되었어도 이메일 주소 반환 (재사용 가능)
            return verification["email"]

        # 만료 확인
        if verification["expires_at"] < datetime.utcnow():
            logger.warning(f"만료된 토큰: {token[:20]}... (만료: {verification['expires_at']})")
            return None

        # 토큰 검증 완료 처리
        await verification_collection.update_one(
            {"_id": verification["_id"]},
            {"$set": {"verified": True, "verified_at": datetime.utcnow()}}
        )

        logger.info(f"✓ 이메일 인증 성공: {verification['email']}")
        return verification["email"]

    except Exception as e:
        logger.error(f"토큰 검증 중 오류: {e}", exc_info=True)
        return None


async def send_verification_email(email: str, token: str, frontend_url: str = "http://localhost:3000"):
    """
    이메일 인증 메일 발송

    Args:
        email: 수신자 이메일
        token: 인증 토큰
        frontend_url: 프론트엔드 URL
    """
    # SMTP 설정
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = settings.SMTP_USER
    smtp_password = settings.SMTP_PASSWORD

    # SMTP 설정 확인
    if not smtp_user or not smtp_password:
        error_msg = "SMTP 설정이 없습니다. .env 파일의 SMTP_USER와 SMTP_PASSWORD를 확인하세요."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"{email}로 이메일 인증 메일을 발송합니다...")
    logger.info(f"SMTP Server: {smtp_server}:{smtp_port}")
    logger.info(f"SMTP User: {smtp_user}")
    logger.info(f"Password Length: {len(smtp_password)} characters")

    try:

        # 이메일 메시지 생성
        msg = MIMEMultipart('alternative')
        msg['Subject'] = '[백석대학교 수업계획서 챗봇] 이메일 인증'
        msg['From'] = smtp_user
        msg['To'] = email

        # 인증 링크
        verification_link = f"{frontend_url}/verify-email?token={token}"

        # HTML 본문
        html_body = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
              <h2 style="color: #0066cc;">백석대학교 수업계획서 챗봇</h2>
              <h3>이메일 인증</h3>
              <p>안녕하세요,</p>
              <p>회원가입을 완료하기 위해 아래 버튼을 클릭하여 이메일을 인증해주세요.</p>
              <div style="text-align: center; margin: 30px 0;">
                <a href="{verification_link}"
                   style="display: inline-block; padding: 12px 30px; background-color: #0066cc; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">
                  이메일 인증하기
                </a>
              </div>
              <p style="color: #666; font-size: 14px;">
                이 링크는 30분 동안 유효합니다.<br>
                버튼이 작동하지 않으면 아래 링크를 복사하여 브라우저에 붙여넣으세요:
              </p>
              <p style="word-break: break-all; color: #0066cc; font-size: 12px;">
                {verification_link}
              </p>
              <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
              <p style="color: #999; font-size: 12px;">
                본인이 요청하지 않았다면 이 메일을 무시하셔도 됩니다.
              </p>
            </div>
          </body>
        </html>
        """

        # 텍스트 본문 (HTML을 지원하지 않는 이메일 클라이언트용)
        text_body = f"""
        백석대학교 수업계획서 챗봇 - 이메일 인증

        안녕하세요,

        회원가입을 완료하기 위해 아래 링크를 클릭하여 이메일을 인증해주세요.

        인증 링크: {verification_link}

        이 링크는 30분 동안 유효합니다.

        본인이 요청하지 않았다면 이 메일을 무시하셔도 됩니다.
        """

        # 메시지에 본문 추가
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)

        # SMTP 서버 연결 및 이메일 발송
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        logger.info(f"✓ 이메일 인증 메일 발송 완료: {email}")

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP 인증 실패: {e}")
        logger.error("Gmail 앱 비밀번호를 확인하세요. 일반 비밀번호가 아닌 앱 비밀번호를 사용해야 합니다.")
        raise ValueError("이메일 발송 실패: SMTP 인증 오류")

    except smtplib.SMTPException as e:
        logger.error(f"SMTP 오류: {e}")
        raise ValueError(f"이메일 발송 실패: {str(e)}")

    except Exception as e:
        logger.error(f"이메일 발송 중 예상치 못한 오류: {e}", exc_info=True)
        raise ValueError(f"이메일 발송 실패: {str(e)}")
