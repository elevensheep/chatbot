@echo off
REM Docker로 챗봇 프로덕션 환경 실행 스크립트

echo ===============================
echo 수업계획서 챗봇 프로덕션 환경 실행
echo ===============================
echo.

REM .env 파일 확인
echo [1] 환경 변수 파일 확인 중...
if not exist ".env" (
    echo [오류] .env 파일이 없습니다!
    echo        프로젝트 루트에 .env 파일을 생성하고 필수 환경 변수를 설정하세요.
    echo.
    echo        필수 변수:
    echo        - MONGODB_URI
    echo        - PINECONE_API_KEY
    echo        - JWT_SECRET_KEY
    echo.
    pause
    exit /b 1
)

echo [1] .env 파일 확인 완료
echo     (docker-compose가 자동으로 .env 파일을 읽습니다)
echo.

echo [2] 프로덕션 Docker Compose 실행 중...
echo     - ENVIRONMENT=production
echo     - CORS: 프로덕션 도메인만 허용
echo     - DEBUG=false
echo     - 메모리: 8GB 제한
echo.

echo [3] Docker 컨테이너 시작...
echo     백엔드: http://localhost:5000
echo     프론트엔드: http://localhost:3000
echo     헬스체크: http://localhost:5000/api/health
echo.

docker-compose -f docker-compose.prod.yml up --build

echo.
echo ===============================
echo 종료되었습니다.
echo ===============================

