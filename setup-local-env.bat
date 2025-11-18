@echo off
chcp 65001 > nul
echo ================================================
echo 로컬 개발 환경 설정 스크립트
echo ================================================
echo.

REM .env 파일이 이미 있는지 확인
if exist .env (
    echo [경고] .env 파일이 이미 존재합니다.
    echo.
    choice /C YN /M "기존 .env 파일을 덮어쓰시겠습니까? (Y/N)"
    if errorlevel 2 (
        echo.
        echo [취소] .env 파일을 유지합니다.
        goto :end
    )
    echo.
    echo [삭제] 기존 .env 파일을 삭제합니다...
    del .env
)

echo [생성] .env 파일을 생성합니다...
echo.

REM .env 파일 생성
(
echo # ================================================
echo # MongoDB Atlas 설정 - 필수
echo # ================================================
echo MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true^&w=majority
echo MONGODB_DATABASE=chatbot_db
echo.
echo # ================================================
echo # PINECONE 벡터 스토어 설정 - 필수
echo # ================================================
echo PINECONE_API_KEY=your-pinecone-api-key-here
echo PINECONE_INDEX_NAME=chatbot-courses
echo.
echo # ================================================
echo # JWT 인증 설정 - 필수 (프로덕션에서는 강력한 키 사용)
echo # ================================================
echo JWT_SECRET_KEY=dev-jwt-secret-key-change-me-in-production
echo JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440
echo.
echo # ================================================
echo # SMTP 이메일 설정 - 이메일 인증 기능 사용 시 필수
echo # ================================================
echo SMTP_USER=your-email@gmail.com
echo SMTP_PASSWORD=your-app-password
echo.
echo # ================================================
echo # CORS 설정
echo # ================================================
echo ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000
echo.
echo # ================================================
echo # SEED 모델 설정 (HyperCLOVAX SEED)
echo # ================================================
echo # 모델 경로: 1.5B (기본), 3B, 또는 다른 모델
echo SEED_MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
echo SEED_DEVICE=cpu
echo HF_TOKEN=
echo SEED_LOAD_IN_8BIT=false
echo SEED_LOAD_IN_4BIT=false
echo HF_USE_FP16=false
echo.
echo # ================================================
echo # 서버 설정
echo # ================================================
echo PORT=5000
echo HOST=0.0.0.0
echo ENVIRONMENT=development
echo DEBUG=true
echo LOG_LEVEL=INFO
echo BACKEND_PORT=5000
echo BACKEND_WORKERS=1
echo.
echo # ================================================
echo # API 문서 설정
echo # ================================================
echo ENABLE_DOCS=true
echo.
echo # ================================================
echo # 레거시 설정 (현재 사용 안 함)
echo # ================================================
echo HYPERCLOVA_API_KEY=
echo HYPERCLOVA_API_GATEWAY_KEY=
echo HYPERCLOVA_REQUEST_ID=
) > .env

echo ================================================
echo [완료] .env 파일이 생성되었습니다!
echo ================================================
echo.
echo ⚠️  중요: .env 파일의 필수 설정을 실제 값으로 교체하세요!
echo.
echo 필수 설정 항목:
echo   [필수] MONGODB_URI - MongoDB Atlas 연결 문자열
echo   [필수] PINECONE_API_KEY - Pinecone API 키
echo   [필수] JWT_SECRET_KEY - JWT 인증 키 (프로덕션에서는 강력한 키 사용)
echo   [선택] SMTP_USER, SMTP_PASSWORD - 이메일 인증 기능 사용 시
echo   [선택] HF_TOKEN - Hugging Face private 모델 접근 시
echo.
echo API 키 발급 방법:
echo   - MongoDB Atlas: https://www.mongodb.com/cloud/atlas
echo   - Pinecone: https://www.pinecone.io/
echo   - Hugging Face: https://huggingface.co/settings/tokens
echo.
echo 모델 설정:
echo   - 기본 모델: SEED 1.5B (3-6GB 메모리)
echo   - 더 큰 모델: SEED 3B (8-bit 양자화 권장)
echo   - 모델 변경: SEED_MODEL_PATH 환경 변수 수정
echo.
echo 이제 로컬에서 개발할 수 있습니다:
echo.
echo   방법 1: start-local-dev.bat 사용 (권장)
echo      start-local-dev.bat
echo.
echo   방법 2: 수동 실행
echo      1. 백엔드 실행:
echo         cd backend
echo         python -m venv venv
echo         venv\Scripts\activate
echo         pip install -r requirements.txt
echo         uvicorn main:app --host 0.0.0.0 --port 5000 --reload
echo.
echo      2. 프론트엔드 실행:
echo         cd frontend
echo         npm install
echo         npm start
echo.
echo   방법 3: 프로덕션 환경 테스트
echo      docker-run-prod.bat
echo.

:end
pause

