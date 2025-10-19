# RAG 시스템 설정 가이드

이 문서는 수업계획서 기반 질의응답 시스템(RAG)을 설정하고 실행하는 방법을 설명합니다.

## 📋 시스템 개요

**RAG (Retrieval-Augmented Generation)** 아키텍처:
1. **벡터 검색**: 사용자 질문과 유사한 수업계획서 정보 검색 (FAISS)
2. **답변 생성**: 검색된 정보를 바탕으로 HyperCLOVA X가 답변 생성

## 🔧 사전 준비

### 1. Python 패키지 설치

```bash
cd backend
pip install -r requirements.txt
```

### 2. HyperCLOVA X API 키 발급

1. [네이버 클라우드 플랫폼](https://console.ncloud.com/) 접속
2. AI·NAVER API > CLOVA Studio 메뉴로 이동
3. 새 프로젝트 생성 (또는 기존 프로젝트 선택)
4. API 키 발급:
   - `X-NCP-CLOVASTUDIO-API-KEY`: CLOVA Studio API 키
   - `X-NCP-APIGW-API-KEY`: API Gateway 키

### 3. 환경 변수 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성:

```bash
cp .env.example .env
```

`.env` 파일을 열고 HyperCLOVA X API 키를 입력:

```env
NAVER_CLOVA_API_KEY=your-clova-api-key-here
NAVER_CLOVA_API_KEY_PRIMARY=your-clova-api-key-primary-here
```

## 🚀 시스템 실행

### 단계 1: 데이터 벡터화

수업계획서 JSON 데이터를 벡터로 변환하고 FAISS 인덱스를 생성합니다.

```bash
cd backend
python vectorize_data.py
```

**출력 예시:**
```
===========================================================
🚀 수업계획서 데이터 벡터화 시작
===========================================================

📂 데이터 로드 중: ../utils/output.json
✅ 데이터 로드 완료
📦 임베딩 모델 로딩 중: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
✅ 모델 로드 완료 (차원: 384)
📄 총 1,234개의 텍스트 청크 추출됨
🔄 텍스트 벡터화 중...
✅ 벡터화 완료: (1234, 384)
🗂️ FAISS 인덱스 생성 중...
✅ 인덱스 생성 완료: 1234개 벡터
💾 FAISS 인덱스 저장: backend/vector_db/faiss_index.bin
💾 메타데이터 저장: backend/vector_db/chunks_metadata.pkl

🎉 벡터화 완료!
```

생성된 파일:
- `backend/vector_db/faiss_index.bin`: FAISS 인덱스
- `backend/vector_db/chunks_metadata.pkl`: 메타데이터

### 단계 2: 벡터 검색 테스트 (선택사항)

```bash
python vector_search.py
```

### 단계 3: RAG 파이프라인 테스트 (선택사항)

HyperCLOVA X API 키가 설정되어 있어야 합니다.

```bash
python rag_pipeline.py
```

### 단계 4: FastAPI 서버 실행

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

서버가 시작되면:
- **API 문서**: http://localhost:5000/docs
- **Health Check**: http://localhost:5000/health
- **RAG 상태**: http://localhost:5000/rag/status

## 📡 API 사용법

### 1. 질의응답 API

**엔드포인트**: `POST /chat`

**요청 예시**:
```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "C언어프로그래밍 수업은 누가 가르치나요?",
    "top_k": 3,
    "return_sources": true
  }'
```

**응답 예시**:
```json
{
  "answer": "C언어프로그래밍 수업은 정원석 교수님이 담당하고 계십니다. 연락처는 010-6357-5409이며, 이메일은 einrock@naver.com입니다.",
  "query": "C언어프로그래밍 수업은 누가 가르치나요?",
  "sources": [
    {
      "subject": "[0367001]C언어프로그래밍",
      "type": "교과목_운영",
      "text": "교과목: [0367001]C언어프로그래밍\n담당교수: 정원석\n...",
      "similarity": 0.87
    }
  ]
}
```

### 2. 검색 API

**엔드포인트**: `POST /search`

**요청 예시**:
```bash
curl -X POST "http://localhost:5000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "프로그래밍 평가 방법",
    "top_k": 5
  }'
```

### 3. RAG 시스템 상태 확인

**엔드포인트**: `GET /rag/status`

```bash
curl http://localhost:5000/rag/status
```

## 🐳 Docker로 실행

Docker Compose를 사용하면 더 간편합니다:

```bash
# 프로젝트 루트 디렉토리에서
docker-compose up --build
```

## 🔍 트러블슈팅

### 문제 1: "FAISS 인덱스를 찾을 수 없습니다"

**해결방법**: 먼저 `vectorize_data.py`를 실행하여 인덱스를 생성하세요.

```bash
python vectorize_data.py
```

### 문제 2: "HyperCLOVA X API 키가 설정되지 않았습니다"

**해결방법**: `.env` 파일에 API 키를 올바르게 설정했는지 확인하세요.

```bash
# .env 파일 확인
cat .env | grep NAVER_CLOVA
```

### 문제 3: "output.json을 찾을 수 없습니다"

**해결방법**: 먼저 `utils/excel_utils.py`를 실행하여 Excel 파일을 JSON으로 변환하세요.

```bash
cd utils
python excel_utils.py
```

### 문제 4: 메모리 부족

**해결방법**: 
- 배치 크기 줄이기: `vectorize_data.py`에서 `batch_size=32`를 `batch_size=16`으로 변경
- 더 작은 임베딩 모델 사용 고려

## 📊 성능 최적화

### 1. 검색 성능

- `top_k` 값 조정 (기본값: 3)
- 더 높은 값 = 더 많은 컨텍스트, 느린 속도
- 더 낮은 값 = 빠른 속도, 정확도 감소

### 2. 답변 생성 성능

- `max_tokens`: 생성할 최대 토큰 수 (기본값: 500)
- `temperature`: 생성 다양성 (기본값: 0.7, 낮을수록 일관적)

## 🎯 다음 단계

1. **프론트엔드 연동**: React 앱에서 `/chat` API 호출
2. **대화 히스토리**: 이전 대화 기록을 포함한 멀티턴 대화
3. **필터링**: 특정 교과목이나 교수님으로 필터링
4. **캐싱**: 자주 묻는 질문에 대한 캐싱

## 📚 참고 자료

- [HyperCLOVA X 공식 문서](https://www.ncloud.com/product/aiService/clovaStudio)
- [FAISS 문서](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)

## 💬 지원

문제가 발생하면 팀원에게 문의하세요! 😊

