# 수업계획서 RAG 시스템

## 🎯 개요

이 시스템은 **RAG (Retrieval-Augmented Generation)** 아키텍처를 사용하여 수업계획서 기반 질의응답 서비스를 제공합니다.

## 🏗️ 시스템 구조

```
사용자 질문
    ↓
[벡터 검색] → FAISS 인덱스 → 관련 수업계획서 정보
    ↓
[컨텍스트 구성]
    ↓
[HyperCLOVA X] → 답변 생성
    ↓
사용자에게 답변 반환
```

## 📦 주요 컴포넌트

### 1. `vectorize_data.py`
- Excel 데이터(JSON)를 벡터로 변환
- FAISS 인덱스 생성
- Sentence Transformers 사용 (한국어 지원)

### 2. `vector_search.py`
- FAISS 인덱스 로드 및 검색
- 유사도 기반 문서 검색
- 싱글톤 패턴

### 3. `hyperclova.py`
- HyperCLOVA X API 클라이언트
- 스트리밍 응답 처리
- 프롬프트 엔지니어링

### 4. `rag_pipeline.py`
- 전체 RAG 파이프라인 통합
- 검색 + 생성 통합
- 컨텍스트 포맷팅

### 5. `main.py`
- FastAPI 엔드포인트
- `/chat`: 질의응답
- `/search`: 벡터 검색만
- `/rag/status`: 시스템 상태

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env.example을 복사
cp .env.example .env

# .env 파일 편집
# NAVER_CLOVA_API_KEY와 NAVER_CLOVA_API_KEY_PRIMARY 설정
```

### 3. 데이터 벡터화
```bash
python vectorize_data.py
```

### 4. 서버 실행
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

### 5. API 테스트
```bash
# 브라우저에서
http://localhost:5000/docs

# curl로
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "C언어 수업은 누가 가르치나요?"}'
```

## 📡 API 엔드포인트

### POST /chat
질의응답 (RAG 전체 파이프라인)

**요청**:
```json
{
  "query": "질문 내용",
  "top_k": 3,
  "return_sources": true,
  "max_tokens": 500,
  "temperature": 0.7
}
```

**응답**:
```json
{
  "answer": "AI 답변",
  "query": "원본 질문",
  "sources": [...]
}
```

### POST /search
벡터 검색만 (답변 생성 없음)

**요청**:
```json
{
  "query": "검색 쿼리",
  "top_k": 5
}
```

**응답**:
```json
{
  "results": [...],
  "query": "검색 쿼리"
}
```

### GET /rag/status
RAG 시스템 상태 확인

**응답**:
```json
{
  "status": "ready",
  "message": "RAG 시스템이 정상 작동 중입니다.",
  "vector_db_loaded": true,
  "llm_loaded": true
}
```

## 🔧 커스터마이징

### 1. 임베딩 모델 변경

`vectorize_data.py`와 `vector_search.py`에서:

```python
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 다른 모델로 변경 가능
```

### 2. 시스템 프롬프트 수정

`rag_pipeline.py`의 `DEFAULT_SYSTEM_PROMPT` 수정

### 3. 검색 개수 조정

```python
# 기본값은 3
rag = RAGPipeline(top_k=5)
```

## 📊 성능 최적화

### 벡터 검색
- **top_k**: 검색할 문서 개수 (1-10 권장)
- **임베딩 모델**: 더 작은 모델 사용 시 속도 향상

### 답변 생성
- **max_tokens**: 생성 토큰 수 (적을수록 빠름)
- **temperature**: 생성 다양성 (낮을수록 일관적)

## 🐳 Docker 실행

프로젝트 루트에서:

```bash
docker-compose up --build
```

## 📝 개발 노트

### 사용 기술
- **벡터 DB**: FAISS (로컬, 무료, 빠름)
- **임베딩**: Sentence Transformers (한국어 지원)
- **LLM**: HyperCLOVA X (네이버 클라우드)
- **프레임워크**: FastAPI

### 왜 FAISS?
- ✅ 완전 무료
- ✅ 로컬 실행 (외부 서비스 불필요)
- ✅ 빠른 검색 속도
- ✅ 쉬운 설정

### 왜 HyperCLOVA X?
- ✅ 한국어 특화
- ✅ 높은 품질
- ✅ 네이버 클라우드 생태계

## 🔍 트러블슈팅

### "FAISS 인덱스를 찾을 수 없습니다"
→ `python vectorize_data.py` 실행

### "HyperCLOVA X API 키가 설정되지 않았습니다"
→ `.env` 파일 확인

### "output.json을 찾을 수 없습니다"
→ `cd ../utils && python excel_utils.py` 실행

### 메모리 부족
→ `vectorize_data.py`에서 `batch_size` 줄이기

## 📚 다음 단계

- [ ] 대화 히스토리 추가 (멀티턴 대화)
- [ ] 교과목별 필터링
- [ ] 자주 묻는 질문 캐싱
- [ ] 답변 품질 평가
- [ ] A/B 테스트

## 📖 상세 문서

- [RAG_SETUP.md](./RAG_SETUP.md): 상세 설정 가이드
- [API 문서](http://localhost:5000/docs): FastAPI 자동 생성 문서

---

**작성일**: 2025-10-19  
**버전**: 1.0.0

