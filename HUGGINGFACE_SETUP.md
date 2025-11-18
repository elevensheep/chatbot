# Hugging Face 모델 설정 가이드

## 현재 코드 상태

코드는 이미 Hugging Face 토큰을 지원하도록 구현되어 있습니다. 따라서 **코드 수정은 필요 없습니다**. 환경 변수만 올바르게 설정하면 됩니다.

## 1.5B → 3B 모델 변경 시 확인 사항

### ✅ 코드는 이미 준비됨
- `HF_TOKEN` 환경 변수 지원 ✅
- 토큰이 있으면 자동으로 사용 ✅
- Private 모델 접근 가능 ✅

### 🔍 확인해야 할 사항

#### 1. 모델 경로 확인
3B 모델의 정확한 Hugging Face 경로를 확인하세요:

**예상 경로:**
- `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B`
- 또는 다른 정확한 경로

**확인 방법:**
- Hugging Face Hub에서 검색: https://huggingface.co/models?search=hyperclovax+seed+3b
- 모델 페이지의 정확한 이름 확인

#### 2. 모델 접근 권한 확인

**Public 모델인 경우:**
- `.env` 파일에서 `HF_TOKEN`을 비워두거나 제거
- 또는 `HF_TOKEN=` (빈 값)

**Private 모델인 경우:**
- Hugging Face에서 토큰 발급 필요
- `.env` 파일에 토큰 추가:
  ```bash
  HF_TOKEN=hf_your_token_here
  ```

#### 3. Hugging Face 사이트에서 확인할 사항

1. **모델 접근 권한**
   - 모델이 Public인지 Private인지 확인
   - Private인 경우 토큰 필요

2. **모델 이름/경로**
   - 정확한 모델 경로 확인
   - 예: `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B`

3. **라이선스 확인**
   - 상업적 사용 가능 여부 확인
   - 사용 조건 확인

## 환경 변수 설정

### .env 파일 예시

```bash
# 3B 모델 사용 (Public 모델인 경우)
SEED_MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B
HF_TOKEN=
SEED_LOAD_IN_8BIT=true  # 메모리 절약 권장

# 3B 모델 사용 (Private 모델인 경우)
SEED_MODEL_PATH=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B
HF_TOKEN=hf_your_token_here
SEED_LOAD_IN_8BIT=true  # 메모리 절약 권장
```

## 코드 동작 방식

현재 코드는 다음과 같이 동작합니다:

1. **토큰이 있으면**: 자동으로 토큰을 사용하여 모델 로드
2. **토큰이 없으면**: Public 모델로 접근 시도
3. **모델 로딩 실패 시**: 에러 로그에 상세 정보 출력

```python
# backend/hyperclova_client.py
token_kwargs = {}
if self.token:  # HF_TOKEN이 설정되어 있으면
    token_kwargs["token"] = self.token
    logger.info("Hugging Face 토큰 사용")

# 토크나이저와 모델 로드 시 토큰 자동 전달
self.tokenizer = AutoTokenizer.from_pretrained(
    self.model_name,
    trust_remote_code=True,
    **token_kwargs  # 토큰이 있으면 자동 포함
)
```

## 문제 해결

### 모델을 찾을 수 없음 (404 에러)
- **원인**: 모델 경로가 잘못되었거나 모델이 존재하지 않음
- **해결**: Hugging Face Hub에서 정확한 모델 경로 확인

### 인증 오류 (401 에러)
- **원인**: Private 모델인데 토큰이 없거나 잘못됨
- **해결**: 
  1. Hugging Face에서 토큰 발급
  2. `.env` 파일에 `HF_TOKEN` 설정
  3. 토큰 권한 확인 (read 권한 필요)

### 메모리 부족
- **원인**: 3B 모델이 1.5B보다 메모리를 더 많이 사용
- **해결**: 
  - `SEED_LOAD_IN_8BIT=true` 설정
  - 또는 `SEED_LOAD_IN_4BIT=true` 설정
  - Docker 메모리 제한 증가

## 체크리스트

- [ ] Hugging Face Hub에서 3B 모델 경로 확인
- [ ] 모델이 Public인지 Private인지 확인
- [ ] Private인 경우 토큰 발급 및 `.env`에 설정
- [ ] `.env` 파일에 `SEED_MODEL_PATH` 업데이트
- [ ] `SEED_LOAD_IN_8BIT=true` 설정 (메모리 절약)
- [ ] 백엔드 재시작 후 로그 확인

## 결론

**코드 수정은 필요 없습니다!** 

다음만 확인하세요:
1. 정확한 모델 경로 (`.env`의 `SEED_MODEL_PATH`)
2. 토큰 필요 여부 (`.env`의 `HF_TOKEN`)
3. 메모리 설정 (`.env`의 `SEED_LOAD_IN_8BIT`)

환경 변수만 올바르게 설정하면 바로 사용 가능합니다.


