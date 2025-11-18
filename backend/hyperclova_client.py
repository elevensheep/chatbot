"""
Hugging Face Transformers를 사용한 한국어 LLM 클라이언트 (비동기 버전)
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
from config import settings

logger = logging.getLogger(__name__)


class HyperCLOVAClient:
    """Hugging Face Transformers를 사용한 한국어 LLM 클라이언트"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        use_fp16: bool = None,
        token: str = None
    ):
        """
        Args:
            model_name: Hugging Face 모델 이름 (기본값: config에서 가져옴)
            device: 사용할 디바이스 ('cuda' 또는 'cpu', 기본값: config에서 가져옴)
            use_fp16: FP16 사용 여부 (기본값: config에서 가져옴)
            token: Hugging Face 토큰 (기본값: config에서 가져옴)
        """
        self.model_name = model_name or settings.HF_MODEL_NAME
        self.device = device or settings.HF_DEVICE
        self.use_fp16 = use_fp16 if use_fp16 is not None else settings.HF_USE_FP16
        self.token = token or settings.HF_TOKEN
        
        # CUDA 사용 가능 여부 확인
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
            self.device = "cpu"
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._model_loaded = False
        
        logger.info(f"Hugging Face 모델 클라이언트 초기화: {self.model_name}, 디바이스: {self.device}")
    
    async def _load_model(self):
        """모델과 토크나이저를 비동기로 로드"""
        if self._model_loaded:
            return
        
        # CPU에서 실행되는 경우 이벤트 루프를 블로킹하지 않도록 별도 스레드에서 실행
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)
    
    def _load_model_sync(self):
        """모델과 토크나이저를 동기적으로 로드 (별도 스레드에서 실행)"""
        try:
            logger.info(f"모델 로딩 시작: {self.model_name}")
            
            # 토큰 설정
            token_kwargs = {}
            if self.token:
                token_kwargs["token"] = self.token
                logger.info("Hugging Face 토큰 사용")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                **token_kwargs
            )
            
            # 모델 로드
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.use_fp16 and self.device == "cuda" else torch.float32,
            }
            
            # 토큰 추가
            if self.token:
                model_kwargs["token"] = self.token
            
            # 8-bit 또는 4-bit 양자화 설정
            if settings.HF_LOAD_IN_8BIT:
                model_kwargs["load_in_8bit"] = True
                logger.info("8-bit 양자화 사용")
            elif settings.HF_LOAD_IN_4BIT:
                model_kwargs["load_in_4bit"] = True
                logger.info("4-bit 양자화 사용")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # 양자화를 사용하지 않는 경우에만 디바이스로 이동
            if not settings.HF_LOAD_IN_8BIT and not settings.HF_LOAD_IN_4BIT:
                self.model = self.model.to(self.device)
            
            self.model.eval()  # 평가 모드로 설정
            
            self._model_loaded = True
            logger.info(f"모델 로딩 완료: {self.model_name}")
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}", exc_info=True)
            self._model_loaded = False
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """메시지 리스트를 프롬프트 형식으로 변환 (HyperCLOVAX SEED 형식)"""
        formatted = []
        system_content = None
        
        # 시스템 메시지 추출
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
                break
        
        # 시스템 프롬프트가 있으면 추가
        if system_content:
            formatted.append(f"### 지시사항:\n{system_content}\n\n")
        
        # 사용자와 어시스턴트 메시지 처리
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                continue  # 이미 처리함
            elif role == "user":
                formatted.append(f"### 질문:\n{content}\n\n")
            elif role == "assistant":
                formatted.append(f"### 답변:\n{content}\n\n")
        
        # 답변 시작 표시
        formatted.append("### 답변:\n")
        return "".join(formatted)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.5,
        top_k: int = 50,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        stop: List[str] = None,
        seed: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Hugging Face 모델을 사용한 채팅 생성 (비동기)
        
        Args:
            messages: 대화 메시지 리스트 [{"role": "system|user|assistant", "content": "..."}]
            max_tokens: 최대 생성 토큰 수 (기본값: 500)
            temperature: 생성 토큰 다양성 (기본값: 0.5)
            top_k: Top-K 샘플링 (기본값: 50)
            top_p: Top-P 샘플링 (기본값: 0.8)
            repetition_penalty: 반복 패널티 (기본값: 1.05)
            stop: 토큰 생성 중단 문자 (기본값: None)
            seed: 랜덤 시드 (기본값: None)
        
        Returns:
            API 응답 딕셔너리 (기존 형식과 호환)
        """
        # 모델이 로드되지 않았으면 로드
        if not self._model_loaded:
            logger.info("모델이 로드되지 않았습니다. 로딩을 시작합니다...")
            try:
                await self._load_model()
            except Exception as e:
                logger.error(f"모델 로딩 중 오류 발생: {e}", exc_info=True)
                raise RuntimeError(f"모델을 로드할 수 없습니다: {e}")
        
        # 프롬프트 생성
        prompt = self._format_messages(messages)
        
        # 토크나이징
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 시드 설정
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # 생성 파라미터 설정 (속도 최적화)
        # temperature가 낮으면 greedy decoding에 가까워져 더 빠름
        use_greedy = temperature < 0.1
        
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if not use_greedy else None,
            "top_k": top_k if not use_greedy else None,
            "top_p": top_p if not use_greedy else None,
            "repetition_penalty": repetition_penalty,
            "do_sample": not use_greedy and temperature > 0,
            "num_beams": 1,  # Greedy decoding (가장 빠름)
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # Stop 토큰 설정 (HyperCLOVAX SEED에 맞는 stop 토큰)
        stop_sequences = stop if stop else []
        # 기본 stop 시퀀스 추가 (반복 방지)
        default_stops = [
            "### 질문:", "### 답변:", "### 지시사항:",
            "### 사용자:", "### 시스템:", "### 어시스턴트:",
            "\n\n\n", "\n\n###", 
            "User:", "System:", "Assistant:",
            "<|endoftext|>", "<|end|>"
        ]
        stop_sequences.extend(default_stops)
        
        # Stop 시퀀스를 토큰 ID로 변환
        stop_token_ids = []
        if stop_sequences:
            try:
                for stop_seq in stop_sequences:
                    try:
                        stop_tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                        if stop_tokens:
                            stop_token_ids.extend(stop_tokens)
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Stop 토큰 변환 실패: {e}")
        
        # EOS 토큰도 stop 토큰에 추가
        if self.tokenizer.eos_token_id:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        
        if stop_token_ids:
            generation_config["eos_token_id"] = stop_token_ids
        
        # 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            outputs = await loop.run_in_executor(
                None,
                lambda: self.model.generate(inputs, **generation_config)
            )
        
        # 디코딩
        generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Stop 시퀀스 제거 (생성된 텍스트에서 stop 시퀀스 이후 부분 제거)
        # 가장 먼저 나타나는 stop 시퀀스를 기준으로 자름
        for stop_seq in stop_sequences:
            if stop_seq in generated_text:
                idx = generated_text.find(stop_seq)
                generated_text = generated_text[:idx]
                break
        
        # 앞뒤 공백 및 개행 제거
        generated_text = generated_text.strip()
        
        # 빈 답변이면 기본 메시지 반환
        if not generated_text:
            generated_text = "안녕하세요! 무엇을 도와드릴까요?"
        
        # 기존 API 응답 형식과 호환되도록 변환
        return {
            "result": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": generated_text.strip()
                        }
                    ]
                },
                "usage": {
                    "promptTokens": inputs.shape[1],
                    "completionTokens": outputs.shape[1] - inputs.shape[1],
                    "totalTokens": outputs.shape[1]
                }
            }
        }
    
    async def classify_intent(self, query: str) -> str:
        """
        사용자 질문의 의도 분류 (비동기)
        
        Args:
            query: 사용자 질문
            
        Returns:
            'course_related' 또는 'casual_chat'
        """
        system_prompt = """당신은 사용자 질문을 분류하는 AI입니다.
                        사용자의 질문을 다음 2가지로만 분류해주세요:

                        1. course_related: 수업계획서와 관련된 모든 질문
                        예시: "임석구 교수님", "임석구 교수님 연락처", "C언어프로그래밍 교수님", 
                                "데이터베이스 과제", "웹프로그래밍 수업시간", "캡스톤디자인 수업계획"

                        2. casual_chat: 일상 대화
                        예시: "안녕", "고마워", "날씨", "시간", "뭐해?"

                        답변은 반드시 다음 중 하나만 출력하세요:
                        - course_related (수업계획서 관련)
                        - casual_chat (일상 대화)"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 질문을 분류하세요: {query}"}
        ]
        
        try:
            response = await self.chat(messages=messages, max_tokens=20, temperature=0.1)
            
            # 응답 추출
            if "result" in response and "message" in response["result"]:
                message = response["result"]["message"]
                content = message.get("content", [])
                
                if isinstance(content, str):
                    result = content.strip().lower()
                elif isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict):
                        result = first_item.get("text", "").strip().lower()
                    else:
                        result = str(first_item).strip().lower()
                else:
                    result = ""
                
                # 응답에 'course' 또는 'related'가 포함되어 있으면 수업 관련
                if 'course' in result or 'related' in result or '수업' in result:
                    return 'course_related'
                else:
                    return 'casual_chat'
            
            # 기본값: 수업 관련으로 처리 (안전)
            return 'course_related'
            
        except Exception as e:
            logger.error(f"의도 분류 실패: {e}")
            # 오류 시 안전하게 수업 관련으로 처리
            return 'course_related'
    
    async def generate_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> str:
        """
        컨텍스트 기반 답변 생성 (비동기)
        
        Args:
            query: 사용자 질문
            context_docs: 검색된 컨텍스트 문서 리스트
            system_prompt: 시스템 프롬프트 (선택)
        
        Returns:
            생성된 답변 텍스트
        """
        # 기본 시스템 프롬프트
        if system_prompt is None:
            system_prompt = """당신은 대학교 수업계획서 기반 챗봇입니다.
                            제공된 수업계획서 정보를 바탕으로 학생들의 질문에 정확하고 도움이 되는 답변을 제공하세요.

                            답변 원칙:
                            1. 제공된 수업계획서 정보만을 바탕으로 답변
                            2. 질문의 의도를 정확히 파악하여 적절한 형태로 답변
                            3. 교수님 이름만 물어보면 해당 교수님의 모든 수업 목록을 보여주기
                            4. 연락처를 물어보면 교수님의 연락처 정보를 제공하기
                            5. 구체적인 수업을 물어보면 해당 수업의 상세 정보를 제공하기
                            6. 친근하고 도움이 되는 톤으로 답변
                            7. 정보가 부족하면 솔직하게 말하기

                            답변 형태 예시:
                            - 교수님 이름만 물어본 경우: "○○ 교수님의 수업은 다음과 같습니다: 1. 강의A 2. 강의B ..."
                            - 연락처를 물어본 경우: "○○ 교수님의 연락처는 010-xxxx-xxxx입니다."
                            - 구체적 수업을 물어본 경우: 해당 수업의 상세 정보 제공"""

        # 컨텍스트 구성
        context_text = "\n\n".join([
            f"[관련 정보 {i+1}]\n{doc.get('content', doc.get('page_content', ''))}"
            for i, doc in enumerate(context_docs)
        ])
        
        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""다음은 수업계획서에서 검색된 관련 정보입니다:

{context_text}

질문: {query}

위 정보를 바탕으로 질문에 답변해주세요."""}
        ]
        
        # API 호출
        response = await self.chat(
            messages=messages,
            max_tokens=300,  # 500 -> 300으로 줄여서 속도 향상
            temperature=0.3,  # 0.5 -> 0.3으로 낮춰서 더 빠른 생성
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.05
        )
        
        # 답변 추출
        try:
            if "result" in response and "message" in response["result"]:
                message = response["result"]["message"]
                content = message.get("content", [])
                
                # content가 문자열인 경우
                if isinstance(content, str):
                    return content
                
                # content가 배열인 경우
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    # 딕셔너리인 경우
                    if isinstance(first_item, dict):
                        return first_item.get("text", "")
                    # 문자열인 경우
                    elif isinstance(first_item, str):
                        return first_item
                
                # 기타 경우
                logger.warning(f"예상치 못한 content 형식: {type(content)}, 값: {content}")
                return str(content) if content else ""
            else:
                logger.error(f"예상치 못한 응답 형식: {response}")
                raise ValueError("API 응답에서 답변을 추출할 수 없습니다")
        except Exception as e:
            logger.error(f"응답 파싱 오류: {e}")
            raise
    
    async def generate_casual_answer(self, query: str) -> str:
        """
        일상 대화 답변 생성 (컨텍스트 없이, 비동기)
        
        Args:
            query: 사용자 질문
            
        Returns:
            생성된 답변 텍스트
        """
        system_prompt = """당신은 친근하고 도움이 되는 대학교 수업 안내 챗봇입니다.
                        학생들과 자연스럽게 대화하며, 필요한 경우 수업계획서 관련 질문을 하도록 안내해주세요.

                        답변 원칙:
                        1. 친근하고 자연스러운 한국어로 답변
                        2. 간단명료하게 답변
                        3. 수업 관련 질문이 있다면 구체적으로 물어보도록 유도"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = await self.chat(
                messages=messages,
                max_tokens=150,  # 200 -> 150으로 줄여서 속도 향상
                temperature=0.3,  # 0.7 -> 0.3으로 낮춰서 더 빠른 생성
                top_p=0.9
            )
            
            # 응답 추출
            if "result" in response and "message" in response["result"]:
                message = response["result"]["message"]
                content = message.get("content", [])
                
                if isinstance(content, str):
                    return content
                elif isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict):
                        return first_item.get("text", "")
                    elif isinstance(first_item, str):
                        return first_item
                
                return str(content) if content else "안녕하세요! 무엇을 도와드릴까요?"
            
            return "안녕하세요! 무엇을 도와드릴까요?"
            
        except Exception as e:
            logger.error(f"일상 대화 답변 생성 실패: {e}")
            return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."


# 전역 HyperCLOVA 클라이언트 인스턴스
_hyperclova_client = None


def get_hyperclova_client() -> HyperCLOVAClient:
    """HyperCLOVA 클라이언트 싱글톤 인스턴스 반환"""
    global _hyperclova_client
    if _hyperclova_client is None:
        _hyperclova_client = HyperCLOVAClient()
    return _hyperclova_client
