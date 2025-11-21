"""
HyperCLOVA X API 클라이언트 (비동기 버전)
"""
import httpx
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger(__name__)


class HyperCLOVAClient:
    """HyperCLOVA X API 클라이언트 (비동기)"""
    
    # HyperCLOVA X Chat Completions API 엔드포인트
    # v3 API 사용, HCX-005 모델
    HOST = "https://clovastudio.stream.ntruss.com"
    API_ENDPOINT = "/v3/chat-completions/HCX-005"
    
    def __init__(
        self,
        api_key: str = None,
        api_gateway_key: str = None,
        request_id: str = None
    ):
        """
        Args:
            api_key: HyperCLOVA X API 키
            api_gateway_key: API Gateway 키 (선택)
            request_id: 요청 ID (선택)
        """
        self.api_key = api_key or settings.HYPERCLOVA_API_KEY
        self.api_gateway_key = api_gateway_key or settings.HYPERCLOVA_API_GATEWAY_KEY
        self.request_id = request_id or settings.HYPERCLOVA_REQUEST_ID
        
        if not self.api_key:
            raise ValueError("HyperCLOVA API 키가 설정되지 않았습니다")
        
        # 비동기 클라이언트는 메서드 호출 시 생성 (연결 풀 공유)
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """비동기 HTTP 클라이언트 반환 (재사용)"""
        if self._client is None:
            # 재시도 로직과 타임아웃 설정
            timeout = httpx.Timeout(30.0, connect=10.0)
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            
            self._client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits
            )
        return self._client
    
    async def _close_client(self):
        """클라이언트 종료"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    def _build_headers(self) -> Dict[str, str]:
        """API 요청 헤더 생성 (HyperCLOVA X v3 표준 형식)"""
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json"
        }
        
        # Authorization: Bearer 형식
        if self.api_key.startswith('Bearer '):
            headers["Authorization"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Request ID (선택적)
        if self.request_id:
            headers["X-NCP-CLOVASTUDIO-REQUEST-ID"] = self.request_id
        
        return headers
    
    def _convert_messages_to_v3_format(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """메시지를 v3 API 형식으로 변환"""
        converted = []
        for msg in messages:
            converted.append({
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"]
                    }
                ]
            })
        return converted
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.5,
        top_k: int = 99,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
        stop: List[str] = None,
        seed: int = 0,
        include_ai_filters: bool = True
    ) -> Dict[str, Any]:
        """
        HyperCLOVA X v3 Chat Completions API 호출 (비동기)
        
        Args:
            messages: 대화 메시지 리스트 [{"role": "system|user|assistant", "content": "..."}]
            max_tokens: 최대 생성 토큰 수 (기본값: 500)
            temperature: 생성 토큰 다양성 (0.00~1.00, 기본값: 0.5)
            top_k: Top-K 샘플링 (0~128, 기본값: 99)
            top_p: Top-P 샘플링 (0~1, 기본값: 0.8)
            repetition_penalty: 반복 패널티 (0.0~2.0, 기본값: 1.05)
            stop: 토큰 생성 중단 문자 (기본값: [])
            seed: 결과 일관성 수준 (0~4294967295, 기본값: 0)
            include_ai_filters: AI 필터 포함 여부 (기본값: true)
        
        Returns:
            API 응답 딕셔너리
        """
        max_retries = 3
        backoff_factor = 0.5
        
        for attempt in range(max_retries):
            try:
                client = await self._get_client()
                headers = self._build_headers()
                
                # v3 API 형식으로 메시지 변환
                v3_messages = self._convert_messages_to_v3_format(messages)
                
                # HyperCLOVA X v3 API 형식
                payload = {
                    "messages": v3_messages,
                    "topP": top_p,
                    "topK": top_k,
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "repetitionPenalty": repetition_penalty,
                    "stop": stop if stop else [],
                    "seed": seed,
                    "includeAiFilters": include_ai_filters
                }
                
                url = self.HOST + self.API_ENDPOINT
                
                logger.info(f"HyperCLOVA X v3 API 호출 중... (메시지 수: {len(messages)}, 시도: {attempt + 1})")
                logger.debug(f"URL: {url}")
                logger.debug(f"요청 헤더: {headers}")
                
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info("HyperCLOVA X API 호출 성공")
                return result
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.error(f"HyperCLOVA X API 호출 실패 (HTTP {status_code}): {e}")
                
                # 재시도 가능한 상태 코드
                if status_code in [408, 429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.info(f"{wait_time}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # 재시도 불가능하거나 마지막 시도
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"응답 내용: {e.response.text}")
                raise
                
            except httpx.RequestError as e:
                logger.error(f"HyperCLOVA X API 네트워크 오류: {e}")
                
                # 네트워크 오류는 재시도
                if attempt < max_retries - 1:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.info(f"{wait_time}초 후 재시도...")
                    await asyncio.sleep(wait_time)
                    continue
                raise
                
            except Exception as e:
                logger.error(f"HyperCLOVA X API 예상치 못한 오류: {e}")
                raise
        
        # 모든 재시도 실패
        raise Exception("HyperCLOVA X API 호출이 최대 재시도 횟수를 초과했습니다")
    
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
            response = await self.chat(messages=messages, max_tokens=10, temperature=0.1)
            
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
        system_prompt: str = None,
        message_history: List[Dict[str, str]] = None
    ) -> str:
        """
        컨텍스트 기반 답변 생성 (비동기)
        
        Args:
            query: 사용자 질문
            context_docs: 검색된 컨텍스트 문서 리스트
            system_prompt: 시스템 프롬프트 (선택)
            message_history: 최근 대화 히스토리 (선택, 최대 3개)
        
        Returns:
            생성된 답변 텍스트
        """
        # 기본 시스템 프롬프트 (개선된 버전)
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
                            8. 이전 대화 내용을 참고하여 맥락에 맞는 답변 제공

                            답변 형태 예시:
                            - 교수님 이름만 물어본 경우: "○○ 교수님의 수업은 다음과 같습니다: 1. 강의A 2. 강의B ..."
                            - 연락처를 물어본 경우: "○○ 교수님의 연락처는 010-xxxx-xxxx입니다."
                            - 구체적 수업을 물어본 경우: 해당 수업의 상세 정보 제공"""

        # 컨텍스트 구성
        context_text = "\n\n".join([
            f"[관련 정보 {i+1}]\n{doc.get('content', doc.get('page_content', ''))}"
            for i, doc in enumerate(context_docs)
        ])
        
        # 메시지 구성 (시스템 프롬프트 + 이전 대화 히스토리 + 현재 질문)
        messages = [{"role": "system", "content": system_prompt}]
        
        # 이전 대화 히스토리 추가 (있는 경우)
        if message_history and len(message_history) > 0:
            for hist_msg in message_history:
                messages.append({
                    "role": hist_msg["role"],
                    "content": hist_msg["content"]
                })
            logger.info(f"대화 히스토리 {len(message_history)}개를 컨텍스트에 포함")
        
        # 현재 질문과 컨텍스트 추가
        user_content = f"""다음은 수업계획서에서 검색된 관련 정보입니다:

{context_text}

질문: {query}

위 정보를 바탕으로 질문에 답변해주세요."""
        
        messages.append({"role": "user", "content": user_content})
        
        # API 호출
        response = await self.chat(
            messages=messages,
            max_tokens=500,
            temperature=0.5,
            top_p=0.8,
            top_k=99,
            repetition_penalty=1.05
        )
        
        # 답변 추출 (HyperCLOVA X v3 응답 형식)
        # 응답 형식: {"status": {...}, "result": {"message": {"role": "assistant", "content": [{"type": "text", "text": "..."}]}, "usage": {...}}}
        try:
            if "result" in response and "message" in response["result"]:
                message = response["result"]["message"]
                content = message.get("content", [])
                
                # content가 문자열인 경우 (일부 API 버전)
                if isinstance(content, str):
                    return content
                
                # content가 배열인 경우 (v3 표준)
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
            logger.error(f"전체 응답: {json.dumps(response, ensure_ascii=False, indent=2)}")
            raise
    
    async def generate_casual_answer(self, query: str, message_history: List[Dict[str, str]] = None) -> str:
        """
        일상 대화 답변 생성 (컨텍스트 없이, 비동기)
        
        Args:
            query: 사용자 질문
            message_history: 최근 대화 히스토리 (선택, 최대 3개)
            
        Returns:
            생성된 답변 텍스트
        """
        system_prompt = """당신은 친근하고 도움이 되는 대학교 수업 안내 챗봇입니다.
                        학생들과 자연스럽게 대화하며, 필요한 경우 수업계획서 관련 질문을 하도록 안내해주세요.

                        답변 원칙:
                        1. 친근하고 자연스러운 한국어로 답변
                        2. 간단명료하게 답변
                        3. 수업 관련 질문이 있다면 구체적으로 물어보도록 유도
                        4. 이전 대화 내용을 참고하여 맥락에 맞는 답변 제공"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # 이전 대화 히스토리 추가 (있는 경우)
        if message_history and len(message_history) > 0:
            for hist_msg in message_history:
                messages.append({
                    "role": hist_msg["role"],
                    "content": hist_msg["content"]
                })
            logger.info(f"일상 대화 - 히스토리 {len(message_history)}개를 컨텍스트에 포함")
        
        # 현재 질문 추가
        messages.append({"role": "user", "content": query})
        
        try:
            response = await self.chat(
                messages=messages,
                max_tokens=200,
                temperature=0.7,
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
    
    def _generate_mock_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Mock 응답 생성 (API 키 문제 해결 전까지 임시 사용)
        FAISS 검색 결과를 바탕으로 간단한 답변 생성
        """
        if not context_docs:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."
        
        # 가장 유사도가 높은 문서 사용
        top_doc = context_docs[0]
        metadata = top_doc.get('metadata', {})
        content = top_doc.get('content', '')
        
        course_name = metadata.get('course_name', '').strip()
        professor = metadata.get('professor', '').strip()
        section = metadata.get('section', '').strip()
        
        # 질문 유형 파악
        query_lower = query.lower()
        
        # 교수 관련 질문
        if '교수' in query or '선생님' in query or '담당' in query:
            if professor and course_name:
                return f"{course_name}은(는) {professor} 교수님이 담당하고 계십니다.\n\n[검색된 정보]\n{content[:300]}..."
            elif professor:
                return f"{professor} 교수님에 대한 정보를 찾았습니다.\n\n[검색된 정보]\n{content[:300]}..."
        
        # 과목/강의 관련 질문
        if '과목' in query or '강의' in query or '수업' in query:
            if course_name and professor:
                return f"{course_name}에 대한 정보입니다.\n담당교수: {professor} 교수님\n\n[검색된 정보]\n{content[:300]}..."
        
        # 목표 관련 질문
        if '목표' in query or '목적' in query:
            if '수업목표' in content or '교과목개요' in content:
                # 수업목표 부분 추출
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '수업목표' in line:
                        goal_text = '\n'.join(lines[i:min(i+5, len(lines))])
                        return f"{course_name}의 수업 목표:\n\n{goal_text}"
        
        # 과제 관련 질문
        if '과제' in query or '숙제' in query:
            if section == '과제':
                return f"{course_name}의 과제 정보:\n\n{content[:400]}..."
        
        # 기본 응답
        answer = f"질문하신 '{query}'에 대한 정보를 찾았습니다.\n\n"
        if course_name and professor:
            answer += f"**강의:** {course_name}\n**담당교수:** {professor} 교수님\n**섹션:** {section}\n\n"
        answer += f"[검색된 정보]\n{content[:400]}..."
        
        return answer


# 전역 HyperCLOVA 클라이언트 인스턴스
_hyperclova_client = None


def get_hyperclova_client() -> HyperCLOVAClient:
    """HyperCLOVA 클라이언트 싱글톤 인스턴스 반환"""
    global _hyperclova_client
    if _hyperclova_client is None:
        _hyperclova_client = HyperCLOVAClient()
    return _hyperclova_client

