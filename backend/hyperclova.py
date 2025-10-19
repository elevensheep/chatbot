"""
HyperCLOVA X API 연동 모듈
네이버 클라우드 플랫폼의 HyperCLOVA X API를 사용하여 텍스트 생성
"""
import os
import requests
import json
from typing import Optional, Dict, Any


class HyperClovaClient:
    """HyperCLOVA X API 클라이언트"""
    
    # HyperCLOVA X API 엔드포인트
    API_URL = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-DASH-001"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        api_key_primary_val: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Args:
            api_key: NCP API Key (X-NCP-CLOVASTUDIO-API-KEY)
            api_key_primary_val: NCP API Key Primary Value (X-NCP-APIGW-API-KEY)
            request_id: 요청 ID (X-NCP-CLOVASTUDIO-REQUEST-ID, 선택사항)
        """
        self.api_key = api_key or os.getenv("NAVER_CLOVA_API_KEY")
        self.api_key_primary_val = api_key_primary_val or os.getenv("NAVER_CLOVA_API_KEY_PRIMARY")
        self.request_id = request_id or os.getenv("NAVER_CLOVA_REQUEST_ID", "default-request-id")
        
        if not self.api_key or not self.api_key_primary_val:
            raise ValueError(
                "HyperCLOVA X API 키가 설정되지 않았습니다.\n"
                "환경 변수 NAVER_CLOVA_API_KEY와 NAVER_CLOVA_API_KEY_PRIMARY를 설정하세요."
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        return {
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
            "X-NCP-APIGW-API-KEY": self.api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream"
        }
    
    def generate(
        self,
        messages: list,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.8,
        repeat_penalty: float = 1.2,
        stop_before: Optional[list] = None,
        include_ai_filters: bool = True
    ) -> str:
        """
        HyperCLOVA X로 텍스트 생성
        
        Args:
            messages: 대화 메시지 리스트 [{"role": "system|user|assistant", "content": "..."}]
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 (0.0~1.0, 높을수록 다양)
            top_k: Top-K 샘플링
            top_p: Top-P (nucleus) 샘플링
            repeat_penalty: 반복 페널티
            stop_before: 생성 중지 토큰 리스트
            include_ai_filters: AI 필터 사용 여부
            
        Returns:
            생성된 텍스트
        """
        # 요청 데이터 구성
        request_data = {
            "messages": messages,
            "topP": top_p,
            "topK": top_k,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "repeatPenalty": repeat_penalty,
            "stopBefore": stop_before or [],
            "includeAiFilters": include_ai_filters
        }
        
        try:
            # API 호출
            response = requests.post(
                self.API_URL,
                headers=self._get_headers(),
                json=request_data,
                stream=True
            )
            
            response.raise_for_status()
            
            # 스트리밍 응답 처리
            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    
                    # SSE 형식 파싱
                    if decoded_line.startswith('data:'):
                        data_str = decoded_line[5:].strip()
                        
                        if data_str and data_str != '[DONE]':
                            try:
                                data = json.loads(data_str)
                                
                                # 메시지 추출
                                if 'message' in data and 'content' in data['message']:
                                    full_response += data['message']['content']
                                
                            except json.JSONDecodeError:
                                continue
            
            return full_response.strip()
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HyperCLOVA X API 호출 실패: {str(e)}")
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        간단한 채팅 인터페이스
        
        Args:
            user_message: 사용자 메시지
            system_prompt: 시스템 프롬프트 (선택)
            context: 컨텍스트 정보 (RAG에서 검색된 문서 등)
            **kwargs: generate() 메서드의 추가 인자
            
        Returns:
            AI 응답
        """
        messages = []
        
        # 시스템 프롬프트 추가
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 컨텍스트가 있으면 사용자 메시지에 포함
        if context:
            user_content = f"""다음은 참고할 수 있는 관련 정보입니다:

{context}

질문: {user_message}

위 정보를 바탕으로 질문에 답변해주세요. 정보에 없는 내용은 추측하지 말고, 모른다고 답변하세요."""
        else:
            user_content = user_message
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return self.generate(messages, **kwargs)


# 싱글톤 인스턴스
_hyperclova_instance = None


def get_hyperclova_client() -> HyperClovaClient:
    """
    HyperClovaClient 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        HyperClovaClient 인스턴스
    """
    global _hyperclova_instance
    
    if _hyperclova_instance is None:
        _hyperclova_instance = HyperClovaClient()
    
    return _hyperclova_instance


if __name__ == "__main__":
    """테스트 코드"""
    import os
    from dotenv import load_dotenv
    
    # 환경 변수 로드
    load_dotenv()
    
    print("=" * 60)
    print("🤖 HyperCLOVA X API 테스트")
    print("=" * 60)
    
    # API 키 확인
    if not os.getenv("NAVER_CLOVA_API_KEY"):
        print("❌ 오류: NAVER_CLOVA_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 API 키를 설정하세요.")
        exit(1)
    
    try:
        # 클라이언트 생성
        client = HyperClovaClient()
        
        # 테스트 질문
        test_query = "안녕하세요. 당신은 누구인가요?"
        
        print(f"\n📝 질문: {test_query}")
        print("-" * 60)
        
        # 응답 생성
        response = client.chat(
            user_message=test_query,
            system_prompt="당신은 대학교 수업계획서 정보를 제공하는 친절한 AI 도우미입니다."
        )
        
        print(f"🤖 응답: {response}")
        
        print("\n" + "=" * 60)
        print("✅ 테스트 완료")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")

