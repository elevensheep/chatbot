import requests
import json
from typing import Dict, Any

class ChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def ask_question(self, question: str) -> Dict[str, Any]:
        """질문하기"""
        payload = {"question": question}
        
        response = requests.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()

def test_chat():
    """채팅 테스트"""
    client = ChatClient()
    
    print("HyperCLOVA X SEED 채팅 테스트")
    print("="*40)
    
    try:
        # 서버 연결 확인
        response = requests.get(f"{client.base_url}/")
        print(f"서버 상태: {response.json()['status']}")
        
        # 간단한 질문 테스트
        question = "안녕하세요!"
        print(f"\n질문: {question}")
        result = client.ask_question(question)
        print(f"답변: {result['answer']}")
        print(f"응답 시간: {result['response_time']:.2f}초")
        
        print("\n모든 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 실패: {e}")

def interactive_chat():
    """대화형 채팅"""
    client = ChatClient()
    
    print("HyperCLOVA X SEED 대화형 채팅")
    print("="*40)
    print("종료하려면 'quit' 또는 'exit'를 입력하세요")
    
    try:
        # 서버 연결 확인
        response = requests.get(f"{client.base_url}/")
        print(f"서버 상태: {response.json()['status']}")
        
        while True:
            question = input("\n질문: ")
            
            if question.lower() in ['quit', 'exit', '종료']:
                print("채팅을 종료합니다.")
                break
                
            if question.strip():
                try:
                    result = client.ask_question(question)
                    print(f"답변: {result['answer']}")
                    print(f"응답 시간: {result['response_time']:.2f}초")
                except Exception as e:
                    print(f"오류: {e}")
            else:
                print("질문을 입력해주세요.")
                
    except Exception as e:
        print(f"연결 실패: {e}")

if __name__ == "__main__":
    print("테스트 모드를 선택하세요:")
    print("1. 자동 테스트")
    print("2. 대화형 채팅")
    
    choice = input("선택 (1/2): ")
    
    if choice == "1":
        test_chat()
    elif choice == "2":
        interactive_chat()
    else:
        print("잘못된 선택입니다. 자동 테스트를 실행합니다.")
        test_chat()

