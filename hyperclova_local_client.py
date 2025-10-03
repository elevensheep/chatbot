import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any

class HyperCLOVALocalClient:
    def __init__(self, model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"):
        """
        HyperCLOVA X SEED 로컬 클라이언트 초기화
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """모델과 토크나이저 로드"""
        try:
            print(f"{self.model_name} 모델을 로드하는 중...")
            print(f"사용 디바이스: {self.device}")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print("모델 로드 완료!")
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return None, None
    
    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 0.3, 
                     top_p: float = 0.7, repetition_penalty: float = 1.05) -> Optional[str]:
        """
        텍스트 생성 - 짧고 간결한 답변만 생성
        """
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = self.load_model()
            
        if self.model is None or self.tokenizer is None:
            return None
            
        try:
            # 프롬프트 개선 - 더 명확한 지시 추가
            improved_prompt = f"질문: {prompt}\n답변:"
            
            # 입력 토큰화
            inputs = self.tokenizer(improved_prompt, return_tensors="pt").to(self.device)
            
            # 텍스트 생성 - 매우 짧게 제한
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=15,  # 매우 짧게 제한
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # 결과 디코딩
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 입력 프롬프트 제거하고 생성된 부분만 반환
            if generated_text.startswith(improved_prompt):
                generated_text = generated_text[len(improved_prompt):].strip()
            
            # 콤마나 불필요한 문자 제거
            generated_text = generated_text.lstrip(',.?!')
            
            # 빈 답변이면 기본 답변 제공
            if not generated_text.strip():
                if "안녕" in prompt or "안녕하세요" in prompt:
                    return "안녕하세요!"
                elif "수학" in prompt or "+" in prompt or "=" in prompt:
                    return "수학 문제를 풀어드릴게요."
                else:
                    return "무엇을 도와드릴까요?"
            
            # 긴 답변을 짧게 자르기 - 첫 번째 문장만
            sentences = generated_text.split('.')
            if len(sentences) > 1:
                generated_text = sentences[0] + '.'
            
            # 특정 키워드가 나타나면 그 앞에서 자르기
            stop_keywords = ['여행', '경험', '이야기', '이제', '그럼', '다음', '또한', '그리고', '또는', '하지만', '그러나', '그런데', '이런', '식으로', '풀']
            for keyword in stop_keywords:
                if keyword in generated_text:
                    generated_text = generated_text.split(keyword)[0].strip()
                    break
            
            # 너무 길면 강제로 자르기 (50자 제한)
            if len(generated_text) > 50:
                generated_text = generated_text[:50].strip()
                if not generated_text.endswith(('.', '!', '?')):
                    generated_text += '.'
                
            return generated_text
            
        except Exception as e:
            print(f"텍스트 생성 실패: {e}")
            return None
    
    def test_model(self) -> bool:
        """모델 테스트"""
        test_prompt = "안녕하세요. 간단한 인사말을 해주세요."
        result = self.generate_text(test_prompt, max_length=30)
        
        if result:
            print("HyperCLOVA X SEED 모델 테스트 성공!")
            print(f"응답: {result}")
            return True
        else:
            print("HyperCLOVA X SEED 모델 테스트 실패")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }
