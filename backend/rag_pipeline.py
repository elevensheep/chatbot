"""
RAG (Retrieval-Augmented Generation) 파이프라인
벡터 검색과 HyperCLOVA X를 결합하여 질의응답 시스템 구현
"""
from typing import Optional, Dict, Any, List
from vector_search import VectorSearch
from hyperclova import HyperClovaClient
import logging

# 로깅 설정
logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG 파이프라인 클래스"""
    
    # 기본 시스템 프롬프트
    DEFAULT_SYSTEM_PROMPT = """당신은 대학교 수업계획서 정보를 제공하는 친절한 AI 도우미입니다.

역할과 지침:
1. 제공된 수업계획서 정보를 바탕으로 정확하고 구체적으로 답변합니다.
2. 정보에 없는 내용은 추측하지 않고, "제공된 정보에서는 해당 내용을 찾을 수 없습니다"라고 솔직하게 답변합니다.
3. 학생들이 이해하기 쉽도록 친절하고 명확하게 설명합니다.
4. 교수님 연락처, 수업 일정, 평가 방법 등 구체적인 정보를 제공할 때는 정확성을 최우선으로 합니다.
5. 한국어로 자연스럽게 답변합니다."""
    
    def __init__(
        self,
        vector_search: Optional[VectorSearch] = None,
        hyperclova_client: Optional[HyperClovaClient] = None,
        top_k: int = 3
    ):
        """
        Args:
            vector_search: VectorSearch 인스턴스
            hyperclova_client: HyperClovaClient 인스턴스
            top_k: 검색할 문서 개수
        """
        self.vector_search = vector_search
        self.hyperclova_client = hyperclova_client
        self.top_k = top_k
        
        # 인스턴스가 제공되지 않으면 기본값 사용
        if self.vector_search is None:
            from vector_search import get_vector_search
            try:
                self.vector_search = get_vector_search()
                logger.info("✅ VectorSearch 로드 완료")
            except Exception as e:
                logger.error(f"❌ VectorSearch 로드 실패: {e}")
                raise
        
        if self.hyperclova_client is None:
            from hyperclova import get_hyperclova_client
            try:
                self.hyperclova_client = get_hyperclova_client()
                logger.info("✅ HyperCLOVA X 클라이언트 로드 완료")
            except Exception as e:
                logger.error(f"❌ HyperCLOVA X 클라이언트 로드 실패: {e}")
                raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        쿼리와 관련된 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 검색할 문서 개수 (None이면 기본값 사용)
            
        Returns:
            검색 결과 리스트
        """
        k = top_k if top_k is not None else self.top_k
        return self.vector_search.search(query, top_k=k)
    
    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        검색 결과를 컨텍스트 문자열로 포맷팅합니다.
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            포맷팅된 컨텍스트
        """
        if not search_results:
            return "관련 정보를 찾을 수 없습니다."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            subject = result["metadata"].get("subject", "알 수 없음")
            doc_type = result["metadata"].get("type", "")
            text = result["text"]
            similarity = result.get("similarity", 0)
            
            context_parts.append(
                f"[참고문서 {i}] ({doc_type}) - {subject} (관련도: {similarity:.2f})\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        컨텍스트를 바탕으로 답변을 생성합니다.
        
        Args:
            query: 사용자 질문
            context: 검색된 컨텍스트
            system_prompt: 시스템 프롬프트 (None이면 기본값 사용)
            **generation_kwargs: HyperCLOVA X 생성 옵션
            
        Returns:
            생성된 답변
        """
        prompt = system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT
        
        return self.hyperclova_client.chat(
            user_message=query,
            system_prompt=prompt,
            context=context,
            **generation_kwargs
        )
    
    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
        return_sources: bool = False,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        질문에 대한 답변을 생성합니다. (전체 RAG 파이프라인)
        
        Args:
            query: 사용자 질문
            top_k: 검색할 문서 개수
            system_prompt: 시스템 프롬프트
            return_sources: 출처 정보 포함 여부
            **generation_kwargs: HyperCLOVA X 생성 옵션
            
        Returns:
            {
                "answer": "생성된 답변",
                "sources": [...],  # return_sources=True인 경우
                "query": "원본 질문"
            }
        """
        logger.info(f"📝 질문: {query}")
        
        # 1. 관련 문서 검색
        logger.info(f"🔍 벡터 검색 중... (top_k={top_k or self.top_k})")
        search_results = self.retrieve(query, top_k)
        logger.info(f"✅ {len(search_results)}개 문서 검색됨")
        
        # 2. 컨텍스트 생성
        context = self.format_context(search_results)
        logger.debug(f"📄 컨텍스트 길이: {len(context)} 문자")
        
        # 3. 답변 생성
        logger.info("🤖 답변 생성 중...")
        answer = self.generate_answer(
            query=query,
            context=context,
            system_prompt=system_prompt,
            **generation_kwargs
        )
        logger.info("✅ 답변 생성 완료")
        
        # 4. 결과 구성
        result = {
            "answer": answer,
            "query": query
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "subject": r["metadata"].get("subject", ""),
                    "type": r["metadata"].get("type", ""),
                    "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "similarity": r.get("similarity", 0)
                }
                for r in search_results
            ]
        
        return result
    
    def answer_stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        스트리밍 방식으로 답변을 생성합니다.
        (현재 HyperCLOVA X API는 스트리밍을 지원하지만, 간단한 구현을 위해 일반 답변 사용)
        
        Args:
            query: 사용자 질문
            top_k: 검색할 문서 개수
            system_prompt: 시스템 프롬프트
            
        Yields:
            답변 청크
        """
        # 간단한 구현: 전체 답변을 한 번에 반환
        result = self.answer(query, top_k, system_prompt, return_sources=False)
        yield result["answer"]


# 싱글톤 인스턴스
_rag_pipeline_instance = None


def get_rag_pipeline() -> RAGPipeline:
    """
    RAGPipeline 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        RAGPipeline 인스턴스
    """
    global _rag_pipeline_instance
    
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    
    return _rag_pipeline_instance


if __name__ == "__main__":
    """테스트 코드"""
    import os
    from dotenv import load_dotenv
    
    # 환경 변수 로드
    load_dotenv()
    
    print("=" * 60)
    print("🚀 RAG 파이프라인 테스트")
    print("=" * 60)
    
    # API 키 확인
    if not os.getenv("NAVER_CLOVA_API_KEY"):
        print("⚠️  경고: NAVER_CLOVA_API_KEY가 설정되지 않았습니다.")
        print("   벡터 검색만 테스트합니다.\n")
        
        # 벡터 검색만 테스트
        from vector_search import VectorSearch
        vs = VectorSearch()
        vs.load()
        
        test_query = "C언어 수업은 누가 가르치나요?"
        print(f"📝 질문: {test_query}\n")
        
        results = vs.search(test_query, top_k=3)
        for r in results:
            print(f"[{r['rank']}] {r['metadata']['subject']}")
            print(f"   유사도: {r['similarity']:.3f}")
            print(f"   내용: {r['text'][:100]}...\n")
    else:
        try:
            # RAG 파이프라인 생성
            rag = RAGPipeline()
            
            # 테스트 질문들
            test_queries = [
                "C언어프로그래밍 수업은 누가 가르치나요?",
                "프로그래밍 과목의 평가 방법은 무엇인가요?",
                "1주차에는 무엇을 배우나요?"
            ]
            
            for query in test_queries:
                print(f"\n{'=' * 60}")
                print(f"📝 질문: {query}")
                print('=' * 60)
                
                # 답변 생성
                result = rag.answer(query, return_sources=True)
                
                print(f"\n🤖 답변:\n{result['answer']}")
                
                print(f"\n📚 참고 문서:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  [{i}] {source['subject']} ({source['type']})")
                    print(f"      유사도: {source['similarity']:.3f}")
            
            print("\n" + "=" * 60)
            print("✅ 테스트 완료")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()

