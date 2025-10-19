"""
벡터 검색 유틸리티
FAISS 인덱스를 로드하고 유사도 검색을 수행합니다.
"""
import pickle
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorSearch:
    """벡터 검색 클래스"""
    
    def __init__(self, index_dir: str = "vector_db"):
        """
        Args:
            index_dir: FAISS 인덱스가 저장된 디렉토리
        """
        self.index_dir = Path(index_dir)
        self.index = None
        self.chunks = None
        self.model = None
        
    def load(self):
        """FAISS 인덱스와 메타데이터를 로드합니다."""
        index_file = self.index_dir / "faiss_index.bin"
        metadata_file = self.index_dir / "chunks_metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS 인덱스를 찾을 수 없습니다: {index_file}\n"
                f"먼저 vectorize_data.py를 실행하여 인덱스를 생성하세요."
            )
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"메타데이터를 찾을 수 없습니다: {metadata_file}"
            )
        
        # FAISS 인덱스 로드
        self.index = faiss.read_index(str(index_file))
        print(f"✅ FAISS 인덱스 로드 완료: {self.index.ntotal}개 벡터")
        
        # 메타데이터 로드
        with open(metadata_file, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"✅ 메타데이터 로드 완료: {len(self.chunks)}개 청크")
        
        # 임베딩 모델 로드 (벡터화할 때 사용한 것과 동일한 모델)
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"📦 임베딩 모델 로딩 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ 모델 로드 완료")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 유사한 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
            
        Returns:
            검색 결과 리스트 (텍스트, 메타데이터, 유사도 점수 포함)
        """
        if self.index is None or self.model is None:
            raise RuntimeError("먼저 load() 메서드를 호출하여 인덱스를 로드하세요.")
        
        # 쿼리를 벡터로 변환
        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = query_vector.astype('float32')
        
        # FAISS로 검색
        distances, indices = self.index.search(query_vector, top_k)
        
        # 결과 포맷팅
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "rank": i + 1,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "distance": float(distance),
                    "similarity": float(1 / (1 + distance))  # 유사도 점수 (0~1)
                })
        
        return results
    
    def search_by_subject(self, query: str, subject_filter: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        특정 교과목으로 필터링하여 검색합니다.
        
        Args:
            query: 검색 쿼리
            subject_filter: 필터링할 교과목 이름 (None이면 전체 검색)
            top_k: 반환할 상위 결과 개수
            
        Returns:
            검색 결과 리스트
        """
        # 먼저 전체 검색
        all_results = self.search(query, top_k * 3)  # 필터링을 고려해 더 많이 검색
        
        if subject_filter:
            # 교과목으로 필터링
            filtered_results = [
                r for r in all_results 
                if subject_filter.lower() in r["metadata"].get("subject", "").lower()
            ]
            return filtered_results[:top_k]
        
        return all_results[:top_k]
    
    def get_context_for_llm(self, query: str, top_k: int = 3) -> str:
        """
        LLM에 전달할 컨텍스트를 생성합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 사용할 상위 결과 개수
            
        Returns:
            포맷팅된 컨텍스트 문자열
        """
        results = self.search(query, top_k)
        
        if not results:
            return "관련 정보를 찾을 수 없습니다."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            subject = result["metadata"].get("subject", "알 수 없음")
            text = result["text"]
            context_parts.append(f"[문서 {i}] {subject}\n{text}\n")
        
        return "\n".join(context_parts)


# 싱글톤 인스턴스 (FastAPI에서 재사용)
_vector_search_instance = None


def get_vector_search(index_dir: str = None) -> VectorSearch:
    """
    VectorSearch 싱글톤 인스턴스를 반환합니다.
    
    Args:
        index_dir: FAISS 인덱스 디렉토리 (None이면 기본값 사용)
        
    Returns:
        VectorSearch 인스턴스
    """
    global _vector_search_instance
    
    if _vector_search_instance is None:
        if index_dir is None:
            index_dir = Path(__file__).parent / "vector_db"
        
        _vector_search_instance = VectorSearch(index_dir)
        _vector_search_instance.load()
    
    return _vector_search_instance


if __name__ == "__main__":
    """테스트 코드"""
    print("=" * 60)
    print("🔍 벡터 검색 테스트")
    print("=" * 60)
    
    # VectorSearch 인스턴스 생성
    vs = VectorSearch()
    vs.load()
    
    # 테스트 쿼리
    test_queries = [
        "C언어 수업은 누가 가르치나요?",
        "프로그래밍 과목의 평가 방법은?",
        "1주차에는 무엇을 배우나요?"
    ]
    
    for query in test_queries:
        print(f"\n📝 질문: {query}")
        print("-" * 60)
        
        results = vs.search(query, top_k=3)
        
        for result in results:
            print(f"\n[순위 {result['rank']}] 유사도: {result['similarity']:.3f}")
            print(f"교과목: {result['metadata'].get('subject', 'N/A')}")
            print(f"타입: {result['metadata'].get('type', 'N/A')}")
            print(f"내용: {result['text'][:200]}...")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료")
    print("=" * 60)

