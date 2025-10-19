"""
데이터 벡터화 및 FAISS 인덱스 생성 스크립트
수업계획서 JSON 데이터를 벡터화하여 FAISS 인덱스로 저장합니다.
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DataVectorizer:
    """데이터 벡터화 클래스"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: 사용할 임베딩 모델 (한국어 지원 모델)
        """
        print(f"📦 임베딩 모델 로딩 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ 모델 로드 완료 (차원: {self.dimension})")
        
    def extract_text_from_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        JSON 데이터에서 텍스트 청크를 추출합니다.
        
        Args:
            data: 수업계획서 JSON 데이터
            
        Returns:
            텍스트 청크 리스트 (메타데이터 포함)
        """
        chunks = []
        
        # _metadata는 건너뜀
        for subject_name, subject_data in data.items():
            if subject_name == "_metadata":
                continue
                
            # 교과목 운영 정보
            if "교과목 운영" in subject_data:
                운영정보 = subject_data["교과목 운영"]
                text = f"""교과목: {운영정보.get('교과목', '')}
담당교수: {운영정보.get('담당교수', '')}
이수구분: {운영정보.get('이수구분', '')}
시간/학점: {운영정보.get('시간/학점', '')}
이론/실습: {운영정보.get('이론/실습', '')}
연락처: {운영정보.get('연락처', '')}
이메일: {운영정보.get('E-Mail', '')}"""
                
                chunks.append({
                    "text": text,
                    "metadata": {
                        "subject": subject_name,
                        "type": "교과목_운영",
                        "professor": 운영정보.get('담당교수', '')
                    }
                })
            
            # 교과목 개요
            if "교과목 개요" in subject_data:
                개요 = subject_data["교과목 개요"]
                text_parts = []
                
                for key, value in 개요.items():
                    if value and value != "NaN" and not isinstance(value, dict):
                        text_parts.append(f"{key}: {value}")
                
                if text_parts:
                    chunks.append({
                        "text": "\n".join(text_parts),
                        "metadata": {
                            "subject": subject_name,
                            "type": "교과목_개요"
                        }
                    })
            
            # 수업계획 (주차별)
            if "수업계획" in subject_data:
                수업계획 = subject_data["수업계획"]
                for week, week_data in 수업계획.items():
                    if isinstance(week_data, dict):
                        text = f"""주차: {week}
수업주제 및 내용: {week_data.get('수업주제 및 내용', '')}
수업방법: {week_data.get('수업방법', '')}
학생성장(역량제고) 전략: {week_data.get('학생성장(역량제고) 전략', '')}"""
                        
                        chunks.append({
                            "text": text,
                            "metadata": {
                                "subject": subject_name,
                                "type": "수업계획",
                                "week": week
                            }
                        })
            
            # 평가개요
            if "평가개요" in subject_data:
                평가개요 = subject_data["평가개요"]
                for eval_type, eval_data in 평가개요.items():
                    if isinstance(eval_data, dict):
                        text = f"평가유형: {eval_type}\n평가내용: {eval_data.get('평가내용', '')}"
                        chunks.append({
                            "text": text,
                            "metadata": {
                                "subject": subject_name,
                                "type": "평가개요"
                            }
                        })
        
        print(f"📄 총 {len(chunks)}개의 텍스트 청크 추출됨")
        return chunks
    
    def vectorize_chunks(self, chunks: List[Dict[str, Any]]) -> tuple:
        """
        텍스트 청크를 벡터로 변환합니다.
        
        Args:
            chunks: 텍스트 청크 리스트
            
        Returns:
            (벡터 배열, 청크 리스트)
        """
        print("🔄 텍스트 벡터화 중...")
        texts = [chunk["text"] for chunk in chunks]
        
        # 배치로 인코딩 (진행률 표시)
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ 벡터화 완료: {embeddings.shape}")
        return embeddings, chunks
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        FAISS 인덱스를 생성합니다.
        
        Args:
            embeddings: 벡터 배열
            
        Returns:
            FAISS 인덱스
        """
        print("🗂️ FAISS 인덱스 생성 중...")
        
        # L2 거리 기반 인덱스 생성
        index = faiss.IndexFlatL2(self.dimension)
        
        # 벡터를 float32로 변환하여 추가
        embeddings_float32 = embeddings.astype('float32')
        index.add(embeddings_float32)
        
        print(f"✅ 인덱스 생성 완료: {index.ntotal}개 벡터")
        return index
    
    def save_index(self, index: faiss.Index, chunks: List[Dict], output_dir: str):
        """
        FAISS 인덱스와 메타데이터를 저장합니다.
        
        Args:
            index: FAISS 인덱스
            chunks: 텍스트 청크 리스트
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        index_file = output_path / "faiss_index.bin"
        faiss.write_index(index, str(index_file))
        print(f"💾 FAISS 인덱스 저장: {index_file}")
        
        # 메타데이터 저장
        metadata_file = output_path / "chunks_metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(chunks, f)
        print(f"💾 메타데이터 저장: {metadata_file}")
        
        print(f"\n✅ 모든 데이터 저장 완료!")


def main():
    """메인 실행 함수"""
    # 경로 설정
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_file = project_root / "utils" / "output.json"
    output_dir = script_dir / "vector_db"
    
    print("=" * 60)
    print("🚀 수업계획서 데이터 벡터화 시작")
    print("=" * 60)
    
    # 1. JSON 데이터 로드
    print(f"\n📂 데이터 로드 중: {data_file}")
    if not data_file.exists():
        print(f"❌ 오류: {data_file} 파일을 찾을 수 없습니다.")
        print(f"   먼저 utils/excel_utils.py를 실행하여 output.json을 생성하세요.")
        return
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ 데이터 로드 완료")
    
    # 2. 벡터화 객체 생성
    vectorizer = DataVectorizer()
    
    # 3. 텍스트 청크 추출
    chunks = vectorizer.extract_text_from_json(data)
    
    if not chunks:
        print("❌ 오류: 추출된 텍스트 청크가 없습니다.")
        return
    
    # 4. 벡터화
    embeddings, chunks = vectorizer.vectorize_chunks(chunks)
    
    # 5. FAISS 인덱스 생성
    index = vectorizer.create_faiss_index(embeddings)
    
    # 6. 저장
    vectorizer.save_index(index, chunks, str(output_dir))
    
    print("\n" + "=" * 60)
    print("🎉 벡터화 완료!")
    print("=" * 60)
    print(f"📊 통계:")
    print(f"  - 총 청크 수: {len(chunks)}")
    print(f"  - 벡터 차원: {vectorizer.dimension}")
    print(f"  - 저장 위치: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

