"""
Forge RAG - 벡터 스토어 (ChromaDB 기반)
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import json
import os

# 데이터 저장 경로
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")

# ChromaDB 클라이언트
client = chromadb.PersistentClient(path=DATA_DIR)

# 컬렉션들
COLLECTIONS = {
    "documents": "isaac_sim_docs",      # Isaac Sim 문서
    "examples": "good_examples",         # 좋은 응답 예시
    "feedback": "user_feedback"          # 사용자 피드백
}


def get_collection(name: str):
    """컬렉션 가져오기 (없으면 생성)"""
    collection_name = COLLECTIONS.get(name, name)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def add_documents(texts: List[str], metadatas: List[Dict], collection_name: str = "documents"):
    """문서 추가"""
    collection = get_collection(collection_name)
    ids = [f"{collection_name}_{i}_{hash(text)}" for i, text in enumerate(texts)]

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    return len(texts)


def add_example(user_input: str, llm_output: dict, rating: str = "good"):
    """좋은 예시 추가 (지속학습)"""
    collection = get_collection("examples")

    doc = f"입력: {user_input}\n출력: {json.dumps(llm_output, ensure_ascii=False)}"
    metadata = {
        "user_input": user_input,
        "output": json.dumps(llm_output, ensure_ascii=False),
        "rating": rating,
        "type": "example"
    }

    doc_id = f"example_{hash(user_input)}_{hash(str(llm_output))}"

    collection.add(
        documents=[doc],
        metadatas=[metadata],
        ids=[doc_id]
    )
    return doc_id


def add_feedback(user_input: str, llm_output: dict, feedback: str, is_positive: bool):
    """사용자 피드백 저장"""
    collection = get_collection("feedback")

    doc = f"입력: {user_input}\n출력: {json.dumps(llm_output, ensure_ascii=False)}\n피드백: {feedback}"
    metadata = {
        "user_input": user_input,
        "output": json.dumps(llm_output, ensure_ascii=False),
        "feedback": feedback,
        "is_positive": is_positive,
        "type": "feedback"
    }

    doc_id = f"feedback_{hash(user_input)}_{hash(feedback)}"

    collection.add(
        documents=[doc],
        metadatas=[metadata],
        ids=[doc_id]
    )

    # 긍정적 피드백이면 좋은 예시로도 추가
    if is_positive:
        add_example(user_input, llm_output, "good")

    return doc_id


def search(query: str, collection_name: str = "documents", n_results: int = 3) -> List[Dict]:
    """유사 문서 검색"""
    collection = get_collection(collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    docs = []
    if results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            docs.append({
                "content": doc,
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "distance": results['distances'][0][i] if results['distances'] else 0
            })

    return docs


def search_similar_examples(user_input: str, n_results: int = 3) -> List[Dict]:
    """유사한 좋은 예시 검색"""
    return search(user_input, "examples", n_results)


def get_collection_count(collection_name: str) -> int:
    """컬렉션 내 문서 수"""
    collection = get_collection(collection_name)
    return collection.count()
