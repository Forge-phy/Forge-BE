"""
초기 데이터 로드 - Isaac Sim 문서 및 예시 (확장판)
"""
import sys
sys.path.insert(0, '/Users/yangjeong-u/Projects/forge')

from rag.vector_store import add_documents, add_example, get_collection_count
from rag.documents import ISAAC_SIM_DOCUMENTS, GOOD_EXAMPLES, ANALYSIS_EXAMPLES


def init_documents():
    """문서 초기화"""
    texts = [doc["text"] for doc in ISAAC_SIM_DOCUMENTS]
    metadatas = [doc["metadata"] for doc in ISAAC_SIM_DOCUMENTS]

    count = add_documents(texts, metadatas, "documents")
    print(f"  문서 {count}개 추가됨")
    return count


def init_examples():
    """좋은 예시 초기화"""
    count = 0
    for ex in GOOD_EXAMPLES:
        if "output" in ex and isinstance(ex["output"], dict):
            # 완전한 출력이 있는 예시만
            if "environment" in ex["output"] or "robots" in ex["output"]:
                add_example(ex["input"], ex["output"], "good")
                count += 1
    print(f"  예시 {count}개 추가됨")
    return count


def init_analysis_examples():
    """분석 예시 초기화"""
    collection_name = "analysis_examples"
    from rag.vector_store import get_collection

    collection = get_collection(collection_name)

    count = 0
    for ex in ANALYSIS_EXAMPLES:
        doc = f"설정: {ex['config_summary']}\n결과: {ex['result_summary']}\n분석:\n{ex['analysis']}"
        metadata = {
            "config_summary": ex["config_summary"],
            "result_summary": ex["result_summary"],
            "type": "analysis_example"
        }
        collection.add(
            documents=[doc],
            metadatas=[metadata],
            ids=[f"analysis_{hash(ex['config_summary'])}"]
        )
        count += 1

    print(f"  분석 예시 {count}개 추가됨")
    return count


def init_all():
    """모든 초기 데이터 로드"""
    print("=" * 60)
    print("Forge RAG 초기 데이터 로드 (확장판)")
    print("=" * 60)

    print("\n[1/3] Isaac Sim 문서 로드...")
    init_documents()

    print("\n[2/3] 좋은 예시 로드...")
    init_examples()

    print("\n[3/3] 분석 예시 로드...")
    init_analysis_examples()

    print("\n" + "-" * 60)
    print("최종 현황:")
    print(f"  - 문서: {get_collection_count('documents')}개")
    print(f"  - 좋은 예시: {get_collection_count('examples')}개")
    print(f"  - 분석 예시: {get_collection_count('analysis_examples')}개")
    print(f"  - 피드백: {get_collection_count('feedback')}개")
    print("=" * 60)


def reset_all():
    """모든 데이터 초기화 (주의: 삭제 후 재생성)"""
    import chromadb
    from rag.vector_store import DATA_DIR, client

    print("경고: 모든 데이터가 삭제됩니다!")

    # 컬렉션 삭제
    for name in ["isaac_sim_docs", "good_examples", "user_feedback", "analysis_examples"]:
        try:
            client.delete_collection(name)
            print(f"  삭제됨: {name}")
        except:
            pass

    # 재생성
    init_all()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_all()
    else:
        init_all()
