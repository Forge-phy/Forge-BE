"""
RAG + 지속학습 테스트
"""
import asyncio
import json
import sys
sys.path.insert(0, '/Users/yangjeong-u/Projects/forge')

from rag.init_data import init_all
from rag.service import rag_parse_environment, submit_feedback
from rag.vector_store import search_similar_examples, get_collection_count


async def test_rag_flow():
    """RAG + 지속학습 전체 플로우 테스트"""

    # 1. 초기 데이터 로드
    print("\n[1단계] 초기 데이터 로드")
    init_all()

    # 2. RAG 기반 파라미터 생성
    print("\n[2단계] RAG 기반 환경 생성")
    print("-" * 50)

    test_input = "40m x 30m 창고에 AGV 4대, 선반 3열로 배치"
    print(f"입력: {test_input}\n")

    result = await rag_parse_environment(test_input)
    print("출력:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 3. 피드백 제출 (긍정적)
    print("\n[3단계] 피드백 제출 (좋아요!)")
    print("-" * 50)

    submit_feedback(test_input, result, is_positive=True, comment="정확한 파라미터 생성")
    print(f"예시 개수: {get_collection_count('examples')}개 (피드백 후 증가)")

    # 4. 유사 예시 검색 확인
    print("\n[4단계] 유사 예시 검색 테스트")
    print("-" * 50)

    similar = search_similar_examples("35m x 25m 창고, AGV 3대", n_results=2)
    print("유사 예시:")
    for i, ex in enumerate(similar, 1):
        print(f"\n[{i}] (거리: {ex['distance']:.4f})")
        print(ex['content'][:200] + "...")

    # 5. 학습된 내용으로 새 질문
    print("\n[5단계] 학습 후 새 질문")
    print("-" * 50)

    new_input = "35m x 25m 창고, AGV 5대"
    print(f"입력: {new_input}\n")

    new_result = await rag_parse_environment(new_input)
    print("출력 (학습된 예시 참조됨):")
    print(json.dumps(new_result, indent=2, ensure_ascii=False))

    # 최종 통계
    print("\n" + "=" * 50)
    print("최종 학습 현황")
    print("=" * 50)
    print(f"  문서: {get_collection_count('documents')}개")
    print(f"  좋은 예시: {get_collection_count('examples')}개")
    print(f"  피드백: {get_collection_count('feedback')}개")


if __name__ == "__main__":
    asyncio.run(test_rag_flow())
