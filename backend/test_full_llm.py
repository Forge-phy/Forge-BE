"""
Forge LLM 전체 테스트 - 완전한 버전
"""
import asyncio
import json
import sys
sys.path.insert(0, '/Users/yangjeong-u/Projects/forge')


async def test_full_flow():
    """전체 LLM 플로우 테스트"""

    print("=" * 70)
    print("Forge LLM 전체 테스트")
    print("=" * 70)

    # 1. 초기 데이터 로드
    print("\n[1] 초기 데이터 로드")
    print("-" * 50)
    from rag.init_data import init_all
    init_all()

    # 2. RAG 서비스 임포트
    from rag.service import rag_parse_environment, rag_analyze_result, submit_feedback
    from rag.vector_store import search, search_similar_examples

    # 3. 환경 생성 테스트
    print("\n[2] 환경 생성 테스트")
    print("-" * 50)

    test_cases = [
        "30m x 20m 창고에 AGV 3대, 선반 2열",
        "대형 물류센터 100x80m, AMR 8대, 컨베이어 5개",
        "작은 연구실 15x10m, 로봇 1대만",
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_input}")

        # 관련 문서/예시 검색
        docs = search(test_input, "documents", n_results=2)
        examples = search_similar_examples(test_input, n_results=2)

        docs_text = "\n".join([d["content"][:200] for d in docs]) if docs else ""
        examples_text = "\n".join([e["content"][:200] for e in examples]) if examples else ""

        result = await rag_parse_environment(test_input)

        if result:
            print(f"  환경: {result.get('environment', {}).get('type')} "
                  f"{result.get('environment', {}).get('width')}x"
                  f"{result.get('environment', {}).get('length')}m")

            robots = result.get('robots', [])
            if robots:
                r = robots[0]
                print(f"  로봇: {r.get('type')} {r.get('count')}대, 속도 {r.get('speed')}m/s")

            obstacles = result.get('obstacles', [])
            if obstacles:
                o = obstacles[0]
                print(f"  장애물: {o.get('type')} {o.get('count')}개")

    # 4. 대화형 수정 테스트
    print("\n[3] 대화형 수정 테스트")
    print("-" * 50)

    from llm.service import modify_parameters, create_session

    # 세션 생성
    session = create_session()
    print(f"세션 ID: {session.session_id}")

    # 초기 설정
    base_config = {
        "environment": {"type": "warehouse", "width": 30, "length": 20, "height": 5},
        "robots": [{"type": "AGV", "count": 3, "speed": 1.5, "payload": 100}],
        "obstacles": [{"type": "shelf", "count": 2, "arrangement": "row"}],
        "simulation": {"duration": 3600, "time_step": 0.01}
    }

    modifications = [
        "속도 올려줘",
        "로봇 2대 더 추가해",
        "시뮬레이션 2시간으로",
    ]

    current = base_config
    for mod in modifications:
        print(f"\n수정 요청: {mod}")

        response = await modify_parameters(current, mod, session.session_id)

        if response.success:
            current = response.data
            # 변경된 부분 표시
            robots = current.get('robots', [{}])[0]
            sim = current.get('simulation', {})
            print(f"  → 로봇: {robots.get('count')}대, 속도: {robots.get('speed')}m/s")
            print(f"  → 시뮬: {sim.get('duration')}초")
        else:
            print(f"  → 실패: {response.error}")

    # 5. 결과 분석 테스트
    print("\n[4] 시뮬레이션 결과 분석")
    print("-" * 50)

    sim_result = {
        "duration": 3600,
        "throughput": 85,
        "total_tasks_completed": 85,
        "total_collisions": 5,
        "collisions": [
            {"time": 45.3, "robot1": "AGV_1", "robot2": "AGV_2", "location": "intersection_A", "severity": "medium"},
            {"time": 156.7, "robot1": "AGV_1", "robot2": "AGV_3", "location": "intersection_A", "severity": "high"},
        ],
        "bottleneck": "intersection_A",
        "efficiency": 72.5,
        "utilization": 68.0
    }

    print("시뮬레이션 결과:")
    print(f"  처리량: {sim_result['throughput']} items/hour")
    print(f"  충돌: {sim_result['total_collisions']}회")
    print(f"  효율성: {sim_result['efficiency']}%")

    analysis = await rag_analyze_result(current, sim_result)
    print(f"\nLLM 분석 (요약):")
    print(analysis[:500] + "..." if len(analysis) > 500 else analysis)

    # 6. 피드백 테스트
    print("\n[5] 지속학습 피드백")
    print("-" * 50)

    from rag.vector_store import get_collection_count

    before_count = get_collection_count("examples")
    print(f"피드백 전 예시 수: {before_count}")

    submit_feedback(
        user_input="30m x 20m 창고에 AGV 3대, 선반 2열",
        llm_output=base_config,
        is_positive=True,
        comment="정확한 파라미터 생성"
    )

    after_count = get_collection_count("examples")
    print(f"피드백 후 예시 수: {after_count} (학습됨!)")

    # 7. 스키마 검증 테스트
    print("\n[6] 스키마 검증 테스트")
    print("-" * 50)

    from llm.schemas import validate_and_normalize

    # 유효한 설정
    try:
        valid = validate_and_normalize(base_config)
        print("유효한 설정: 검증 통과")
    except Exception as e:
        print(f"유효한 설정 실패: {e}")

    # 잘못된 설정
    invalid_config = {
        "environment": {"type": "warehouse", "width": -10, "length": 20},  # 음수 width
        "robots": [{"type": "AGV", "count": 0}]  # count 0
    }

    try:
        validate_and_normalize(invalid_config)
        print("잘못된 설정: 검증 통과 (문제!)")
    except Exception as e:
        print(f"잘못된 설정 거부: {str(e)[:100]}...")

    # 최종 요약
    print("\n" + "=" * 70)
    print("테스트 완료 - 최종 현황")
    print("=" * 70)
    print(f"  문서: {get_collection_count('documents')}개")
    print(f"  좋은 예시: {get_collection_count('examples')}개")
    print(f"  피드백: {get_collection_count('feedback')}개")


if __name__ == "__main__":
    asyncio.run(test_full_flow())
