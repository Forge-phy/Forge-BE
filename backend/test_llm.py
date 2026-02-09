"""
LLM 서비스 테스트
"""
import asyncio
import sys
sys.path.insert(0, '/Users/yangjeong-u/Projects/forge')

from llm.service import parse_environment_request, analyze_simulation_result

async def test_environment_parsing():
    """자연어 → 파라미터 변환 테스트"""
    print("=" * 50)
    print("테스트: 자연어 환경 설정")
    print("=" * 50)

    test_input = "30m x 20m 창고에 AGV 3대 배치해줘. 선반은 2열로"
    print(f"\n입력: {test_input}\n")

    result = await parse_environment_request(test_input)

    import json
    print("출력 (Isaac Sim 파라미터):")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

async def test_result_analysis(config: dict):
    """결과 분석 테스트"""
    print("\n" + "=" * 50)
    print("테스트: 시뮬레이션 결과 분석")
    print("=" * 50)

    # 가상 시뮬레이션 결과
    sim_result = {
        "duration": 3600,
        "throughput": 95,
        "unit": "items/hour",
        "collisions": [
            {"time": 45.3, "robot1": "AGV_1", "robot2": "AGV_2", "location": "intersection_A"},
            {"time": 156.7, "robot1": "AGV_1", "robot2": "AGV_3", "location": "intersection_A"},
            {"time": 892.1, "robot1": "AGV_2", "robot2": "AGV_3", "location": "shelf_row_2"}
        ],
        "idle_time": {
            "AGV_1": 420,
            "AGV_2": 380,
            "AGV_3": 510
        },
        "bottleneck": "intersection_A"
    }

    print("\n시뮬레이션 결과:")
    import json
    print(json.dumps(sim_result, indent=2, ensure_ascii=False))

    print("\nLLM 분석 결과:")
    analysis = await analyze_simulation_result(config, sim_result)
    print(analysis)

async def main():
    config = await test_environment_parsing()
    if config:
        await test_result_analysis(config)

if __name__ == "__main__":
    asyncio.run(main())
