"""
Forge v2.0 테스트 - OpenAI + Qwen 하이브리드
"""
import asyncio
import json
import os
import sys
sys.path.insert(0, '/Users/yangjeong-u/Projects/forge')


async def test_v2_hybrid():
    """v2.0 하이브리드 LLM 테스트"""

    print("=" * 70)
    print("Forge v2.0 하이브리드 LLM 테스트")
    print("=" * 70)

    # 1. 설정 초기화
    from llm.providers import init_config, LLMProvider, SensitivityLevel, classify_sensitivity

    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[경고] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("export OPENAI_API_KEY='your-key' 로 설정하세요.")
        print("OpenAI 없이 Qwen만 사용합니다.\n")

    config = init_config(
        openai_api_key=api_key,
        openai_model="gpt-4o-mini",
        ollama_model="qwen2.5:7b"
    )

    print(f"\n설정:")
    print(f"  OpenAI: {'활성화' if api_key else '비활성화'}")
    print(f"  OpenAI 모델: {config.openai_model}")
    print(f"  Qwen 모델: {config.ollama_model}")

    # 2. 민감도 분류 테스트
    print("\n" + "-" * 50)
    print("[1] 민감도 분류 테스트")
    print("-" * 50)

    test_texts = [
        ("30m x 20m 창고에 AGV 3대 배치", "일반 정보"),
        ("생산량 데이터를 분석해줘", "내부 정보"),
        ("비밀번호를 변경하고 싶어", "기밀 정보"),
        ("로봇 속도 올려줘", "일반 정보"),
        ("매출 데이터 기반으로 예측해줘", "내부 정보"),
    ]

    for text, expected in test_texts:
        sensitivity = classify_sensitivity(text)
        status = "O" if (
            (expected == "일반 정보" and sensitivity == SensitivityLevel.PUBLIC) or
            (expected == "내부 정보" and sensitivity == SensitivityLevel.INTERNAL) or
            (expected == "기밀 정보" and sensitivity == SensitivityLevel.CONFIDENTIAL)
        ) else "X"
        print(f"  [{status}] '{text[:30]}...' → {sensitivity.value} (예상: {expected})")

    # 3. LLM 호출 테스트
    print("\n" + "-" * 50)
    print("[2] LLM 호출 테스트 (민감도 기반 분기)")
    print("-" * 50)

    from llm.service import parse_environment_request

    # 일반 정보 → OpenAI (API 키 있으면)
    print("\n테스트 A: 일반 정보 (창고 설정)")
    result = await parse_environment_request("25m x 15m 창고에 AGV 2대, 선반 1열")

    if result.success:
        print(f"  프로바이더: {result.provider}")
        print(f"  민감도: {result.sensitivity}")
        env = result.data.get("environment", {})
        print(f"  결과: {env.get('type')} {env.get('width')}x{env.get('length')}m")
    else:
        print(f"  실패: {result.error}")

    # 4. LSTM → LLM 보고서 생성 테스트 (v2.0 핵심)
    print("\n" + "-" * 50)
    print("[3] LSTM → LLM 보고서 생성 (v2.0 핵심)")
    print("-" * 50)

    from llm.service import generate_report_from_lstm

    lstm_result = {
        "sim_throughput": 100,
        "predicted_real_throughput": 85,
        "gap_percent": 15,
        "confidence": 0.89,
        "factors": ["temperature", "humidity"],
        "confidence_interval": [83, 87]
    }

    sim_config = {
        "environment": {"type": "warehouse", "width": 30, "length": 20},
        "robots": [{"type": "AGV", "count": 3}]
    }

    print("\nLSTM 예측 결과:")
    print(f"  시뮬 처리량: {lstm_result['sim_throughput']}개/h")
    print(f"  예상 현장: {lstm_result['predicted_real_throughput']}개/h")
    print(f"  갭: {lstm_result['gap_percent']}%")
    print(f"  신뢰도: {lstm_result['confidence']*100}%")

    print("\nLLM 보고서 생성 중...")
    report = await generate_report_from_lstm(lstm_result, sim_config)

    if report.success:
        print(f"\n프로바이더: {report.provider}")
        print(f"민감도: {report.sensitivity}")
        print("\n=== 생성된 보고서 ===")
        print(report.data[:1000] + "..." if len(report.data) > 1000 else report.data)
    else:
        print(f"실패: {report.error}")

    # 5. 강제 프로바이더 테스트
    print("\n" + "-" * 50)
    print("[4] 프로바이더 강제 지정 테스트")
    print("-" * 50)

    # Qwen 강제 사용
    print("\nQwen 강제 사용:")
    result = await parse_environment_request(
        "10m x 10m 테스트 공간",
        force_provider=LLMProvider.OLLAMA
    )
    if result.success:
        print(f"  프로바이더: {result.provider}")
    else:
        print(f"  실패: {result.error}")

    print("\n" + "=" * 70)
    print("테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_v2_hybrid())
