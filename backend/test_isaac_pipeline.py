"""
Forge Isaac Sim Pipeline 테스트
"""

import asyncio
import sys
sys.path.insert(0, '/Users/yangjeong-u/Projects/forge')

from isaac import IsaacSimPipeline, create_default_warehouse_config


async def test_natural_language():
    """자연어 파이프라인 테스트"""
    print("=" * 60)
    print("테스트 1: 자연어 입력")
    print("=" * 60)

    pipeline = IsaacSimPipeline()

    # 자연어 요청
    request = "30m x 20m 창고에 AGV 5대, 온도 32도, 습도 70% 환경에서 1시간 시뮬레이션"
    print(f"\n요청: {request}\n")

    result = await pipeline.run_from_natural_language(request)

    if result.success:
        print(result.report.summary)
        print("\n■ 권장사항")
        for rec in result.report.recommendations:
            print(f"  • {rec}")
        if result.report.warnings:
            print("\n■ 경고")
            for warn in result.report.warnings:
                print(f"  {warn}")
    else:
        print(f"실패: {result.error_message}")


async def test_config_based():
    """설정 기반 파이프라인 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 설정 기반 실행")
    print("=" * 60)

    pipeline = IsaacSimPipeline()

    # 환경 설정 생성
    env_config = create_default_warehouse_config(
        width=40.0,
        length=30.0,
        robot_count=8
    )
    env_config.temperature = 28.0
    env_config.humidity = 65.0

    print(f"\n환경: {env_config.warehouse.width}m x {env_config.warehouse.length}m")
    print(f"로봇: {len(env_config.robots)}대")
    print(f"온도: {env_config.temperature}°C, 습도: {env_config.humidity}%\n")

    result = await pipeline.run_from_config(env_config)

    if result.success:
        print(result.report.summary)
    else:
        print(f"실패: {result.error_message}")


async def main():
    """메인 테스트"""
    await test_natural_language()
    await test_config_based()

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
