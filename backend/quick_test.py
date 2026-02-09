"""간단 테스트"""
import asyncio
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, '.')
from llm.providers import init_config
from llm.service import parse_environment_request
import json

async def test():
    init_config()

    print("=" * 50)
    print("Forge v2.0 간단 테스트")
    print("=" * 50)

    prompt = "30m x 20m 창고에 AGV 3대 배치해줘"
    print(f"\n입력: {prompt}\n")
    print("처리 중...")

    result = await parse_environment_request(prompt)

    print(f"\n프로바이더: {result.provider}")
    print(f"민감도: {result.sensitivity}")
    print(f"\n생성된 파라미터:")
    print(json.dumps(result.data, indent=2, ensure_ascii=False))

asyncio.run(test())
