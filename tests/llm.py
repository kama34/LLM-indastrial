import aiohttp
import asyncio

class LLM:
    # url: str = 'http://10.100.30.240:1222/generate'  # vllm-server-qwen2-7b
    url: str = 'http://10.100.30.240:1224/generate'  # vllm-server-qwen2-72b

    async def get_response(self, json_data) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=json_data) as response:
                if response.status != 200:
                    raise Exception(f"Error: {response.status}")
                return await response.json()
            

answer = asyncio.run(LLM().get_response(
            {
                "prompt": "Почему небо голубое?",
                "stop": None,
                "max_tokens": 300,
                "choice": None,
                "schema": None,
                "regex": None,
                "temperature": 0.1,
            }
        ))
print(answer)