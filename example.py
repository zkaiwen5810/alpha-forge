import os

from openai import OpenAI


api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Set SILICONFLOW_API_KEY or OPENAI_API_KEY before running this script."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash",
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn",
        }
    ],
)

print(response.choices[0].message.content)
