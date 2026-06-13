from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="Qwen3.5-4B",
    input="Write a one-sentence bedtime story about a unicorn.",
)

print(response.output_text)
