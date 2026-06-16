from openai import OpenAI

client = OpenAI()

# history = [{"role": "user", "content": "tell me a joke"}]

# response = client.responses.create(
#     model="DeepSeek-V4-Flash",
#     input=history,
#     store=False,
#     include=["reasoning.encrypted_content"],
# )

# print(response.output_text)

# # Add all response output items, including encrypted reasoning items, to the conversion
# history += response.output

# history += [{"role": "user", "content": "tell me another"}]

# second_response = client.responses.create(
#     model="DeepSeek-V4-Flash",
#     input=history,
#     store=False,
#     include=["reasoning.encrypted_content"],
# )

# print(second_response.output_text)

response = client.responses.create(
    model="Qwen3.6-35B-A3B",
    input="tell me a joke",
    store=True,
)
print(response.output_text)
print("response.id: ", response.id)

second_response = client.responses.create(
    model="Qwen3.6-35B-A3B",
    previous_response_id=response.id,
    input=[{"role": "user", "content": "explain why this is funny."}],
)
print(second_response.output_text)
