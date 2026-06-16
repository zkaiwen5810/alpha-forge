from time import sleep
from uuid import uuid4

from openai import OpenAI

client = OpenAI()
session_id = f"conversation-state-{uuid4()}"

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
    # extra_body={"litellm_session_id": session_id},
)
print(response.output_text)

# LiteLLM writes spend logs asynchronously. The session handler rehydrates
# from those logs, so an immediate follow-up can race the DB write.
sleep(10)

second_response = client.responses.create(
    model="Qwen3.6-35B-A3B",
    previous_response_id=response.id,
    input=[{"role": "user", "content": "explain why this is funny."}],
    # extra_body={"litellm_session_id": session_id},
)
print(second_response.output_text)
