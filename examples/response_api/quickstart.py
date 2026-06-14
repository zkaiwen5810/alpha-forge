from openai import OpenAI
client = OpenAI()

# response = client.responses.create(
#     model="DeepSeek-V4-Flash",
#     input="Write a one-sentence bedtime story about a unicorn."
# )

# response = client.responses.create(
#     model="Qwen3.6-35B-A3B",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "input_text",
#                     "text": "What teams are playing in this image?",
#                 },
#                 {
#                     "type": "input_image",
#                     "image_url": "https://api.nga.gov/iiif/a2e6da57-3cd1-4235-b20e-95dcaefed6c8/full/!800,800/0/default.jpg",
#                 }
#             ]
#         }
#     ]
# )

# response = client.responses.create(
#     model="DeepSeek-V4-Flash",
#     tools=[{"type": "web_search"}],
#     input="What was a positive news story from today?"
# )

stream = client.responses.create(
    model="Qwen3.6-35B-A3B",
    input=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' ten times fast.",
        }
    ],
    stream=True,
)

for event in stream:
    print(event)

# print(response.output_text)
