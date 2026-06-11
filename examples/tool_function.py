from common import create_client, get_model, load_local_env, print_response


def main() -> None:
    env_path = load_local_env()
    print(f"Loaded environment variables from {env_path}")

    client = create_client()
    model = get_model()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogota, Colombia",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "What is the weather like in Paris today",
            },
        ],
        model=model,
        tools=tools,
    )

    print_response(response)


if __name__ == "__main__":
    main()
