from common import create_client, get_model, load_local_env, print_response


def main() -> None:
    env_path = load_local_env()
    print(f"Loaded environment variables from {env_path}")

    client = create_client()
    model = get_model()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Write a one-sentence bedtime story about a unicorn",
            }
        ],
    )

    print_response(response)


if __name__ == "__main__":
    main()
