import sys
from pathlib import Path

from examples.common import create_client, get_model, load_local_env, print_response


def main() -> None:
    env_path = load_local_env()
    if env_path:
        print(f"Loaded environment variables from {env_path}")

    client = create_client()
    model = get_model()
    response = client.responses.create(
        model=model,
        input="Write a one-sentence bedtime story about a unicorn.",
    )

    print_response(response)


if __name__ == "__main__":
    main()
