# alpha-forge

This project aims to implement an agent on top of the OpenAI SDK or the surrounding OpenAI ecosystem.

Project constraint:

- OpenAI is the default and intended provider path.
- Do not introduce SiliconFlow-specific API keys, base URLs, model names, or config as defaults.
- Use provider-specific settings only when they are strictly necessary, and keep them opt-in behind generic OpenAI-compatible overrides such as `OPENAI_BASE_URL`.

## Local secrets

Store local API keys and other secrets in `.env.local`.

1. Copy `.env.example` to `.env.local`
2. Replace placeholder values with your real local secrets
3. Export `UV_ENV_FILE=.env.local`
4. Open an example script in Zed
5. Run it with a Zed task or from the built-in terminal

Example:

```env
OPENAI_API_KEY=your-real-key
OPENAI_MODEL=gpt-4.1-mini
GITHUB_TOKEN=your-real-token
```

Read values in Python with `os.getenv("OPENAI_API_KEY")`.

Optional override:

```env
OPENAI_BASE_URL=https://your-openai-compatible-provider.example/v1
```

Leave `OPENAI_BASE_URL` unset for the normal OpenAI API path. Only set it when you intentionally want to target an OpenAI-compatible non-default provider.

The example scripts load environment variables from the file path stored in `UV_ENV_FILE`. In the devcontainer, [.devcontainer/devcontainer.json](/workspaces/alpha-forge/.devcontainer/devcontainer.json) already sets `UV_ENV_FILE` to `${containerWorkspaceFolder}/.env.local`.

Install the project environment first:

```sh
uv sync
```

Zed note:

Zed's Python REPL cell execution is currently not supported in devcontainers using `docker exec`, so these examples are set up as normal Python scripts and should be run with Zed tasks or the terminal instead of `# %%` cells.

Current shell session:

```sh
export UV_ENV_FILE=.env.local
zed .
```

If you want that to be automatic every time you enter the repo, use a shell tool like `direnv` to export `UV_ENV_FILE=.env.local` for this directory.

Run the examples from Zed:

1. Open the command palette
2. Run `task: spawn`
3. Choose `Run current Python file with uv`, `Run chat example`, or `Run tool function example`

You can also run the examples directly from a terminal:

```sh
UV_ENV_FILE=.env.local uv run python examples/chat.py
UV_ENV_FILE=.env.local uv run python examples/tool_function.py
```

Available examples:

- `examples/chat.py`
- `examples/tool_function.py`

These examples are OpenAI-first:

- They require `OPENAI_API_KEY`.
- They default to `OPENAI_MODEL=gpt-4.1-mini`.
- They only use `OPENAI_BASE_URL` if you explicitly set it.

`.env.local` is git-ignored. Keep real secrets out of committed files and use your deployment platform's secret manager in non-local environments.
