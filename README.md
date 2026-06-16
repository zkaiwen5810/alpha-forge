# alpha-forge

This project aims to implement an agent on top of the OpenAI SDK or the surrounding OpenAI ecosystem.

Project constraint:

- OpenAI is the default and intended provider path.
- Do not introduce SiliconFlow-specific API keys, base URLs, model names, or config as defaults.
- Use provider-specific settings only when they are strictly necessary, and keep them opt-in behind generic OpenAI-compatible overrides such as `OPENAI_BASE_URL`.

## Architecture

The application is built around the OpenAI SDK and the OpenAI API shape.

Default path:

- App code uses the OpenAI SDK directly.
- `OPENAI_API_KEY` is required.
- `OPENAI_MODEL` defaults to an OpenAI model such as `gpt-4.1-mini`.
- `OPENAI_BASE_URL` stays unset, so requests go to OpenAI directly.

Gateway path:

- The devcontainer also runs a local LiteLLM gateway.
- App code can point the same OpenAI SDK client at LiteLLM by setting `OPENAI_BASE_URL`.
- LiteLLM acts as an OpenAI-compatible gateway in front of the upstream provider.
- This keeps application code OpenAI-shaped even when traffic is routed through a proxy.

Design intent:

- OpenAI remains the primary and documented provider path.
- LiteLLM is part of the local and deployment architecture when a gateway is useful.
- Non-OpenAI providers must remain opt-in and should be introduced through generic OpenAI-compatible settings, not provider-specific defaults in app code.

## Local secrets

Store devcontainer startup secrets in host-side env files under `.devcontainer/`. Docker Compose injects each file into the service that needs it. `minio.env` is shared with both MinIO and LiteLLM so LiteLLM can authenticate to the local cold-storage backend.

1. Copy `.devcontainer/app.env.example` to `.devcontainer/app.env`
2. Copy `.devcontainer/litellm.env.example` to `.devcontainer/litellm.env`
3. Copy `.devcontainer/minio.env.example` to `.devcontainer/minio.env`
4. Replace placeholder values with your real local secrets
5. Recreate the devcontainer after changing any env file
6. Open an example script in Zed
7. Run it with a Zed task or from the built-in terminal

`app.env` example:

```env
OPENAI_API_KEY=sk-your-local-litellm-master-key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=http://litellm:4000/v1
```

`litellm.env` example:

```env
OPENAI_API_KEY=your-real-openai-key
LITELLM_MASTER_KEY=sk-your-local-litellm-master-key
LITELLM_SALT_KEY=your-local-litellm-salt-key
```

`minio.env` example:

```env
MINIO_ROOT_USER=your-minio-root-user
MINIO_ROOT_PASSWORD=your-minio-root-password
MINIO_BUCKET_NAME=litellm-logs
MINIO_REGION=us-east-1
```

Read values in Python with `os.getenv("OPENAI_API_KEY")`.

Optional override:

Set `OPENAI_BASE_URL` in `.devcontainer/litellm.env` only when you intentionally want LiteLLM itself to call a non-default OpenAI-compatible provider.

Leave `OPENAI_BASE_URL` unset for the normal OpenAI API path. Only set it when you intentionally want to target an OpenAI-compatible non-default provider.

In the devcontainer, environment variables come from the host-side `.devcontainer/app.env`, `.devcontainer/litellm.env`, and `.devcontainer/minio.env` files. Because the workspace is mounted as a Docker volume, do not use a repo-root `.env.local` as the devcontainer source of truth.

Avoid setting a placeholder `GITHUB_TOKEN` in `.devcontainer/app.env`. Zed may pass that variable through when downloading language server binaries such as Ruff, and an invalid token causes GitHub API `401 Bad credentials` failures.

Env file changes are runtime configuration. Restart or reopen the devcontainer to make new values effective; rebuilding the image is not required unless Dockerfile or image inputs changed.

## LiteLLM proxy

The devcontainer starts a LiteLLM proxy next to the main app container with Docker Compose. The proxy is available at `http://localhost:4000` from the host.

When enabled, LiteLLM is a gateway layer in the architecture:

- Inside the app, code still uses the OpenAI SDK.
- The gateway is selected by setting `OPENAI_BASE_URL`.
- LiteLLM then forwards requests either to OpenAI or to another OpenAI-compatible upstream configured on the gateway side.
- Some OpenAI features may require extra LiteLLM configuration to behave the same way through the gateway.
- Responses API session continuity is backed by the local MinIO service through LiteLLM cold storage.

Local infrastructure for the gateway:

- LiteLLM API: `http://localhost:4000`
- MinIO S3 API: `http://localhost:9000`
- MinIO console: `http://localhost:9001`

MinIO is used as a self-hosted S3-compatible cold-storage backend. On startup, the devcontainer creates the configured bucket automatically so LiteLLM can persist prompt and response content for `previous_response_id` session continuity.

To route app code through LiteLLM from inside the devcontainer, set this in `.devcontainer/app.env`:

```env
OPENAI_BASE_URL=http://litellm:4000/v1
```

To route host-side tools through LiteLLM, use:

```env
OPENAI_BASE_URL=http://localhost:4000/v1
```

Leave `OPENAI_BASE_URL` unset for the normal OpenAI API path.

Install the project environment first:

```sh
uv sync
```

Zed note:

Zed's Python REPL cell execution is currently not supported in devcontainers using `docker exec`, so these examples are set up as normal Python scripts and should be run with Zed tasks or the terminal instead of `# %%` cells.

Current shell session:

```sh
zed .
```

Run the examples from Zed:

1. Open the command palette
2. Run `task: spawn`
3. Choose `Run current Python file with uv`, `Run chat example`, or `Run tool function example`

You can also run the examples directly from a terminal:

```sh
uv run python examples/chat.py
uv run python examples/tool_function.py
```

Available examples:

- `examples/chat.py`
- `examples/tool_function.py`

These examples are OpenAI-first:

- They require `OPENAI_API_KEY`.
- They default to `OPENAI_MODEL=gpt-4.1-mini`.
- They only use `OPENAI_BASE_URL` if you explicitly set it.

`.devcontainer/app.env`, `.devcontainer/litellm.env`, and `.devcontainer/minio.env` should stay git-ignored. Keep real secrets out of committed files and use your deployment platform's secret manager in non-local environments.
