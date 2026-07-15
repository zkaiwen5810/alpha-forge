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

### Tool calling

Tool definitions and execution are isolated in `alpha_forge.tools`. The package
provides:

- `Tool` metadata, including canonical name, aliases, human description,
  model prompt, JSON input schema, implementation function, and reserved
  `is_mcp` and `validate_input` extension points.
- `ToolRegistry` registration, lookup, OpenAI schema generation, and dispatch.
- `load_builtin_tools()` for loading the tools shipped with Alpha Forge.

The default registry includes a safe `calculator` tool for arithmetic
expressions. The chat session sends registered tool definitions through the
OpenAI Chat Completions API, records calls and results in the in-flight model
context and visible transcript, and continues the same user turn until the
model produces a response without tool calls. Completed cross-turn history
retains the user prompt and final assistant response. A turn stops with an
error after 10 model iterations to prevent runaway tool loops.

Tool result content is bounded before it is added to the model context. One
result may contain at most 16,000 Unicode characters, and all results requested
by one assistant iteration may contain at most 32,000 characters in aggregate.
When multiple results exceed the aggregate limit, Alpha Forge shares the
available space fairly while leaving results that already fit their share
unchanged.

An oversized result is replaced by a self-identifying preview containing the
beginning and end of the result, the original character count, the truncation
reason, and the local path of the complete result. Only previewed originals are
persisted. The default layout is:

```text
$XDG_DATA_HOME/alpha-forge/<session-id>/tool-results/<tool-call-id>.{txt,json}
```

When `XDG_DATA_HOME` is unset, the base directory is
`~/.local/share/alpha-forge`. Results that parse as JSON use `.json`; all other
results use `.txt`. `/clear` starts a new storage session but does not delete
previous files. Persisted results are not automatically restored, exposed as a
file-reading tool, or cleaned up. The limits and persistence location are
built-in policy rather than CLI or user-config fields.

To build a controller with a custom registry:

```python
from alpha_forge.repl_controller import ChatReplController
from alpha_forge.tools import Tool, ToolRegistry

registry = ToolRegistry([
    Tool(
        name="greet",
        aliases=("hello",),
        description="Returns a greeting for a person.",
        prompt="Use this to create a greeting for a named person.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        function=lambda arguments: f"Hello, {arguments['name']}!",
    )
])
controller = ChatReplController(config, tool_registry=registry)
```

`is_mcp` and `validate_input` are definition-only in this release; MCP
execution and custom validator invocation are intentionally deferred.

### Conversation history

REPL orchestration lives in `alpha_forge.repl_controller`, while transcript
view models and rendering logic live in `alpha_forge.ui_state`. The controller
exposes the latter as `controller.ui_state`. `alpha_forge.session` remains a
compatibility import facade and contains no orchestration or UI implementation.

The history panel groups each user turn into one block. Model iterations are
stored as bundled snapshots so streaming text, tool calls, tool results, and
intermediate assistant notes share one rendering path.

Within a turn, tool calls and results are indented and rendered without blank
lines. Calls appear atomically before execution. All results from one assistant
iteration are collected before aggregate budgeting; the resulting full values
or previews then appear in call order. Model text continues streaming whenever
available. Text from a tool-requesting iteration is retained beneath its tool
result as an italic cyan `Assistant note`. Separate turns and command notices
have one blank line between their blocks. When the provider includes token
usage, the latest model iteration of the latest turn ends with a right-aligned
summary such as `Total tokens: 1,555 | Prompt cache: 72% reused`. Providers that
omit cache details still show the total, while raw cached-token counts are not
displayed.

## Local secrets

Store devcontainer startup secrets in host-side env files under `.devcontainer/`. Docker Compose injects each file into the service that needs it.

1. Copy `.devcontainer/app.env.example` to `.devcontainer/app.env`
2. Copy `.devcontainer/litellm.env.example` to `.devcontainer/litellm.env`
3. Copy `.devcontainer/cf-r2.env.example` to `.devcontainer/cf-r2.env`
4. Replace placeholder values with your real local secrets
5. Recreate the devcontainer after changing any env file
6. Open an example script in Zed
7. Run it with a Zed task or from the built-in terminal

`app.env` example:

```env
OPENAI_API_KEY=sk-your-local-litellm-master-key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=http://litellm:4000/v1
OPENAI_TIMEOUT=30
```

`litellm.env` example:

```env
OPENAI_API_KEY=your-real-openai-key
LITELLM_MASTER_KEY=sk-your-local-litellm-master-key
LITELLM_SALT_KEY=your-local-litellm-salt-key
```

`cf-r2.env` example:

```env
CF_R2_ACCESS_KEY_ID=your-cf-r2-access-key-id
CF_R2_SECRET_ACCESS_KEY=your-cf-r2-secret-access-key
CF_R2_BUCKET_NAME=litellm-logs
CF_R2_REGION=us-east-1
CF_R2_ENDPOINT_URL=your-cf-r2-endpoint-url
```

Read values in Python with `os.getenv("OPENAI_API_KEY")`.

Optional override:

Set `OPENAI_BASE_URL` in `.devcontainer/litellm.env` only when you intentionally want LiteLLM itself to call a non-default OpenAI-compatible provider.

Leave `OPENAI_BASE_URL` unset for the normal OpenAI API path. Only set it when you intentionally want to target an OpenAI-compatible non-default provider.

In the devcontainer, environment variables come from the host-side `.devcontainer/app.env`, `.devcontainer/litellm.env`, and `.devcontainer/cf-r2.env` files. Because the workspace is mounted as a Docker volume, do not use a repo-root `.env.local` as the devcontainer source of truth.

Avoid setting a placeholder `GITHUB_TOKEN` in `.devcontainer/app.env`. Zed may pass that variable through when downloading language server binaries such as Ruff, and an invalid token causes GitHub API `401 Bad credentials` failures.

Env file changes are runtime configuration. Restart or reopen the devcontainer to make new values effective; rebuilding the image is not required unless Dockerfile or image inputs changed.

## Configuration

The CLI resolves configuration from three layers, highest priority first:

1. **CLI flags** — `--model`, `--base-url`, `--timeout`, and `--init-config` (one-shot).
2. **User config file** — a TOML file at the XDG path `~/.config/alpha-forge/config.toml` (honors `$XDG_CONFIG_HOME`).
3. **Environment variables** — `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_TIMEOUT`. A repo-root `.env` file is also loaded.

If no layer provides an `api_key`, the CLI exits with a hint pointing at the user config file. Run `alpha-forge --init-config` to write a commented template you can edit.

### User config file

The user config file is the recommended place for persistent personal settings (your API key, your preferred model, your LiteLLM base URL). Generate it once and edit it in place:

```sh
alpha-forge --init-config
# then edit ~/.config/alpha-forge/config.toml
```

Format — one `[openai]` table, only the keys you need:

```toml
[openai]
api_key = "sk-..."
model = "gpt-4.1-mini"
base_url = "https://api.openai.com/v1"
timeout = 30
```

Unknown top-level keys are ignored for forward compatibility. A malformed file is a hard error — the CLI will not silently fall through to env vars. API keys are never accepted on the command line; use the file or an env var.

### CLI flags

```sh
alpha-forge --model gpt-4o
alpha-forge --base-url http://localhost:4000/v1
alpha-forge --timeout 60
alpha-forge --init-config
```

The CLI flags override the user file, which overrides env vars. The built-in
defaults are `gpt-4.1-mini` for `model` and 30 seconds for `timeout`.

### Environment variables

The env var layer is the lowest-priority source. It is also the easiest way to script the CLI in CI or in shell aliases without touching a config file:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT=30
```

`.env` files in the working directory are loaded via `python-dotenv` (existing var values are not overridden).

## LiteLLM proxy

The devcontainer starts a LiteLLM proxy next to the main app container with Docker Compose. The proxy is available at `http://localhost:4000` from the host.

When enabled, LiteLLM is a gateway layer in the architecture:

- Inside the app, code still uses the OpenAI SDK.
- The gateway is selected by setting `OPENAI_BASE_URL`.
- LiteLLM then forwards requests either to OpenAI or to another OpenAI-compatible upstream configured on the gateway side.
- Some OpenAI features may require extra LiteLLM configuration to behave the same way through the gateway.
- Responses API session continuity is backed by Cloudflare R2 through LiteLLM cold storage.

Local infrastructure for the gateway:

- LiteLLM API: `http://localhost:4000`

Cloudflare R2 is used as the S3-compatible cold-storage backend. LiteLLM persists prompt and response content there so `previous_response_id` can restore prior conversation state.

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

`.devcontainer/app.env`, `.devcontainer/litellm.env`, and `.devcontainer/cf-r2.env` should stay git-ignored. Keep real secrets out of committed files and use your deployment platform's secret manager in non-local environments.
