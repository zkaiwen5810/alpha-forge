# AGENTS.md

## Project memory

This repository is for building an agent on top of the OpenAI SDK or the broader OpenAI ecosystem.

Persistent constraint:

- OpenAI is the default and intended provider path.
- The architecture may include LiteLLM as an OpenAI-compatible gateway, but that gateway must not replace OpenAI as the default documented path.
- The local development architecture may include an S3-compatible persistence layer for LiteLLM cold storage and session continuity.
- Do not add SiliconFlow-specific API keys, base URLs, model defaults, or configuration as the standard project behavior.
- If support for another OpenAI-compatible provider is ever needed, keep it opt-in and minimal.
- Prefer generic settings such as `OPENAI_BASE_URL` over provider-branded configuration when possible.
- Documentation, examples, and scaffolding should present the OpenAI path first.
- When documenting architecture, describe LiteLLM as a gateway/proxy layer selected via `OPENAI_BASE_URL`, not as the primary application API surface.
- When documenting persistence, describe the S3-compatible storage layer as infrastructure for LiteLLM cold storage, not as an application-facing dependency.

## Configuration

The CLI resolves configuration by merging three layers, highest priority first: **CLI flags > user config file > env vars > built-in defaults**. The user config file is a TOML document at the XDG path `~/.config/alpha-forge/config.toml` (honoring `$XDG_CONFIG_HOME`), with a single `[openai]` table.

Security rule: **API keys are never accepted on the command line.** Do not add `--api-key` or any flag that takes a key as a value. Keys must come from the user config file or an env var so they stay out of process listings (`/proc/*/cmdline`) and shell history.

### Adding a config field

When adding a new configurable knob to the CLI, follow this three-step process so the layered merge stays consistent:

1. Add the `Optional` field to `ConfigSource` in `alpha_forge/config.py`.
2. Wire the read in `load_user_config` (TOML key) and `load_env_config` (env var), and add a CLI flag in `alpha_forge/cli.py` only if appropriate. Do not add a CLI flag for secrets.
3. Update the priority test in `tests/test_config.py` so the new field is covered by the layered-merge matrix.

### Non-features

- `ALPHA_FORGE_CONFIG` env var as a path override for the user config file is intentionally not implemented. Tests use `$XDG_CONFIG_HOME`; users who want a non-XDG location can symlink.
