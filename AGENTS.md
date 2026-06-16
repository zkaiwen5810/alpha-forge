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
