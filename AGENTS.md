# AGENTS.md

## Project memory

This repository is for building an agent on top of the OpenAI SDK or the broader OpenAI ecosystem.

Persistent constraint:

- OpenAI is the default and intended provider path.
- Do not add SiliconFlow-specific API keys, base URLs, model defaults, or configuration as the standard project behavior.
- If support for another OpenAI-compatible provider is ever needed, keep it opt-in and minimal.
- Prefer generic settings such as `OPENAI_BASE_URL` over provider-branded configuration when possible.
- Documentation, examples, and scaffolding should present the OpenAI path first.
