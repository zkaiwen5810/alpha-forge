# Feasibility of a separate LangChain implementation

Assessment date: 2026-10-09. This is a source and documentation investigation,
not a tested implementation. No dependencies were installed or model calls made.

The objective is a standalone implementation from scratch using LangChain.
The existing application is a functional reference, not a runtime dependency.
There is no requirement to reuse its code, transcript schema, query protocol,
projectors, or provider types.

This is feasible. Start with LangChain `create_agent`, `ChatOpenAI`, freshly
implemented tools and middleware, and LangGraph persistence. Use a custom
LangGraph workflow only when functional acceptance checks establish a need.
The earlier assessment overemphasized preserving the current architecture.

## Functional mapping

| Capability | Fresh implementation | Difficulty |
| --- | --- | --- |
| Streaming chat, tool progress, usage | Agent streaming and new presentation adapter | Moderate |
| File reading/writing, Bash, calculator | New schema-defined LangChain tools | Moderate |
| Approvals | Human-in-the-loop or custom approval middleware | Moderate |
| Queued prompts | Async FIFO outside the agent | Low |
| Slash commands and model listing | New controller and OpenAI catalog service | Low to moderate |
| Saved conversations | Local LangGraph checkpointer and session metadata | Moderate |
| History-only resume | Restoration policy distinct from graph continuation | High |
| Large-result previews and paging | Raw-result store, preview middleware, retrieval tool | Moderate |
| Terminal layout, scrolling, focus | Newly implemented prompt-toolkit UI | Moderate to high |
| Layered configuration | New resolver preserving documented precedence | Low |
| Custom tools, hooks, context policies | Tool registration and middleware | Moderate |

LangChain provides the model/tool loop, message state, and middleware. See
[agents](https://docs.langchain.com/oss/python/langchain/agents).
Token, update, and custom progress streams can drive a new UI event adapter:
see [streaming](https://docs.langchain.com/oss/python/langchain/streaming).

## Standalone package

Create `packages/alpha-forge-langchain/` with its own `pyproject.toml`, namespace
`alpha_forge_langchain`, command, tests, and dependency resolution. Do not depend
on `alpha-forge` or require a shared-core extraction.

```text
CLI/config -> terminal/controller -> prompt FIFO -> LangChain agent
                  |                                  |
            session manager                  tools + middleware
                  |                                  |
        checkpoint/history store              approval service
                  |
           raw-result store
```

Suggested new modules: config, agent, tools, middleware, sessions, application,
and ui. Use `langchain`, `langchain-openai`, and a local LangGraph checkpointer.
Declare `langgraph` and `langchain-core` directly if importing their APIs.
SQLite is a candidate for local persistence; validate async support and Python
3.14 dependency compatibility in an isolated environment.

Framework message state can become the conversation authority. Additional
storage can hold command outcomes, presentation history, and full tool results.
Define one owner for each fact and a recovery strategy. Do not preserve the old
ledger/projector design merely for architectural similarity.

## Behavior requiring deliberate design

- Bash and writes must wait for approval. Denial prevents invocation and produces
  a usable error result; shutdown resolves pending approvals. Check whether built-in
  approval batching suits the terminal, and use custom middleware if necessary.
- Opening a saved conversation must wait for a new prompt, without executing an
  unfinished graph. Distinguish approval continuation from restoration after a
  restart. Normalize incomplete tool exchanges and mark unknown outcomes without
  retrying side effects.
- Store full tool output separately from model previews, and provide paging by
  stable result ID scoped to the active session, including after restart.
- Preserve meaningful tool behavior: input validation, replacement match-count
  checks, command timeouts, and predictable errors. Execute dependent calls in order.
- Checkpoints cannot guarantee exactly-once external actions across a crash between
  execution and persistence. Avoid automatic retries of unknown side effects.
- Define loop limits explicitly; framework recursion limits need not reproduce
  the current ten-round implementation exactly.

LangGraph interrupts restart their containing node from its beginning. Keep
side effects after approval and prevent duplicate writes during replay. See
[interrupt semantics](https://docs.langchain.com/oss/python/langgraph/interrupts).

If strict tool ordering and durable-result-before-next-action remain acceptance
requirements, implement explicit gates, possibly with a sequential tool node or
custom graph. This would be a newly built framework workflow, not an adapter
around the current agent runtime.

## OpenAI and gateway behavior

Keep OpenAI as the default. Reimplement the documented configuration layers
and pass model, key, timeout, and endpoint explicitly to `ChatOpenAI` or its
configured SDK client. Account for LangChain's additional endpoint environment
fallbacks so they cannot bypass this app's documented merge behavior. Never
introduce an API-key command-line flag.

Keep LiteLLM opt-in through `OPENAI_BASE_URL`. S3-compatible cold storage remains
gateway infrastructure. It need not become an application dependency.

There is a specific compatibility risk: the current adapter extracts gateway
reasoning fields, whereas `ChatOpenAI` documents that it does not preserve
non-standard provider fields. Streaming usage defaults can also differ when a
custom endpoint is present. Retaining those extensions may need a narrow custom
adapter; verify refusal, reasoning, cached tokens, tool-call chunks, invalid tool
arguments, and finish reasons with fixtures. Model listing still needs a catalog
service. See the official
[ChatOpenAI integration](https://docs.langchain.com/oss/python/integrations/chat/openai).

## Implementation sequence and validation

1. Resolve dependencies on Python 3.14. Build a headless `create_agent` with newly
   written tools, configuration, approval middleware, and fake-model tests.
2. Implement local persistence and session restoration. Verify that opening a
   saved session invokes neither the model nor tools, including interrupted cases.
3. Implement raw-result storage, previews, and paging across restarts.
4. Build the terminal UI, FIFO, slash commands, session switching, and shutdown.
5. Run OpenAI smoke tests and functional acceptance scenarios. Add a custom graph
   only for requirements the standard agent cannot satisfy satisfactorily.

Use current tests as a source of behavioral scenarios, not mandatory internal
contracts. Write new acceptance tests for tool semantics, approvals, queued
prompts, session switching, interrupted actions, persistence failure, layered
configuration, and large-result retrieval.

A basic agent is straightforward; the full terminal application is a moderate
rewrite. Most custom work lies in terminal interaction, session recovery, and
large-result management. LangChain replaces the handwritten loop and provides
extension primitives; it does not supply all product behavior automatically.
There is no demonstrated answer-quality or token-cost benefit from changing
frameworks alone. See [custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
for the primary extension mechanism.

This remains a source/documentation assessment, not a tested implementation.
