# LangChain shell sandbox design (deferred)

Status: design only, not implemented. Recorded 2026-10-14.

The LangChain package currently uses Deep Agents' `LocalShellBackend`: file
operations and shell commands run directly in the application's environment.
The shell starts in the launch directory for each call and inherits the parent
environment. This document preserves the alternative OS sandbox design for
future implementation.

## Integration

Keep the existing `create_agent`, streaming, and SQLite recovery runtime. Replace
the local shell backend with `FilesystemBackend` for host file tools, omitting
`execute` from its filesystem middleware allowlist. Add LangChain's
`ShellToolMiddleware` with `CodexSandboxExecutionPolicy`, exposing the native
`shell` tool (`command` or `restart`).

Use a workspace-write policy with shell network access disabled and no additional
writable roots. Normal workspace and temporary-directory writes remain available;
reads follow the Codex policy, including its protected paths and managed
requirements. Resolve the launch directory once and use it for file tools and
shell startup. Do not silently fall back to unrestricted host execution if the
CLI is missing, its version is unsupported, or sandbox startup fails.

The boundary is **shell execution only**. Existing host file tools can still
read and edit normal host paths, and web fetch runs in the application. This is
not isolation of the whole agent. Moving all file operations into a sandbox
would require a Deep Agents sandbox backend and a separate workspace design.

## CLI compatibility and prerequisites

Inspection used Codex CLI 0.162.1 and LangChain 1.4.4. The inspected LangChain
policy builds `codex sandbox linux ... -- COMMAND` or the macOS equivalent.
The installed CLI instead accepts `codex sandbox ... -- COMMAND`, automatically
selecting the platform. A small execution-policy adapter must reconcile this
command shape; verify the supported CLI and LangChain versions before shipping.
Use explicit sandbox configuration overrides rather than inheriting a user's
broader default policy.

Codex must be on `PATH`. Linux requires a working bubblewrap/user-namespace setup;
macOS uses Seatbelt. Nested container restrictions can prevent startup even when
the CLI is installed. A no-op launch check succeeded outside this workspace's
outer command sandbox during investigation. It failed inside that outer sandbox
because the app-server socket directory could not satisfy its ownership/mode
requirements. This check does not establish full runtime integration.

## Session ownership, streaming, and recovery

Use one persistent shell session per input. Commands within that input share
directory and environment changes; the next input starts a new session. Shell
processes and environment state are never restored from a conversation checkpoint.

The middleware stores live process resources in `shell_session_resources`, an
untracked/private graph-state field. These objects contain locks and cannot be
deep-copied. Before the runtime copies stream updates or terminal state, remove
that field from state-bearing dictionaries and omit updates containing only
private resources. Preserve ordinary messages, tool results, and other state.
Do not filter arbitrary user tool payloads by dictionary key.

Wrap native middleware with explicit per-input resource ownership. Close the
graph stream and release shell resources in a `finally` block on completion,
model/tool failure, cancellation, early stream closure, and runtime shutdown.
Do not rely only on `after_agent` or garbage collection for cleanup. Make cleanup
idempotent because native successful-run cleanup may already have executed.

Keep completed-checkpoint promotion unchanged. Restoring history must never
start a shell or replay commands. Failed turns can leave external effects;
recovery does not undo them or provide exactly-once execution.

## Future Python interface

Add `shell_execution_policy: BaseExecutionPolicy | None = None` to `AgentRuntime`.
`None` selects the supported OS sandbox policy. Explicit policy injection supports
offline tests and caller-controlled alternative execution environments. Keep
default CLI operation sandboxed, with no automatic fallback or API-key flag.
Any future CLI settings must follow the repository's layered configuration rules.

## Acceptance tests

- Verify the CLI launch arguments against supported versions and platforms.
- Verify workspace writes, blocked writes outside allowed roots, and disabled
  network access with real sandbox integration tests.
- Verify command ordering, exit codes, timeout, truncation, `restart`, state shared
  within an input, and reset between inputs.
- Verify cleanup after success, failure, cancellation, and early stream closure,
  including long-running commands and child processes.
- Verify no process handles appear in public stream/state data or checkpoints;
  completed history and restoration never invoke the shell.
- Verify missing CLI and unsupported sandbox environments fail clearly without
  switching to host execution. Use injected policies for ordinary offline tests.

## References

- [LangChain shell middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in#shell-tool)
- [Deep Agents backends](https://docs.langchain.com/oss/python/deepagents/backends)
- [Codex sandbox](https://learn.chatgpt.com/docs/sandboxing)
- [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
- [Codex developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
