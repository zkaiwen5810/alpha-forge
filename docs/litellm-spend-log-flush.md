# LiteLLM Spend Log Flush

This note documents the LiteLLM persistence behavior observed while testing
`previous_response_id` with the Responses API through the LiteLLM gateway.

## Summary

OpenAI-native conversation continuity uses `store=True` on the first response and
`previous_response_id` on the follow-up request.

When LiteLLM emulates the Responses API through a chat-completions backend, it
rehydrates conversation state from its proxy spend logs. Those spend logs are not
guaranteed to be visible immediately after the first response returns. A follow-up
request can therefore race the LiteLLM logging pipeline and see no previous
messages.

## Persistence Path

For the LiteLLM proxy path:

1. The client calls `/v1/responses`.
2. LiteLLM transforms the Responses request into a chat-completions request when
   the backend does not natively support Responses.
3. The model response is returned to the client.
4. LiteLLM logging callbacks build spend-log payloads.
5. Spend-log writes are queued in memory.
6. Background jobs flush queued spend logs into Postgres.
7. Cold storage, such as Cloudflare R2 through the S3-compatible callback, stores
   larger prompt/response payloads and can be referenced from spend-log metadata.

Cold storage success alone does not prove rehydration is ready. The session
handler first needs the relevant `LiteLLM_SpendLogs` row in Postgres.

## Rehydration Path

For a follow-up request with `previous_response_id`, LiteLLM:

1. Decodes the LiteLLM-managed `resp_...` id to the original response id.
2. Queries `LiteLLM_SpendLogs` for a row whose `request_id` matches that decoded
   response id.
3. Reads the matching row's `session_id`.
4. Loads all spend-log rows for that `session_id`, ordered by end time.
5. Rebuilds chat messages from the stored request and response payloads.
6. Prepends those rebuilt messages to the new user input.

If the spend-log row has not been flushed yet, the query returns no history. The
second model call still succeeds, but it only receives the new user message.

## Impact

The failure mode is subtle because there may be no LiteLLM error log. The second
turn simply behaves as if no previous conversation exists.

A typical symptom:

- First turn: `tell me a joke`
- Second turn: `explain why this is funny`
- Failed rehydration: model asks what joke you mean
- Successful rehydration: model explains the previous joke

The LiteLLM request log can also be misleading. The logged `messages` field may
show the original client request, not necessarily the final internally rehydrated
message list.

## Mitigations

For examples and local testing, add a short delay before the follow-up request:

```python
response = client.responses.create(
    model="...",
    input="tell me a joke",
    store=True,
)

sleep(3)

second_response = client.responses.create(
    model="...",
    previous_response_id=response.id,
    input="explain why this is funny.",
)
```

For production, prefer retry or polling over a fixed sleep. The application can
retry the follow-up when the model clearly did not receive prior context, or it
can query LiteLLM/Postgres directly if that is acceptable in the deployment.

The most robust application-level option is to keep conversation history in the
application and send explicit message history instead of depending on LiteLLM
log rehydration timing.

## Tuning Knobs

LiteLLM exposes environment variables that can reduce the flush window:

```env
PROXY_BATCH_WRITE_AT=1
SPEND_LOG_QUEUE_POLL_INTERVAL=0.25
SPEND_LOG_QUEUE_SIZE_THRESHOLD=1
```

These settings increase DB write frequency and should be measured before use in
production.

- `PROXY_BATCH_WRITE_AT`: scheduled DB flush interval. The checked LiteLLM source
  default is `10` seconds.
- `SPEND_LOG_QUEUE_POLL_INTERVAL`: spend-log queue monitor interval. The checked
  LiteLLM source default is `2.0` seconds.
- `SPEND_LOG_QUEUE_SIZE_THRESHOLD`: queue size that triggers queue processing.
  Setting it to `1` allows a single queued log to be processed on the next poll.

## Checked Sources

- LiteLLM Responses API docs:
  https://docs.litellm.ai/docs/response_api
- LiteLLM proxy logging docs:
  https://docs.litellm.ai/docs/proxy/logging
- LiteLLM session handler source:
  https://github.com/BerriAI/litellm/blob/main/litellm/responses/litellm_completion_transformation/session_handler.py
- LiteLLM Responses-to-chat transformation source:
  https://github.com/BerriAI/litellm/blob/main/litellm/responses/litellm_completion_transformation/handler.py
- LiteLLM constants for flush defaults:
  https://github.com/BerriAI/litellm/blob/main/litellm/constants.py
- LiteLLM proxy spend-log queue and update source:
  https://github.com/BerriAI/litellm/blob/main/litellm/proxy/utils.py
- Related LiteLLM issue discussing `previous_response_id` and session handling:
  https://github.com/BerriAI/litellm/issues/15930
