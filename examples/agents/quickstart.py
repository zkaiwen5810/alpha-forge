import asyncio

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_tracing_disabled,
)

set_tracing_disabled(True)

# agent = Agent(
#     name="History tutor",
#     instructions="You answer history questions clearly and concisely",
#     model="Qwen3.6-35B-A3B",
# )


# async def main() -> None:
#     result = await Runner.run(agent, "When did the Roman Empire fall?")
#     print(result.final_output)


# @function_tool
# def history_fun_fact() -> str:
#     """Return a short history fact."""
#     return "Sharks are older than tree."


# agent = Agent(
#     name="History tutor",
#     instructions="Answer history questions clearly. Use history_fun_fact when it helps.",
#     tools=[history_fun_fact],
#     model="Qwen3.6-35B-A3B",
# )


# async def main() -> None:
#     result = await Runner.run(
#         agent,
#         "Tell me something surprising about ancient life on Earth.",
#     )
#     print(result.final_output)

history_tutor = Agent(
    name="history_tutor",
    handoff_description="Specialist for history questions.",
    instructions="Answer history questions clearly and concisely.",
    model="Qwen3.6-35B-A3B",
)

math_tutor = Agent(
    name="math_tutor",
    handoff_description="Specialist for math questions.",
    instructions="Explain math step by step and include worked examples.",
    model="Qwen3.6-35B-A3B",
)

triage_agent = Agent(
    name="homework_triage",
    instructions="Route each homework question to the right specialist.",
    handoffs=[history_tutor, math_tutor],
    model="Qwen3.6-35B-A3B",
)


async def main() -> None:
    # LiteLLM's OpenAI-compatible Responses support is incomplete for this handoff flow.
    # Force Chat Completions so the example works against a LiteLLM gateway.
    set_default_openai_api("chat_completions")

    result = await Runner.run(
        triage_agent,
        "Who was the first president of the United States?",
    )
    print(result.final_output)
    print(result.last_agent.name)


if __name__ == "__main__":
    asyncio.run(main())
