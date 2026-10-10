"""Thin LLM wrapper: one text-in, text-out call, plus JSON extraction."""

import json
import re
from typing import Callable, Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    """Claude via the Anthropic SDK. Credentials come from the environment
    (ANTHROPIC_API_KEY, or an `ant auth login` profile)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 64000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        # Streaming: scripts are long outputs. Server-side fallback re-runs a
        # refused request on another model instead of failing the item.
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        if message.stop_reason == "max_tokens":
            raise RuntimeError("output truncated at max_tokens")
        return "".join(block.text for block in message.content if block.type == "text")


class FunctionLLM:
    """Wrap a plain function (tests, or a local model behind your own code)."""

    def __init__(self, fn: Callable[[str, str], str]):
        self.fn = fn

    def complete(self, system: str, user: str) -> str:
        return self.fn(system, user)


def extract_json(text: str):
    """Parse the last ```json fenced block, else the outermost {...}."""
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    for candidate in reversed(blocks):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start: end + 1])
    raise ValueError("no JSON found in model output")
