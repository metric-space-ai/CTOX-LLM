"""Strict conversion from source conversations to the pinned model template."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ValueError("tool call has no non-empty function name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(f"tool call {name!r} has invalid JSON arguments") from error
    if not isinstance(arguments, dict):
        raise ValueError(f"tool call {name!r} arguments must decode to an object")
    return {"name": name, "arguments": arguments}


def normalize_content(value: Any, message_index: int, field: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (dict, list, bool, int, float)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    raise ValueError(
        f"message {message_index} {field} has unsupported type {type(value).__name__}"
    )


def normalize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    normalized = deepcopy(messages)
    for index, message in enumerate(normalized):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"message {index} has no role")
        if "content" not in message:
            raise ValueError(f"message {index} has no content field")
        message["content"] = normalize_content(message["content"], index, "content")
        if "reasoning_content" in message:
            message["reasoning_content"] = normalize_content(
                message["reasoning_content"], index, "reasoning_content"
            )
        tool_calls = message.get("tool_calls")
        if tool_calls:
            if not isinstance(tool_calls, list):
                raise ValueError(f"message {index} tool_calls is not a list")
            message["tool_calls"] = [normalize_tool_call(call) for call in tool_calls]
        else:
            message.pop("tool_calls", None)
    return normalized


def render_record(tokenizer: Any, record: dict[str, Any]) -> str:
    if "messages" not in record:
        return str(record["prompt"])
    messages = normalize_messages(record["messages"])
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if record.get("tools"):
        kwargs["tools"] = record["tools"]
    return tokenizer.apply_chat_template(messages, **kwargs)
