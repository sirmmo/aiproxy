"""Reflex backends: answers without a model behind them.

"System one" responders. A reflex backend never makes an HTTP call; it answers
every request reflexively with one *polarity*:

* ``no``     – "No.", ``false``, ``0``, empty lists, no tool calls.
* ``yes``    – "Yes.", ``true``, ``1``, one-element lists, calls the offered tool.
* ``maybe``  – "Maybe.", ``null`` where the schema allows it, ``0.5``, coin flips.
* ``random`` – one of the three above, picked per request by ``weights``.

The answer adapts to the *shape* of the request so clients always get a
well-formed response: plain text for chat, a filled-in document for
``response_format`` (``json_object`` / ``json_schema``), tool calls with
schema-valid arguments when tools are offered (``yes`` only, once per turn),
and SSE deltas when streaming. Useful as a deterministic stand-in for an LLM in
tests and demos, as a "no as a service" endpoint, and as a cheap negative
baseline for evaluations.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import uuid
from typing import Any, AsyncIterator, Optional

from ..config import BackendConfig
from .base import Completion, StreamEvent, ToolCall

POLARITIES = ("no", "yes", "maybe")

_DEFAULT_TEXT = {"no": "No.", "yes": "Yes.", "maybe": "Maybe."}

# Enum members a polarity prefers, matched case-insensitively from the start.
_ENUM_PREFERENCE = {
    "no": re.compile(r"^(no|false|never|none|deny|denied|reject|rejected|off|disabled?|0)\b", re.I),
    "yes": re.compile(r"^(yes|true|always|ok|okay|accept|accepted|allow|allowed|on|enabled?|1)\b", re.I),
    "maybe": re.compile(
        r"^(maybe|perhaps|unknown|unsure|uncertain|undecided|null|neutral|partial|defer|deferred|pending|tbd|later|hold|abstain|skip)",
        re.I,
    ),
}

_MAX_DEPTH = 12


# --------------------------------------------------------------------------- #
# Schema filling
# --------------------------------------------------------------------------- #
class SchemaFiller:
    """Produce a JSON value that validates against a JSON schema and carries
    one polarity: every boolean, number, string, enum and array in the document
    says "no" (or "yes", or "maybe")."""

    def __init__(self, root: dict[str, Any], polarity: str, rng: random.Random):
        self._root = root
        self._polarity = polarity
        self._rng = rng

    # --- refs --------------------------------------------------------------
    def _resolve(self, schema: Any) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {}
        ref = schema.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return schema
        node: Any = self._root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                return {}
        merged = {k: v for k, v in schema.items() if k != "$ref"}
        if isinstance(node, dict):
            merged = {**node, **merged}
        return self._resolve(merged) if "$ref" in merged else merged

    # --- entry -------------------------------------------------------------
    def fill(self, schema: Any, depth: int = 0) -> Any:
        schema = self._resolve(schema)
        if depth > _MAX_DEPTH:
            return None
        if "const" in schema:
            return schema["const"]
        if isinstance(schema.get("enum"), list) and schema["enum"]:
            return self._pick_enum(schema["enum"])
        for key in ("anyOf", "oneOf"):
            options = schema.get(key)
            if isinstance(options, list) and options:
                return self.fill(self._pick_option(options), depth + 1)
        if isinstance(schema.get("allOf"), list) and schema["allOf"]:
            merged: dict[str, Any] = {}
            for part in schema["allOf"]:
                merged.update(self._resolve(part))
            return self.fill(merged, depth + 1)

        type_ = self._pick_type(schema.get("type"), schema)
        if type_ == "object":
            return self._fill_object(schema, depth)
        if type_ == "array":
            return self._fill_array(schema, depth)
        if type_ == "boolean":
            return self._boolean()
        if type_ == "integer":
            return self._number(schema, integer=True)
        if type_ == "number":
            return self._number(schema, integer=False)
        if type_ == "string":
            return self._string(schema)
        return None

    # --- choices -----------------------------------------------------------
    def _pick_type(self, type_: Any, schema: dict[str, Any]) -> Optional[str]:
        if isinstance(type_, list):
            types = [t for t in type_ if isinstance(t, str)]
            if not types:
                return None
            if "null" in types and (self._polarity == "maybe" or len(types) == 1):
                return "null"
            non_null = [t for t in types if t != "null"]
            return non_null[0] if non_null else "null"
        if isinstance(type_, str):
            return type_
        # Untyped: infer from the keywords present.
        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
        return None

    def _pick_option(self, options: list[Any]) -> Any:
        resolved = [self._resolve(o) for o in options]
        if self._polarity == "maybe":
            for o in resolved:
                if o.get("type") == "null" or (isinstance(o.get("type"), list) and "null" in o["type"]):
                    return o
        non_null = [o for o in resolved if o.get("type") != "null"] or resolved
        return non_null[0]

    def _pick_enum(self, values: list[Any]) -> Any:
        rx = _ENUM_PREFERENCE[self._polarity]
        for v in values:
            if isinstance(v, str) and rx.search(v.strip()):
                return v
            if isinstance(v, bool) and v is (self._polarity == "yes") and self._polarity != "maybe":
                return v
            if (
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and v == (1 if self._polarity == "yes" else 0)
                and self._polarity != "maybe"
            ):
                return v
        if self._polarity == "maybe":
            if None in values:
                return None
            return values[len(values) // 2]
        return values[0]

    # --- scalars -----------------------------------------------------------
    def _boolean(self) -> bool:
        if self._polarity == "yes":
            return True
        if self._polarity == "no":
            return False
        return self._rng.random() < 0.5

    def _number(self, schema: dict[str, Any], integer: bool) -> Any:
        lo = schema.get("minimum", schema.get("exclusiveMinimum"))
        hi = schema.get("maximum", schema.get("exclusiveMaximum"))
        lo = lo if isinstance(lo, (int, float)) and not isinstance(lo, bool) else None
        hi = hi if isinstance(hi, (int, float)) and not isinstance(hi, bool) else None
        if self._polarity == "no":
            value: float = lo if lo is not None else 0
        elif self._polarity == "yes":
            value = hi if hi is not None else 1
        else:
            if lo is not None and hi is not None:
                value = (lo + hi) / 2
            else:
                value = 0.5
        if integer:
            value = int(math.floor(value)) if value >= 0 else int(math.ceil(value))
            if lo is not None and value < lo:
                value = int(math.ceil(lo))
            if hi is not None and value > hi:
                value = int(math.floor(hi))
        return value

    def _string(self, schema: dict[str, Any]) -> str:
        fmt = schema.get("format")
        if fmt in ("date", "date-time", "time"):
            base = {"date": "1970-01-01", "date-time": "1970-01-01T00:00:00Z", "time": "00:00:00"}[fmt]
            if self._polarity == "yes":
                base = {"date": "9999-12-31", "date-time": "9999-12-31T23:59:59Z", "time": "23:59:59"}[fmt]
            return base
        if fmt == "uuid":
            return str(uuid.UUID(int=0 if self._polarity == "no" else (2**128 - 1 if self._polarity == "yes" else 2**127)))
        text = _DEFAULT_TEXT[self._polarity].rstrip(".").lower()
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(text) < min_len:
            text = (text + " ") * (min_len // (len(text) + 1) + 1)
            text = text[:min_len]
        return text

    # --- compounds ---------------------------------------------------------
    def _fill_object(self, schema: dict[str, Any], depth: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        props = schema.get("properties")
        if isinstance(props, dict):
            for name, sub in props.items():
                out[name] = self.fill(sub, depth + 1)
        return out

    def _fill_array(self, schema: dict[str, Any], depth: int) -> list[Any]:
        items = schema.get("items", {})
        if isinstance(items, list):  # tuple validation
            return [self.fill(s, depth + 1) for s in items]
        min_items = schema.get("minItems")
        min_items = min_items if isinstance(min_items, int) and min_items > 0 else 0
        count = {"no": 0, "yes": 1, "maybe": 0}[self._polarity]
        count = max(count, min_items)
        return [self.fill(items, depth + 1) for _ in range(count)]


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
def _estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, len(text) // 4) if text else 0


def _split_words(text: str) -> list[str]:
    """Split into deltas that re-join to the original (whitespace attached)."""
    return re.findall(r"\S+\s*|\s+", text) or [text]


class ReflexBackend:
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.kind = cfg.kind
        self._rng = random.Random(cfg.seed)

    # --- decision ----------------------------------------------------------
    def polarity(self) -> str:
        if self.kind in POLARITIES:
            return self.kind
        weights = [max(0.0, float(self.cfg.weights.get(p, 1.0))) for p in POLARITIES]
        if sum(weights) <= 0:
            weights = [1.0, 1.0, 1.0]
        return self._rng.choices(POLARITIES, weights=weights, k=1)[0]

    def _text(self, polarity: str) -> str:
        if self.cfg.text is not None:
            return self.cfg.text
        return _DEFAULT_TEXT[polarity]

    def _content(self, polarity: str, params: dict[str, Any]) -> str:
        rf = params.get("response_format")
        if not isinstance(rf, dict):
            return self._text(polarity)
        rtype = rf.get("type")
        if rtype == "json_schema":
            spec = rf.get("json_schema") if isinstance(rf.get("json_schema"), dict) else {}
            schema = spec.get("schema") if isinstance(spec.get("schema"), dict) else rf.get("schema")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {"answer": {"type": "boolean"}}}
            value = SchemaFiller(schema, polarity, self._rng).fill(schema)
            return json.dumps(value, ensure_ascii=False)
        if rtype == "json_object":
            answer = {"no": False, "yes": True, "maybe": None}[polarity]
            return json.dumps({"answer": answer, "text": self._text(polarity)}, ensure_ascii=False)
        return self._text(polarity)

    def _tool_calls(
        self, polarity: str, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]]
    ) -> list[ToolCall]:
        """``yes`` says yes to the offered tools, but only once per turn: after
        a tool result is in view it answers in words so the agent loop ends."""
        if polarity != "yes" or not tools or self.cfg.tool_calls == "none":
            return []
        if messages and messages[-1].get("role") == "tool":
            return []
        chosen = tools if self.cfg.tool_calls == "all" else tools[:1]
        calls: list[ToolCall] = []
        for i, tool in enumerate(chosen):
            fn = tool.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            schema = fn.get("parameters") or {"type": "object", "properties": {}}
            args = SchemaFiller(schema, polarity, self._rng).fill(schema)
            if not isinstance(args, dict):
                args = {}
            calls.append(
                ToolCall(id=f"call_reflex_{i}", name=name, arguments=json.dumps(args, ensure_ascii=False))
            )
        return calls

    def _answer(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> Completion:
        polarity = self.polarity()
        tool_calls = self._tool_calls(polarity, messages, tools)
        content = None if tool_calls else self._content(polarity, params)
        prompt_tokens = sum(_estimate_tokens(m.get("content")) for m in messages)
        completion_tokens = _estimate_tokens(content) + sum(_estimate_tokens(tc.arguments) for tc in tool_calls)
        return Completion(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            extras={"x_reflex": {"kind": self.kind, "polarity": polarity}},
        )

    # --- Backend protocol --------------------------------------------------
    async def complete(
        self,
        model: Optional[str],
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> Completion:
        if self.cfg.latency:
            await asyncio.sleep(self.cfg.latency)
        return self._answer(messages, tools, params)

    async def stream(
        self,
        model: Optional[str],
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        if self.cfg.latency:
            await asyncio.sleep(self.cfg.latency)
        completion = self._answer(messages, tools, params)
        for i, tc in enumerate(completion.tool_calls):
            yield StreamEvent(type="tool_call", index=i, id=tc.id, name=tc.name, arguments="")
            yield StreamEvent(type="tool_call", index=i, arguments=tc.arguments)
        if completion.content:
            for j, piece in enumerate(_split_words(completion.content)):
                if j and self.cfg.stream_delay:
                    await asyncio.sleep(self.cfg.stream_delay)
                yield StreamEvent(type="content", text=piece)
        yield StreamEvent(type="finish", finish_reason=completion.finish_reason)
        yield StreamEvent(type="usage", usage=completion.usage)

    async def aclose(self) -> None:
        pass
