"""
Shared JSON repair utilities for ARGUS.

The primary problem: Gemini (and other LLMs) frequently produce JSON with
unescaped double-quotes inside string values, e.g.:

    {"type": "Lack of a definition for "general intelligence" impedes..."}

This is invalid JSON.  The helpers here attempt progressively aggressive
repairs so that downstream code can parse the LLM output.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _fix_unescaped_quotes(text: str) -> str:
    r"""Escape double-quotes that appear *inside* JSON string values.

    Strategy: walk char-by-char tracking whether we're inside a JSON string.
    A quote is "structural" (opens/closes a string) when it is preceded by
    an even number of backslashes and follows a structural token
    (``:``, ``,``, ``[``, ``{``, or another closing quote).  Any other
    quote inside a string is an unescaped inner quote → replace with ``\"``.
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch == "\\" and in_string:
            # Escaped character — pass through both chars
            out.append(ch)
            if i + 1 < n:
                i += 1
                out.append(text[i])
            i += 1
            continue

        if ch == '"':
            if not in_string:
                # Opening quote — always structural
                in_string = True
                out.append(ch)
            else:
                # Are we at the real end of this string value?
                # Look ahead: skip whitespace, the next non-ws char must be
                # a structural JSON token  , } ] :  or end-of-string.
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                next_ch = text[j] if j < n else ""
                if next_ch in ("", ",", "}", "]", ":"):
                    # Structural close
                    in_string = False
                    out.append(ch)
                else:
                    # Inner quote — escape it
                    out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _balance_brackets(candidate: str) -> str:
    """Append missing ``]`` and ``}`` to balance brackets."""
    candidate += "]" * max(0, candidate.count("[") - candidate.count("]"))
    candidate += "}" * max(0, candidate.count("{") - candidate.count("}"))
    return candidate


def repair_json(candidate: str) -> str:
    """Try increasingly aggressive repairs and return the first parseable result.

    Order:
      1. Balance brackets only.
      2. Fix unescaped inner quotes, then balance.
      3. Walk backwards to the last cleanly-closed element, then balance.
    """
    # Fast path — already valid
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    # 1. Simple bracket balance
    attempt = _balance_brackets(candidate)
    try:
        json.loads(attempt)
        return attempt
    except json.JSONDecodeError:
        pass

    # 2. Fix unescaped quotes + balance
    attempt = _balance_brackets(_fix_unescaped_quotes(candidate))
    try:
        json.loads(attempt)
        return attempt
    except json.JSONDecodeError:
        pass

    # 3. Walk backwards to last cleanly-closed element
    fixed = _fix_unescaped_quotes(candidate)
    for pos in range(len(fixed) - 1, 0, -1):
        if fixed[pos] in "}]":
            chunk = _balance_brackets(fixed[: pos + 1])
            try:
                json.loads(chunk)
                return chunk
            except json.JSONDecodeError:
                continue

    # Nothing worked — return bracket-balanced version (caller will handle error)
    return _balance_brackets(_fix_unescaped_quotes(candidate))


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON **object** from LLM output.

    Handles markdown fences, preamble text, unescaped quotes, and truncation.
    Raises ``ValueError`` if no object can be recovered.
    """
    text = text.strip()

    # 1. Strip markdown fences (complete or truncated)
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        inner = m.group(1).strip()
    else:
        m = re.search(r"```(?:json)?\s*([\s\S]+)", text)
        inner = m.group(1).strip() if m else text

    # 2. Direct parse
    try:
        obj = json.loads(inner)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 3. Repair then parse
    repaired = repair_json(inner)
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 4. Find first '{' to last '}' in original text
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        snippet = text[first: last + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            repaired = repair_json(snippet)
            try:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

    # 5. From first '{' onward (truncated)
    if first != -1:
        repaired = repair_json(text[first:])
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot extract JSON object from LLM output: {text[:200]}")


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON **array** from LLM output.

    Handles markdown fences, preamble, unescaped quotes, truncation,
    and single-object responses (auto-wrapped).
    Raises ``ValueError`` if nothing can be recovered.
    """
    text = text.strip()

    def _as_list(obj: Any) -> list[dict[str, Any]]:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        raise TypeError(f"Expected list or dict, got {type(obj).__name__}")

    # 1. Strip markdown fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        inner = m.group(1).strip()
    else:
        m = re.search(r"```(?:json)?\s*([\s\S]+)", text)
        inner = m.group(1).strip() if m else text

    # 2. Direct parse
    try:
        return _as_list(json.loads(inner))
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Repair then parse
    repaired = repair_json(inner)
    try:
        return _as_list(json.loads(repaired))
    except (json.JSONDecodeError, TypeError):
        pass

    # 4. Find first '[' to last ']'
    first_sq = text.find("[")
    last_sq = text.rfind("]")
    if first_sq != -1 and last_sq > first_sq:
        snippet = text[first_sq: last_sq + 1]
        try:
            return _as_list(json.loads(snippet))
        except (json.JSONDecodeError, TypeError):
            repaired = repair_json(snippet)
            try:
                return _as_list(json.loads(repaired))
            except (json.JSONDecodeError, TypeError):
                pass

    # 5. Find first '{' to last '}' (single object)
    first_br = text.find("{")
    last_br = text.rfind("}")
    if first_br != -1 and last_br > first_br:
        snippet = text[first_br: last_br + 1]
        try:
            return _as_list(json.loads(snippet))
        except (json.JSONDecodeError, TypeError):
            repaired = repair_json(snippet)
            try:
                return _as_list(json.loads(repaired))
            except (json.JSONDecodeError, TypeError):
                pass

    # 6. From first bracket/brace onward (truncated)
    start = min(
        first_sq if first_sq != -1 else len(text),
        first_br if first_br != -1 else len(text),
    )
    if start < len(text):
        repaired = repair_json(text[start:])
        try:
            return _as_list(json.loads(repaired))
        except (json.JSONDecodeError, TypeError):
            pass

    raise ValueError(f"Cannot extract JSON array from LLM output: {text[:200]}")
