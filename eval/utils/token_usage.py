"""
Accumulate OpenAI-compatible usage.* from chat completions and embeddings.

First level: which env role (VLM_MAIN_MODEL / VLM_RESOLVE_MODEL / VLM_EMBEDDING_MODEL).
Second level: models.json config name (e.g. gpt-5-mini). Same name in different roles
does not merge. Reset once per Game.run() / task.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Dict, Optional

# Keys match environment variable names for clarity in result.json
SCOPE_MAIN = "VLM_MAIN_MODEL"
SCOPE_RESOLVE = "VLM_RESOLVE_MODEL"
SCOPE_EMBEDDING = "VLM_EMBEDDING_MODEL"

_lock = threading.Lock()
# scope -> model_config_key -> counters
_usage: Dict[str, Dict[str, Dict[str, Any]]] = {}


def reset() -> None:
    """Clear counters for the current task (call at start of Game.run)."""
    global _usage
    with _lock:
        _usage = {}


def _empty_row() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "chat_calls": 0,
        "embedding_tokens": 0,
        "embedding_calls": 0,
    }


def _ensure_row(scope: str, model_key: str) -> Dict[str, Any]:
    if scope not in _usage:
        _usage[scope] = {}
    if model_key not in _usage[scope]:
        _usage[scope][model_key] = _empty_row()
    return _usage[scope][model_key]


def record_chat_completion(scope: str, model_key: Optional[str], response: Any) -> None:
    """Add tokens from chat.completions.create (main or resolve)."""
    if not scope or not model_key:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    pt = int(getattr(usage, "prompt_tokens", None) or 0)
    ct = int(getattr(usage, "completion_tokens", None) or 0)
    tt = getattr(usage, "total_tokens", None)
    if tt is None:
        tt = pt + ct
    else:
        tt = int(tt)
    with _lock:
        row = _ensure_row(scope, model_key)
        row["prompt_tokens"] += pt
        row["completion_tokens"] += ct
        row["total_tokens"] += tt
        row["chat_calls"] += 1


def record_embedding_usage(scope: str, model_key: Optional[str], response: Any) -> None:
    """Add tokens from embeddings.create."""
    if not scope or not model_key:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    et = getattr(usage, "total_tokens", None)
    if et is None:
        et = int(getattr(usage, "prompt_tokens", None) or 0)
    else:
        et = int(et)
    with _lock:
        row = _ensure_row(scope, model_key)
        row["embedding_tokens"] += et
        row["embedding_calls"] += 1


def snapshot() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Thread-safe copy (for result.json)."""
    with _lock:
        return copy.deepcopy(_usage)


def total_tokens_all_models() -> int:
    """Sum of total_tokens + embedding_tokens across all scopes and model keys."""
    with _lock:
        s = 0
        for scope_dict in _usage.values():
            for row in scope_dict.values():
                s += int(row.get("total_tokens", 0))
                s += int(row.get("embedding_tokens", 0))
        return s
