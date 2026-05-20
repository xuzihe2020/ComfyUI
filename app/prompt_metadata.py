"""Validation for client-supplied per-prompt metadata (extra_data.metadata)."""

from typing import Optional


MAX_METADATA_KEYS = 16
MAX_METADATA_KEY_LEN = 64
MAX_METADATA_VALUE_LEN = 256

# Server-emitted top-level fields on prompt-scoped WebSocket events.
# Client-supplied metadata may not shadow these — payload-wins-on-conflict
# only protects keys present in each individual frame, so reserve them
# at the submission boundary as defense in depth.
RESERVED_METADATA_KEYS = frozenset({
    "prompt_id", "node", "display_node", "output", "nodes", "node_id",
    "node_type", "executed", "exception_message", "exception_type",
    "traceback", "current_inputs", "current_outputs", "timestamp",
    "sid", "status", "prompt", "value", "max",
})


def validate_client_metadata(raw) -> tuple[Optional[dict], Optional[str]]:
    """Return ``(cleaned_metadata, error_message)``.

    A missing field (``None``) is treated as empty metadata. Anything else
    must be a flat ``dict[str, str]`` within the size caps and free of
    reserved keys.
    """
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return None, "extra_data.metadata must be an object"
    if len(raw) > MAX_METADATA_KEYS:
        return None, f"extra_data.metadata exceeds {MAX_METADATA_KEYS} keys"
    cleaned: dict = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key or len(key) > MAX_METADATA_KEY_LEN:
            return None, f"metadata key must be a non-empty string up to {MAX_METADATA_KEY_LEN} chars"
        if key in RESERVED_METADATA_KEYS:
            return None, f"metadata key '{key}' is reserved"
        if not isinstance(value, str) or len(value) > MAX_METADATA_VALUE_LEN:
            return None, f"metadata value for '{key}' must be a string up to {MAX_METADATA_VALUE_LEN} chars"
        cleaned[key] = value
    return cleaned, None
