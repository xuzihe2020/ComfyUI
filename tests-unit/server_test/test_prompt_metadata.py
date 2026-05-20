"""Tests for the opaque per-prompt metadata mechanism on PromptServer."""

from unittest.mock import MagicMock

import pytest

from app.prompt_metadata import (
    MAX_METADATA_KEY_LEN,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LEN,
    RESERVED_METADATA_KEYS,
    validate_client_metadata,
)
from comfy_execution.jobs import extract_workflow_id


class TestExtractWorkflowId:

    def test_returns_id_when_present(self):
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {"id": "wf-1"}}}) == "wf-1"

    def test_returns_none_when_missing(self):
        assert extract_workflow_id({}) is None
        assert extract_workflow_id({"extra_pnginfo": {}}) is None
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {}}}) is None

    def test_returns_none_for_empty_or_wrong_type(self):
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {"id": ""}}}) is None
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {"id": 42}}}) is None
        assert extract_workflow_id({"extra_pnginfo": {"workflow": {"id": None}}}) is None

    def test_returns_none_for_non_dict_input(self):
        assert extract_workflow_id(None) is None
        assert extract_workflow_id("not a dict") is None
        assert extract_workflow_id({"extra_pnginfo": "not a dict"}) is None
        assert extract_workflow_id({"extra_pnginfo": {"workflow": "not a dict"}}) is None


class _FakeServer:
    """Minimal PromptServer stand-in mirroring send_sync verbatim.

    ``active_prompt_metadata`` is ``Optional[tuple[str, dict]]`` — the
    ``prompt_id`` it belongs to plus the opaque dict. send_sync only merges
    when the outgoing payload's ``prompt_id`` matches the active one, so
    unrelated queue/status broadcasts are not contaminated.
    """

    def __init__(self):
        self.active_prompt_metadata = None
        self.captured = []
        self.loop = MagicMock()
        self.loop.call_soon_threadsafe.side_effect = (
            lambda fn, msg: self.captured.append(msg)
        )
        self.messages = MagicMock()
        self.messages.put_nowait = MagicMock()

    def send_sync(self, event, data, sid=None):
        slot = self.active_prompt_metadata
        if slot is not None and isinstance(data, dict):
            active_prompt_id, meta = slot
            if meta and data.get("prompt_id") == active_prompt_id:
                data = {**meta, **data}
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid)
        )


@pytest.fixture
def server():
    return _FakeServer()


class TestSendSyncMerge:
    def test_spreads_active_metadata_onto_dict_payload(self, server):
        server.active_prompt_metadata = ("p1", {"workflow_id": "wf-1"})

        server.send_sync(
            "executing", {"node": "n1", "prompt_id": "p1"}, "client-1"
        )

        event, data, sid = server.captured[0]
        assert event == "executing"
        assert data == {
            "workflow_id": "wf-1",
            "node": "n1",
            "prompt_id": "p1",
        }
        assert sid == "client-1"

    def test_passthrough_when_no_active_metadata(self, server):
        server.active_prompt_metadata = None

        server.send_sync("executing", {"node": "n1", "prompt_id": "p1"})

        _, data, _ = server.captured[0]
        assert data == {"node": "n1", "prompt_id": "p1"}

    def test_passthrough_when_metadata_is_empty_dict(self, server):
        server.active_prompt_metadata = ("p1", {})

        server.send_sync("executing", {"node": "n1", "prompt_id": "p1"})

        _, data, _ = server.captured[0]
        assert data == {"node": "n1", "prompt_id": "p1"}

    def test_event_payload_wins_on_key_conflict(self, server):
        server.active_prompt_metadata = (
            "p1",
            {"workflow_id": "wf-1", "prompt_id": "from-meta"},
        )

        server.send_sync("executing", {"node": "n1", "prompt_id": "p1"}, "c1")

        _, data, _ = server.captured[0]
        assert data["prompt_id"] == "p1"
        assert data["workflow_id"] == "wf-1"

    def test_non_dict_payload_passes_through_untouched(self, server):
        server.active_prompt_metadata = ("p1", {"workflow_id": "wf-1"})

        server.send_sync("text", b"\x00\x00\x00\x03foobar", "c1")

        _, data, _ = server.captured[0]
        assert data == b"\x00\x00\x00\x03foobar"

    def test_terminal_executing_frame_includes_metadata(self, server):
        server.active_prompt_metadata = ("p1", {"workflow_id": "wf-1"})

        server.send_sync(
            "executing", {"node": None, "prompt_id": "p1"}, "client-1"
        )

        _, data, _ = server.captured[0]
        assert data == {
            "workflow_id": "wf-1",
            "node": None,
            "prompt_id": "p1",
        }

    def test_opaque_dict_supports_arbitrary_keys(self, server):
        server.active_prompt_metadata = (
            "p1",
            {"workflow_id": "wf-1", "trace_id": "trace-123", "tenant": "acme"},
        )

        server.send_sync("executing", {"node": "n1", "prompt_id": "p1"})

        _, data, _ = server.captured[0]
        assert data["workflow_id"] == "wf-1"
        assert data["trace_id"] == "trace-123"
        assert data["tenant"] == "acme"


class TestStatusBroadcastsAreNotContaminated:
    """Regression tests for the contamination bug:

    ``send_sync`` previously spread metadata onto any dict payload, so a
    status broadcast fired while a prompt was running picked up that
    prompt's metadata even though it had nothing to do with that prompt.
    """

    def test_status_payload_without_prompt_id_is_untouched(self, server):
        server.active_prompt_metadata = ("p-running", {"workflow_id": "wf-1"})

        server.send_sync("status", {"status": {"exec_info": {"queue_remaining": 1}}})

        _, data, _ = server.captured[0]
        assert data == {"status": {"exec_info": {"queue_remaining": 1}}}
        assert "workflow_id" not in data

    def test_payload_for_different_prompt_is_untouched(self, server):
        # Active prompt is p-running; we send a frame for p-other (e.g. another
        # client's queued item). The merge must not leak across prompts.
        server.active_prompt_metadata = ("p-running", {"workflow_id": "wf-1"})

        server.send_sync("executing", {"node": "n1", "prompt_id": "p-other"})

        _, data, _ = server.captured[0]
        assert data == {"node": "n1", "prompt_id": "p-other"}
        assert "workflow_id" not in data

    def test_queue_updated_frame_during_active_prompt_is_clean(self, server):
        server.active_prompt_metadata = ("p-running", {"workflow_id": "wf-1"})

        server.send_sync("status", {"status": {"exec_info": {"queue_remaining": 0}}})

        _, data, _ = server.captured[0]
        assert "workflow_id" not in data


class TestWorkerSerializationIsolatesMetadata:
    def test_two_prompts_sharing_prompt_id_get_correct_metadata(self, server):
        # Prompt A
        server.active_prompt_metadata = ("P-shared", {"workflow_id": "wf-AAA"})
        server.send_sync("execution_start", {"prompt_id": "P-shared"})
        server.send_sync("executing", {"node": "n1", "prompt_id": "P-shared"})
        server.send_sync("executing", {"node": None, "prompt_id": "P-shared"})
        server.active_prompt_metadata = None

        # Prompt B — same prompt_id, different workflow
        server.active_prompt_metadata = ("P-shared", {"workflow_id": "wf-BBB"})
        server.send_sync("execution_start", {"prompt_id": "P-shared"})
        server.send_sync("executing", {"node": "n2", "prompt_id": "P-shared"})
        server.send_sync("executing", {"node": None, "prompt_id": "P-shared"})
        server.active_prompt_metadata = None

        frames = [d for (_, d, _) in server.captured]
        a_frames = frames[:3]
        b_frames = frames[3:]

        assert all(f["workflow_id"] == "wf-AAA" for f in a_frames)
        assert all(f["workflow_id"] == "wf-BBB" for f in b_frames)
        assert all(f["prompt_id"] == "P-shared" for f in frames)


class TestValidateClientMetadata:
    def test_none_returns_empty_dict(self):
        cleaned, error = validate_client_metadata(None)
        assert cleaned == {}
        assert error is None

    def test_flat_string_dict_is_accepted(self):
        cleaned, error = validate_client_metadata(
            {"workflow_id": "wf-1", "trace_id": "trace-abc"}
        )
        assert cleaned == {"workflow_id": "wf-1", "trace_id": "trace-abc"}
        assert error is None

    def test_non_dict_is_rejected(self):
        _, error = validate_client_metadata("not a dict")
        assert error is not None
        assert "object" in error

    def test_list_is_rejected(self):
        _, error = validate_client_metadata([("workflow_id", "wf-1")])
        assert error is not None

    def test_nested_dict_value_is_rejected(self):
        _, error = validate_client_metadata({"workflow": {"id": "wf-1"}})
        assert error is not None
        assert "string" in error

    def test_non_string_value_is_rejected(self):
        _, error = validate_client_metadata({"workflow_id": 42})
        assert error is not None

    def test_non_string_key_is_rejected(self):
        _, error = validate_client_metadata({123: "wf-1"})
        assert error is not None

    def test_empty_key_is_rejected(self):
        _, error = validate_client_metadata({"": "wf-1"})
        assert error is not None

    def test_key_exceeding_limit_is_rejected(self):
        _, error = validate_client_metadata({"k" * (MAX_METADATA_KEY_LEN + 1): "v"})
        assert error is not None
        assert str(MAX_METADATA_KEY_LEN) in error

    def test_value_exceeding_limit_is_rejected(self):
        _, error = validate_client_metadata({"workflow_id": "v" * (MAX_METADATA_VALUE_LEN + 1)})
        assert error is not None
        assert str(MAX_METADATA_VALUE_LEN) in error

    def test_too_many_keys_is_rejected(self):
        raw = {f"k{i}": "v" for i in range(MAX_METADATA_KEYS + 1)}
        _, error = validate_client_metadata(raw)
        assert error is not None
        assert str(MAX_METADATA_KEYS) in error

    def test_max_size_dict_is_accepted(self):
        raw = {f"k{i}": "v" for i in range(MAX_METADATA_KEYS)}
        cleaned, error = validate_client_metadata(raw)
        assert error is None
        assert len(cleaned) == MAX_METADATA_KEYS

    def test_max_length_strings_are_accepted(self):
        raw = {"k" * MAX_METADATA_KEY_LEN: "v" * MAX_METADATA_VALUE_LEN}
        cleaned, error = validate_client_metadata(raw)
        assert error is None
        assert cleaned == raw

    @pytest.mark.parametrize("reserved_key", sorted(RESERVED_METADATA_KEYS))
    def test_reserved_keys_are_rejected(self, reserved_key):
        _, error = validate_client_metadata({reserved_key: "anything"})
        assert error is not None
        assert reserved_key in error
