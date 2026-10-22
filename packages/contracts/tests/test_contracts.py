from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from ragforge_contracts.api import CollectionCreate, ConnectorCreate, ConnectorKind, Problem
from ragforge_contracts.sse import (
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    VerificationEvent,
    VerificationStatus,
    decode_sse_data,
    encode_sse,
)


@pytest.mark.parametrize(
    "event",
    [
        TokenEvent(text="hello"),
        VerificationEvent(status=VerificationStatus.partial),
        DoneEvent(conversation_id=UUID(int=1), usage=[], prompt_version="chat@1"),
        ErrorEvent(problem=Problem(title="boom", status=500)),
    ],
)
def test_sse_round_trip(event: object) -> None:
    wire = encode_sse(event).decode()  # type: ignore[arg-type]
    header, data_line, *_ = wire.split("\n")
    assert header == f"event: {event.type}"  # type: ignore[attr-defined]
    assert decode_sse_data(data_line.removeprefix("data: ")) == event


def test_unknown_event_type_is_skipped_for_forward_compat() -> None:
    assert decode_sse_data('{"type": "from_the_future", "x": 1}') is None


def test_malformed_known_event_still_rejected() -> None:
    with pytest.raises(ValidationError):
        decode_sse_data('{"type": "token"}')


def test_responses_ignore_new_server_fields() -> None:
    event = decode_sse_data('{"type": "token", "text": "hi", "added_later": true}')
    assert event == TokenEvent(text="hi")
    assert Problem.model_validate({"title": "x", "status": 400, "new_field": 1}).status == 400


def test_collection_name_validated() -> None:
    with pytest.raises(ValidationError):
        CollectionCreate(name="Has Spaces")


def test_connector_token_never_leaks() -> None:
    c = ConnectorCreate(
        kind=ConnectorKind.notion,
        target_collection_id=uuid4(),
        display_name="mine",
        token="secret_abc",
    )
    assert "secret_abc" not in repr(c)
    assert "secret_abc" not in c.model_dump_json()
    assert c.token.get_secret_value() == "secret_abc"


def test_requests_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CollectionCreate(name="ok", surprise=1)  # type: ignore[call-arg]


def test_non_object_sse_payload_rejected_cleanly() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        decode_sse_data("[1, 2]")


def test_job_event_round_trip() -> None:
    from ragforge_contracts.api import JobEvent, JobStatus
    from ragforge_contracts.sse import decode_job_event, encode_job_event

    ev = JobEvent(job_id=UUID(int=5), status=JobStatus.running, progress={"pages": 3})
    wire = encode_job_event(ev).decode()
    assert wire.startswith("event: job\n")
    assert decode_job_event(wire.split("\n")[1].removeprefix("data: ")) == ev


def test_new_enum_values_from_newer_servers_do_not_crash_clients() -> None:
    from ragforge_contracts.api import Job, JobStatus

    job = Job.model_validate(
        {
            "id": str(UUID(int=1)),
            "kind": "sync",
            "status": "paused_by_admin",
            "attempts": 0,
            "last_error": None,
            "enqueued_at": "2026-10-16T00:00:00Z",
        }
    )
    assert job.status == "paused_by_admin"
    known = Job.model_validate({**job.model_dump(mode="json"), "status": "running"})
    assert known.status is JobStatus.running
    ev = decode_sse_data('{"type": "verification", "status": "new_status"}')
    assert isinstance(ev, VerificationEvent)
    assert ev.status == "new_status"
