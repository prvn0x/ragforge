"""Smoke tests of the real Streamlit page (AppTest) against a fake `Backend`. No network."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from streamlit import config
from streamlit.runtime.scriptrunner_utils.exceptions import StopException
from streamlit.testing.v1 import AppTest
from ui_support import (
    ALL_INTERFACES,
    APP,
    COLLECTION,
    FakeBackend,
    app_with,
    citation,
    cited_answer,
    done,
    texts,
)

from ragforge_client.client import ApiError
from ragforge_contracts.api import ApprovalDecision, Collection, Connector, Document, Problem
from ragforge_contracts.sse import (
    ChatEvent,
    CitationEvent,
    ErrorEvent,
    ExtraEvent,
    InterruptEvent,
    TokenEvent,
    ToolCallEvent,
    VerificationEvent,
)
from ragforge_ui import config as ui_config
from ragforge_ui.api import SdkBackend
from ragforge_ui.config import ALLOW_REMOTE_ENV
from ragforge_ui.render import md_literal


def _ask(at: AppTest, message: str) -> AppTest:
    return at.chat_input[0].set_value(message).run()


def _page(at: AppTest) -> str:
    return "\n".join(texts(at.main))


def _index(items: list[str], needle: str) -> int:
    return next(i for i, text in enumerate(items) if needle in text)


def _full_turn(conversation_id: UUID) -> list[ChatEvent]:
    return [
        TokenEvent(text="**In simple terms:** Consumers poll \\[S1\\]. "),
        TokenEvent(text="**Technically:** Offsets commit \\[S1\\]."),
        CitationEvent(citation=citation(1)),
        CitationEvent(citation=citation(2, inferred=True, title="Offsets")),
        VerificationEvent(status="partial", unsupported_claims=["Offsets never lag"]),
        ExtraEvent(kind="incident", text="Common failure pattern\\: a rebalance storm\\."),
        done(conversation_id),
    ]


def test_a_chat_turn_renders_answer_badge_citations_then_extras_last() -> None:
    conversation_id = uuid4()
    backend = FakeBackend(streams=[_full_turn(conversation_id)])
    at = _ask(app_with(backend), "what is a consumer?")
    assert not at.exception
    assert backend.chats == [("what is a consumer?", [COLLECTION.id], None)]
    items = texts(at.main)
    answer = _index(items, "Consumers poll")
    badge = _index(items, "Partly verified")
    claim = _index(items, "Not found in your notes")  # an old server: no reasons, so red
    sources = _index(items, "Sources from your notes")
    extras = _index(items, "Not from your notes")
    incident = _index(items, "rebalance storm")
    assert answer < badge < claim < sources < extras < incident
    assert "\n\n**Technically\\:**" in items[answer]  # sections are separate paragraphs
    assert "\\[S2\\*\\]" in items[sources]
    assert "matched by the server" in items[sources]
    assert "https://www.notion.so/Kafka-basics-abc#block1" in items[sources]
    assert any("first token" in c.value for c in at.caption)
    # the incident is general knowledge: never inside the answer, never above the verdict
    assert "rebalance" not in items[answer]


def test_follow_up_reuses_the_conversation_and_new_conversation_resets_it() -> None:
    first = uuid4()
    backend = FakeBackend(streams=[cited_answer(first), cited_answer(first), cited_answer(uuid4())])
    at = _ask(app_with(backend), "q1")
    at = _ask(at, "q2")
    assert backend.chats[1][2] == first
    at = at.sidebar.button[1].click().run()  # New conversation
    assert "q1" not in _page(at)
    _ask(at, "q3")
    assert backend.chats[2][2] is None


def test_show_the_code_sends_the_follow_up_then_disappears() -> None:
    conversation_id = uuid4()
    code_turn: list[ChatEvent] = [
        TokenEvent(text="Here \\[S1\\]."),
        CitationEvent(citation=citation(1)),
        ExtraEvent(kind="code", text="```python\nconsumer.poll(1.0)\n```"),
        done(conversation_id),
    ]
    backend = FakeBackend(streams=[cited_answer(conversation_id), code_turn])
    at = _ask(app_with(backend), "how do consumers work?")
    buttons = [b for b in at.button if b.label == "Show the code"]
    assert len(buttons) == 1
    at = buttons[0].click().run()
    assert backend.chats[1] == ("show the code", [COLLECTION.id], conversation_id)
    assert [c.value for c in at.code] == ["consumer.poll(1.0)"]
    assert at.code[0].language == "python"
    assert not [b for b in at.button if b.label == "Show the code"]


def test_no_code_offer_for_an_uncited_answer() -> None:
    backend = FakeBackend(
        streams=[[TokenEvent(text="Your notes don't cover this."), done(uuid4())]]
    )
    at = _ask(app_with(backend), "q")
    assert not [b for b in at.button if b.label == "Show the code"]


def test_401_is_friendly_and_the_conversation_survives() -> None:
    conversation_id = uuid4()
    unauthorized = ApiError(Problem(title="Unauthorized", status=401))
    backend = FakeBackend(streams=[cited_answer(conversation_id), unauthorized])
    at = _ask(app_with(backend), "first question")
    at = _ask(at, "second question")
    page = _page(at)
    assert "first question" in page
    assert "Consumers poll" in page
    assert "The API rejected this key" in at.error[0].value
    assert not at.exception


def test_a_pre_verdict_error_shows_the_draft_as_unverified_plain_text() -> None:
    """Integrated review P1: before `verification` the draft was never checked or saved, so
    it must not look like an answer (no Markdown), and must say so."""
    events: list[ChatEvent] = [
        TokenEvent(text="**In simple terms:** Kafka is **certainly** safe"),
        ErrorEvent(problem=Problem(title="Too Many Requests", status=429)),
    ]
    at = _ask(app_with(FakeBackend(streams=[events])), "q")
    assert "Too many requests right now" in at.error[0].value
    assert any("Unverified draft" in c.value for c in at.caption)
    assert not any("certainly" in m.value for m in at.markdown)  # never rendered as Markdown
    assert any("certainly" in t.value for t in at.text)


def test_api_down_is_a_friendly_message_naming_the_url() -> None:
    backend = FakeBackend(streams=[httpx.ConnectError("refused")])
    at = _ask(app_with(backend), "q")
    assert "Cannot reach the RAGForge API" in at.error[0].value
    assert md_literal("http://api.test") in at.error[0].value


def test_connection_dropped_mid_answer_keeps_what_was_shown() -> None:
    dropped: list[ChatEvent | BaseException] = [
        TokenEvent(text="Kafka keeps"),
        httpx.ReadError("reset"),
    ]
    backend = FakeBackend(streams=[dropped, cited_answer(uuid4())])
    at = _ask(app_with(backend), "q")
    assert "Kafka keeps" in _page(at)
    assert "The connection dropped" in at.error[0].value
    at = _ask(at, "again")  # the page still works, and the earlier turn is still there
    assert "Kafka keeps" in _page(at)
    assert "Consumers poll" in _page(at)


def test_api_down_before_anything_loads() -> None:
    backend = FakeBackend(fail_collections=httpx.ConnectError("refused"))
    at = app_with(backend)
    assert "Cannot reach the RAGForge API" in at.error[0].value
    assert not at.exception


def test_interrupt_approve_resumes_with_a_decision_for_every_call() -> None:
    run_id, conversation_id = uuid4(), uuid4()
    pending = [
        ToolCallEvent(tool_call_id="t1", tool="remove_from_index", args={"document_id": "d1"}),
    ]
    interrupted: list[ChatEvent] = [
        InterruptEvent(run_id=run_id, conversation_id=conversation_id, pending=pending)
    ]
    after: list[ChatEvent] = [TokenEvent(text="Removed it \\[T1\\]."), done(conversation_id)]
    backend = FakeBackend(streams=[interrupted, after])
    at = _ask(app_with(backend), "/remove Kafka basics")
    assert md_literal("remove_from_index") in _page(at)
    assert '"document_id": "d1"' in at.code[0].value
    approve = next(b for b in at.button if b.label == "Approve")
    at = approve.click().run()
    assert backend.resumes == [(run_id, {"t1": ApprovalDecision.approve})]
    page = _page(at)
    assert "Approved: remove_from_index" in page
    assert "Removed it" in page
    assert not [b for b in at.button if b.label in ("Approve", "Reject")]


def test_interrupt_reject() -> None:
    run_id = uuid4()
    pending = [ToolCallEvent(tool_call_id="t1", tool="forget_fact", args={})]
    backend = FakeBackend(
        streams=[[InterruptEvent(run_id=run_id, pending=pending)], [done(uuid4())]]
    )
    at = _ask(app_with(backend), "/forget f1")
    next(b for b in at.button if b.label == "Reject").click().run()
    assert backend.resumes == [(run_id, {"t1": ApprovalDecision.reject})]


def test_a_new_question_expires_the_pending_approval() -> None:
    pending = [ToolCallEvent(tool_call_id="t1", tool="forget_fact", args={})]
    backend = FakeBackend(
        streams=[[InterruptEvent(run_id=uuid4(), pending=pending)], cited_answer(uuid4())]
    )
    at = _ask(app_with(backend), "/forget f1")
    at = _ask(at, "never mind, what is kafka?")
    assert not [b for b in at.button if b.label in ("Approve", "Reject")]
    assert "expired" in _page(at)
    assert backend.resumes == []


def test_sources_page_lists_connectors_and_syncs() -> None:
    connector = Connector(
        id=uuid4(),
        kind="notion",
        target_collection_id=COLLECTION.id,
        display_name="My Notion",
        status="error",
        last_error="token expired",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    backend = FakeBackend(connectors_list=[connector])
    at = app_with(backend)
    at = at.sidebar.radio[0].set_value("Sources").run()
    page = _page(at)
    assert "My Notion" in page
    assert "Into notes" in page
    assert "token expired" in at.error[0].value
    assert len(at.dataframe) == 1
    at = next(b for b in at.button if b.label == "Sync now").click().run()
    assert backend.synced == [connector.id]
    assert "Sync queued" in at.success[0].value
    assert "running" in _page(at)
    assert "pages: 3" in _page(at)


def test_the_key_is_never_rendered_even_on_errors() -> None:
    key = "rf_live_do_not_show"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"title": "Unauthorized", "status": 401, "detail": "bad"})

    backend = SdkBackend("http://api.test", SecretStr(key), transport=httpx.MockTransport(handler))
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["rf_backend"] = backend
    at.run()
    everything = "\n".join(texts(at.main) + texts(at.sidebar))
    assert "The API rejected this key" in everything
    assert key not in everything


def test_a_shared_key_file_is_refused_without_showing_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "ui.key"
    key_file.write_text("rf_live_do_not_show")
    key_file.chmod(0o644)
    monkeypatch.setenv("RAGFORGE_KEY_FILE", str(key_file))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert "chmod 600" in at.error[0].value
    assert "rf_live_do_not_show" not in "\n".join(texts(at.main) + texts(at.sidebar))


# ------------------------------------------------------------------ no input while streaming
def _verified_then_stopped() -> list[ChatEvent | BaseException]:
    """What a rerun mid-stream looks like: verification arrived, then Streamlit stopped the
    script before `done` (extras still pending)."""
    return [
        TokenEvent(text="Offsets commit \\[S1\\]."),
        CitationEvent(citation=citation(1)),
        VerificationEvent(status="verified", unsupported_claims=[]),
        StopException(),
    ]


def test_every_input_is_disabled_while_a_turn_streams() -> None:
    backend = FakeBackend(streams=[_verified_then_stopped()])
    at = _ask(app_with(backend), "what is an offset?")  # this run stops mid-stream
    assert not at.exception
    assert at.chat_input[0].disabled  # no next message before `done` (SSE contract)
    sidebar = {b.label: b.disabled for b in at.sidebar.button}
    assert sidebar == {"Refresh sources": True, "New conversation": True}
    assert at.sidebar.radio[0].disabled
    assert at.sidebar.multiselect[0].disabled


def test_a_turn_cut_off_after_verification_reads_as_complete_and_input_comes_back() -> None:
    backend = FakeBackend(streams=[_verified_then_stopped()])
    at = _ask(app_with(backend), "what is an offset?").run()
    page = _page(at)
    assert "the answer and its check above are complete" in page
    assert "Send it again" not in page
    assert "Verified" in page
    assert not at.chat_input[0].disabled  # the busy flag is cleared even on a stop


def test_a_turn_cut_off_before_verification_asks_to_send_again() -> None:
    backend = FakeBackend(streams=[[TokenEvent(text="Offsets"), StopException()]])
    at = _ask(app_with(backend), "q").run()
    assert "interrupted before it finished. Send it again." in _page(at)


def test_show_the_code_click_disables_inputs_from_that_very_run() -> None:
    conversation_id = uuid4()
    backend = FakeBackend(
        streams=[cited_answer(conversation_id), [TokenEvent(text="x"), StopException()]]
    )
    at = _ask(app_with(backend), "how do consumers work?")
    at = next(b for b in at.button if b.label == "Show the code").click().run()
    assert backend.chats[1] == ("show the code", [COLLECTION.id], conversation_id)
    assert at.chat_input[0].disabled
    assert all(b.disabled for b in at.button if b.label == "Show the code")


# ------------------------------------------------------------------ approvals
def _two_calls(run_id: UUID) -> list[ChatEvent]:
    pending = [
        ToolCallEvent(tool_call_id="t1", tool="remove_from_index", args={"document_id": "d1"}),
        ToolCallEvent(tool_call_id="t2", tool="forget_fact", args={"fact_id": "f9"}),
    ]
    return [InterruptEvent(run_id=run_id, conversation_id=uuid4(), pending=pending)]


def test_several_pending_calls_are_decided_one_by_one() -> None:
    run_id = uuid4()
    backend = FakeBackend(streams=[_two_calls(run_id), [done(uuid4())]])
    at = _ask(app_with(backend), "/remove Kafka basics")
    assert [c.value for c in at.checkbox] == [False, False]  # nothing approved by default
    at = at.checkbox[0].check().run()
    at = next(b for b in at.button if b.label == "Approve checked, reject the rest").click().run()
    assert backend.resumes == [
        (run_id, {"t1": ApprovalDecision.approve, "t2": ApprovalDecision.reject})
    ]


def test_reject_all_rejects_every_pending_call() -> None:
    run_id = uuid4()
    backend = FakeBackend(streams=[_two_calls(run_id), [done(uuid4())]])
    at = _ask(app_with(backend), "/remove Kafka basics")
    at = at.checkbox[0].check().run()  # a checked box does not sneak into "Reject all"
    next(b for b in at.button if b.label == "Reject all").click().run()
    assert backend.resumes == [
        (run_id, {"t1": ApprovalDecision.reject, "t2": ApprovalDecision.reject})
    ]


def test_a_failed_resume_keeps_approve_reject_for_a_retry() -> None:
    run_id = uuid4()
    pending = [ToolCallEvent(tool_call_id="t1", tool="forget_fact", args={})]
    backend = FakeBackend(
        streams=[
            [InterruptEvent(run_id=run_id, pending=pending)],
            httpx.ConnectError("refused"),  # POST /v1/approvals never reached the server
            [TokenEvent(text="Forgotten."), done(uuid4())],
        ]
    )
    at = _ask(app_with(backend), "/forget f1")
    at = next(b for b in at.button if b.label == "Approve").click().run()
    assert "Cannot reach the RAGForge API" in at.error[0].value
    at = next(b for b in at.button if b.label == "Approve").click().run()  # still offered
    assert [r[0] for r in backend.resumes] == [run_id, run_id]
    assert "Forgotten." in _page(at)
    assert not [b for b in at.button if b.label in ("Approve", "Reject")]


# ------------------------------------------------------------------ collections
def test_collections_with_the_same_name_are_both_selectable() -> None:
    twin = Collection(
        id=uuid4(), name=COLLECTION.name, active_space_id=2, created_at=COLLECTION.created_at
    )
    backend = FakeBackend(collections_list=[COLLECTION, twin], streams=[cited_answer(uuid4())])
    at = app_with(backend)
    assert len(at.sidebar.multiselect[0].options) == 2
    _ask(at, "q")
    assert backend.chats[0][1] == [COLLECTION.id, twin.id]


def test_a_failed_refresh_keeps_the_conversation_and_the_last_collections() -> None:
    backend = FakeBackend(streams=[cited_answer(uuid4()), cited_answer(uuid4())])
    at = _ask(app_with(backend), "first question")
    backend.fail_collections = httpx.ConnectError("refused")
    at = next(b for b in at.sidebar.button if b.label == "Refresh sources").click().run()
    assert "Cannot reach the RAGForge API" in at.error[0].value
    page = _page(at)
    assert "first question" in page
    assert "Consumers poll" in page
    assert not at.chat_input[0].disabled  # the last good list is still in use
    _ask(at, "second")
    assert backend.chats[1][1] == [COLLECTION.id]


# ------------------------------------------------------------------ config failures
def test_an_unreadable_key_file_is_a_friendly_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "ui.key"
    key_file.write_bytes(b"rf_live_\xffsecret")
    key_file.chmod(0o600)
    monkeypatch.setenv("RAGFORGE_KEY_FILE", str(key_file))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "Cannot read the API key file" in at.error[0].value
    assert "UnicodeDecodeError" in at.error[0].value
    assert "secret" not in "\n".join(texts(at.main) + texts(at.sidebar))


def test_a_key_file_path_that_is_a_directory_is_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "ui.key"
    folder.mkdir(mode=0o700)
    monkeypatch.setenv("RAGFORGE_KEY_FILE", str(folder))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "Cannot read the API key file" in at.error[0].value


def test_a_bad_flag_is_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["app.py", "--url"])  # --url without its value
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "Bad command-line flags" in at.error[0].value


def test_the_key_file_message_is_escaped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "[x](https:evil)" / "ui.key"
    key_file.parent.mkdir()
    key_file.write_text("rf_live_do_not_show")
    key_file.chmod(0o644)
    monkeypatch.setenv("RAGFORGE_KEY_FILE", str(key_file))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert "\\[x\\]\\(https\\:evil\\)" in at.error[0].value  # a path can't become a link


# ------------------------------------------------------------------ network exposure
@pytest.mark.parametrize("address", [None, ALL_INTERFACES, "192.168.1.20"])
def test_a_page_reachable_from_the_network_never_loads_the_key(
    address: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "ui.key"
    key_file.write_text("rf_live_do_not_show")
    key_file.chmod(0o600)
    monkeypatch.setenv("RAGFORGE_KEY_FILE", str(key_file))
    reads: list[Path] = []

    def read_key(path: Path) -> str:
        reads.append(path)
        return "k"

    monkeypatch.setattr(ui_config, "read_key", read_key)
    config.set_option("server.address", address)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "reachable from other machines" in at.error[0].value
    assert "127.0.0.1" in at.error[0].value  # says how to fix it
    assert reads == []  # the key file was never even read
    assert not at.chat_input


def test_an_injected_backend_is_refused_too_when_exposed() -> None:
    config.set_option("server.address", ALL_INTERFACES)
    backend = FakeBackend()
    at = app_with(backend)
    assert "reachable from other machines" in at.error[0].value
    assert not at.chat_input


def test_the_explicit_remote_opt_in_allows_it(monkeypatch: pytest.MonkeyPatch) -> None:
    config.set_option("server.address", ALL_INTERFACES)
    monkeypatch.setenv(ALLOW_REMOTE_ENV, "1")
    at = app_with(FakeBackend())
    assert not at.error
    assert at.chat_input


# ------------------------------------------------------------------ approvals survive / links
def test_a_question_that_never_reached_the_server_keeps_approve_reject() -> None:
    run_id = uuid4()
    pending = [ToolCallEvent(tool_call_id="t1", tool="forget_fact", args={})]
    backend = FakeBackend(
        streams=[
            [InterruptEvent(run_id=run_id, pending=pending)],
            httpx.ConnectError("refused"),  # the new question never got to the server
            [done(uuid4())],
        ]
    )
    at = _ask(app_with(backend), "/forget f1")
    at = _ask(at, "what is kafka?")
    assert "Cannot reach the RAGForge API" in at.error[0].value
    assert "expired" not in _page(at)
    at = next(b for b in at.button if b.label == "Approve").click().run()  # still offered
    assert backend.resumes == [(run_id, {"t1": ApprovalDecision.approve})]


def test_an_approval_link_this_page_cannot_load_is_explained_not_ignored() -> None:
    """MCP sends the owner to `<ui>/?run_id=…`; a reload lands the same way. The API can't list
    pending calls yet, so the page must say so instead of showing nothing."""
    run_id = uuid4()
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["rf_backend"] = FakeBackend()
    at.query_params["run_id"] = str(run_id)
    at.run()
    assert "waiting for approval" in at.warning[0].value
    assert str(run_id)[:8] in at.warning[0].value
    assert not [b for b in at.button if b.label in ("Approve", "Reject")]  # nothing to decide
    at = next(b for b in at.button if b.label == "Dismiss").click().run()
    assert not at.warning
    assert "run_id" not in at.query_params


def test_a_malformed_run_id_is_ignored_and_cleared() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["rf_backend"] = FakeBackend()
    at.query_params["run_id"] = "<script>"
    at.run()
    assert not at.warning
    assert "run_id" not in at.query_params


def test_the_open_approval_is_in_the_url_until_it_is_decided() -> None:
    run_id = uuid4()
    pending = [ToolCallEvent(tool_call_id="t1", tool="forget_fact", args={})]
    backend = FakeBackend(
        streams=[[InterruptEvent(run_id=run_id, pending=pending)], [done(uuid4())]]
    )
    at = _ask(app_with(backend), "/forget f1")
    assert at.query_params["run_id"] == str(run_id)  # a reload lands on the notice
    assert not at.warning  # this session holds it: shown as Approve/Reject, not as a notice
    at = next(b for b in at.button if b.label == "Approve").click().run()
    assert "run_id" not in at.query_params


# ------------------------------------------------------------------ approval card names pages
def _document(title: str) -> Document:
    return Document(
        id=uuid4(),
        collection_id=COLLECTION.id,
        title=title,
        external_url=None,
        external_updated_at=None,
        fetched_at=None,
    )


def _remove(document_id: UUID) -> list[ChatEvent]:
    pending = [
        ToolCallEvent(
            tool_call_id="t1", tool="remove_from_index", args={"document_id": str(document_id)}
        )
    ]
    return [InterruptEvent(run_id=uuid4(), pending=pending)]


def test_the_approval_card_names_the_page_a_removal_touches() -> None:
    doc = _document("Kafka basics")
    backend = FakeBackend(streams=[_remove(doc.id)], documents_list=[_document("Other"), doc])
    at = _ask(app_with(backend), "/remove Kafka basics")
    assert "Remove “Kafka basics” from the index" in _page(at)
    assert str(doc.id) in at.code[0].value  # the raw call stays visible too
    at.run()
    assert backend.documents_calls == 1  # looked up once, not on every rerun


def test_a_page_already_cited_is_named_without_an_api_call() -> None:
    conversation_id = uuid4()
    cited = citation(1, title="Kafka basics")
    first: list[ChatEvent] = [
        TokenEvent(text="Consumers poll \\[S1\\]."),
        CitationEvent(citation=cited),
        done(conversation_id),
    ]
    backend = FakeBackend(streams=[first, _remove(cited.document_id)])
    at = _ask(app_with(backend), "q")
    at = _ask(at, "/remove that page")
    assert "Remove “Kafka basics” from the index" in _page(at)
    assert backend.documents_calls == 0


def test_an_unknown_document_is_flagged_and_a_failed_lookup_falls_back() -> None:
    backend = FakeBackend(streams=[_remove(uuid4())], fail_documents=httpx.ConnectError("refused"))
    at = _ask(app_with(backend), "/remove x")
    assert "unknown document" in _page(at)
    assert next(b for b in at.button if b.label == "Approve")  # still decidable
    assert not at.exception


# ------------------------------------------------------------------ sync now
def _connector() -> Connector:
    return Connector(
        id=uuid4(),
        kind="notion",
        target_collection_id=COLLECTION.id,
        display_name="My Notion",
        status="active",
        last_error=None,
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


def _sources(backend: FakeBackend) -> AppTest:
    return app_with(backend).sidebar.radio[0].set_value("Sources").run()


def test_sync_now_is_disabled_while_its_job_runs_and_cannot_double_queue() -> None:
    connector = _connector()
    backend = FakeBackend(connectors_list=[connector])
    at = _sources(backend)
    at = at.button(key=f"sync-{connector.id}").click().run()
    button = at.button(key=f"sync-{connector.id}")
    assert button.disabled
    assert button.label == "Syncing…"
    assert backend.synced == [connector.id]
    job_id = next(iter(backend.jobs))
    backend.jobs[job_id] = backend.jobs[job_id].model_copy(update={"status": "done"})
    at = next(b for b in at.button if b.label == "Refresh status").click().run()
    assert not at.button(key=f"sync-{connector.id}").disabled  # finished: offered again


def test_the_second_click_of_a_double_click_queues_nothing() -> None:
    """Both clicks of a double click can reach the server before the page redraws the button
    disabled. The callback guard is what stops the second: simulate the state the first click's
    callback leaves (connector marked syncing) with the button still drawn enabled."""
    connector = _connector()
    backend = FakeBackend(connectors_list=[connector])
    at = _sources(backend)
    at.session_state["rf_syncing"] = {connector.id}
    at.button(key=f"sync-{connector.id}").click().run()
    assert backend.synced == []


def test_a_job_the_api_no_longer_knows_is_dropped_not_errored_forever() -> None:
    connector = _connector()
    backend = FakeBackend(connectors_list=[connector])
    at = _sources(backend)
    at = at.button(key=f"sync-{connector.id}").click().run()
    job_id = next(iter(backend.jobs))
    backend.job_errors[job_id] = ApiError(Problem(title="Not Found", status=404))
    at = next(b for b in at.button if b.label == "Refresh status").click().run()
    assert not at.error
    assert "Recent syncs" not in _page(at)
    assert at.session_state["rf_jobs"] == []
    assert not at.button(key=f"sync-{connector.id}").disabled


def test_a_failed_sync_is_friendly_and_leaves_the_button_usable() -> None:
    connector = _connector()

    class Failing(FakeBackend):
        def sync_connector(self, connector_id: UUID) -> UUID:
            raise ApiError(Problem(title="Too Many Requests", status=429))

    at = _sources(Failing(connectors_list=[connector]))
    at = at.button(key=f"sync-{connector.id}").click().run()
    assert "Too many requests right now" in at.error[0].value
    assert not at.button(key=f"sync-{connector.id}").disabled


def test_a_failure_after_the_verdict_says_the_answer_is_saved() -> None:
    """Re-review P2: after `verification` the answer is committed; the page must not tell the
    owner it is incomplete or to send it again (that would duplicate the turn)."""
    events: list[ChatEvent | BaseException] = [
        TokenEvent(text="Kafka keeps logs \\[S1\\]."),
        CitationEvent(citation=citation()),
        VerificationEvent(status="verified", unsupported_claims=[]),
        httpx.ReadError("reset"),
    ]
    at = _ask(app_with(FakeBackend(streams=[events])), "q")
    assert "stopped after the answer was checked" in at.error[0].value
    assert "No need to send it again" in at.error[0].value


@pytest.mark.parametrize("check", ["host_refused", "origin_refused"])
def test_the_page_really_calls_the_host_and_origin_checks(
    check: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-review P2: unit tests of the checks prove nothing if `_backend` stops calling them."""
    monkeypatch.setattr(ui_config, check, lambda *args: True)
    backend = FakeBackend(streams=[])
    at = app_with(backend)
    assert any("opened by another web page" in e.value for e in at.error)
    assert not backend.chats
