"""
Gateway tests for the WhatsApp webhook (`src/api/webhooks.py`). No live infra:
Redis is an AsyncMock, and the deep-path enqueue (`taskiq_process_message.kiq`) is patched,
so nothing touches a real broker, Redis, or Meta.

Covered:
- GET verification handshake (token match / mismatch).
- HMAC signature gate (bad signature -> 401, never enqueues).
- Malformed JSON body -> 400.
- Valid text/audio message -> 202 and exactly one enqueue with the raw message dict.
- Ingestion idempotency (SETNX says duplicate) -> 202 but no enqueue.
- Status-update payloads (delivery/read receipts) are ignored.
"""
import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import pytest

# webhooks.py imports `redis.asyncio` at module load. The tests mock the Redis *server*
# (no live infra needed), but the redis-py *library* must be importable. It ships with the
# project deps (Docker api/worker images, full `pip install -r requirements.txt`); skip
# cleanly in a partial interpreter that lacks it rather than erroring collection.
pytest.importorskip("redis")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings
from src.api import webhooks
from src.api.dependencies import get_redis

# The prefix webhooks.router is mounted under in src/api/router.py. Requests below go to this
# path so the tests exercise the same URL Meta actually calls (minus the /api/v1 app prefix).
WEBHOOK_PREFIX = "/webhooks"


def _sign(body: bytes) -> str:
    """Produce the `sha256=<hex>` header value the endpoint expects, using the same
    secret the endpoint reads — so the assertion is about the logic, not the secret value."""
    digest = hmac.new(settings.META_APP_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _text_payload(thread_id="15550001111", msg_id="wamid.TEST1", body="hello"):
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": thread_id,
                        "id": msg_id,
                        "type": "text",
                        "text": {"body": body},
                    }]
                }
            }]
        }]
    }


@pytest.fixture
def redis_mock():
    r = AsyncMock()
    # SETNX returns truthy for a brand-new key (not a duplicate).
    r.set = AsyncMock(return_value=True)
    return r


@pytest.fixture
def kiq_mock():
    return AsyncMock()


@pytest.fixture
def client(redis_mock, kiq_mock, monkeypatch):
    # Patch the deep-path enqueue so no broker/Redis is touched.
    monkeypatch.setattr(webhooks.taskiq_process_message, "kiq", kiq_mock)

    app = FastAPI()
    # Mount under the SAME prefix production uses (src/api/router.py), not bare. webhooks.router
    # declares both "" and "/" paths so Meta can hit /webhooks or /webhooks/; newer FastAPI
    # refuses to include a router carrying an empty-path route unless a prefix is supplied.
    app.include_router(webhooks.router, prefix=WEBHOOK_PREFIX)
    app.dependency_overrides[get_redis] = lambda: redis_mock
    return TestClient(app)


# ─── GET verification handshake ────────────────────────────────────────────

def test_verify_webhook_success(client):
    resp = client.get(WEBHOOK_PREFIX, params={
        "hub.mode": "subscribe",
        "hub.verify_token": settings.WHATSAPP_VERIFY_TOKEN,
        "hub.challenge": "challenge-123",
    })
    assert resp.status_code == 200
    assert resp.text == "challenge-123"


def test_verify_webhook_bad_token(client):
    resp = client.get(WEBHOOK_PREFIX, params={
        "hub.mode": "subscribe",
        "hub.verify_token": str(settings.WHATSAPP_VERIFY_TOKEN) + "_WRONG",
        "hub.challenge": "challenge-123",
    })
    assert resp.status_code == 403


# ─── POST signature / body gates ───────────────────────────────────────────

def test_invalid_signature_rejected(client, kiq_mock):
    body = json.dumps(_text_payload()).encode("utf-8")
    resp = client.post(WEBHOOK_PREFIX, content=body, headers={"X-Hub-Signature-256": "sha256=deadbeef"})
    assert resp.status_code == 401
    kiq_mock.assert_not_awaited()


def test_missing_signature_rejected(client, kiq_mock):
    body = json.dumps(_text_payload()).encode("utf-8")
    resp = client.post(WEBHOOK_PREFIX, content=body)  # no signature header at all
    assert resp.status_code == 401
    kiq_mock.assert_not_awaited()


def test_malformed_json_rejected(client, kiq_mock):
    body = b"not-json{"
    resp = client.post(WEBHOOK_PREFIX, content=body, headers={"X-Hub-Signature-256": _sign(body)})
    assert resp.status_code == 400
    kiq_mock.assert_not_awaited()


# ─── Happy path + routing ──────────────────────────────────────────────────

def test_valid_text_message_enqueued(client, kiq_mock):
    body = json.dumps(_text_payload(thread_id="15551234567", msg_id="wamid.ABC")).encode("utf-8")
    resp = client.post(WEBHOOK_PREFIX, content=body, headers={"X-Hub-Signature-256": _sign(body)})
    assert resp.status_code == 202
    kiq_mock.assert_awaited_once()
    enqueued = kiq_mock.await_args.args[0]
    assert enqueued["from"] == "15551234567"
    assert enqueued["id"] == "wamid.ABC"
    assert enqueued["type"] == "text"


def test_audio_message_enqueued(client, kiq_mock):
    payload = {
        "entry": [{"changes": [{"value": {"messages": [{
            "from": "15550009999", "id": "wamid.AUDIO", "type": "audio",
            "audio": {"id": "media-abc"},
        }]}}]}]
    }
    body = json.dumps(payload).encode("utf-8")
    resp = client.post(WEBHOOK_PREFIX, content=body, headers={"X-Hub-Signature-256": _sign(body)})
    assert resp.status_code == 202
    kiq_mock.assert_awaited_once()
    assert kiq_mock.await_args.args[0]["type"] == "audio"


def test_duplicate_message_blocked(client, kiq_mock, redis_mock):
    # SETNX returns falsy -> Meta retry storm / duplicate; must NOT re-enqueue.
    redis_mock.set = AsyncMock(return_value=False)
    body = json.dumps(_text_payload()).encode("utf-8")
    resp = client.post(WEBHOOK_PREFIX, content=body, headers={"X-Hub-Signature-256": _sign(body)})
    assert resp.status_code == 202
    kiq_mock.assert_not_awaited()


def test_status_update_ignored(client, kiq_mock):
    payload = {"entry": [{"changes": [{"value": {"statuses": [
        {"id": "wamid.S1", "status": "delivered"}
    ]}}]}]}
    body = json.dumps(payload).encode("utf-8")
    resp = client.post(WEBHOOK_PREFIX, content=body, headers={"X-Hub-Signature-256": _sign(body)})
    assert resp.status_code == 202
    kiq_mock.assert_not_awaited()


def test_message_without_from_or_id_skipped(client, kiq_mock):
    payload = {"entry": [{"changes": [{"value": {"messages": [
        {"type": "text", "text": {"body": "no ids"}}  # missing from + id
    ]}}]}]}
    body = json.dumps(payload).encode("utf-8")
    resp = client.post(WEBHOOK_PREFIX, content=body, headers={"X-Hub-Signature-256": _sign(body)})
    assert resp.status_code == 202
    kiq_mock.assert_not_awaited()


class TestSendingSurvivesABlipOnTheWayToMeta:
    """
    Observed live: one `httpx.ConnectTimeout` on the POST to Meta escaped `dispatch_message`, escaped
    the turn, and handed the whole thing back to TaskIQ to re-run from the top — LLM call included.
    Worse, the idempotency claim is taken BEFORE the send, so that retry then skipped the failed
    bubble as "already sent" and the customer never received it at all.

    A blip on one socket should cost one request, not a conversation turn and not a reply.
    """

    @staticmethod
    def _service(monkeypatch, outcomes):
        """`outcomes` is a list of exceptions or status codes, consumed one per attempt."""
        import httpx
        from src.services import whatsapp as wa

        calls, released = [], []

        class _Resp:
            def __init__(self, status):
                self.status_code = status
                self.text = "boom"
                self.request = httpx.Request("POST", "https://graph.facebook.com/x")

            def json(self):
                return {"messages": [{"id": "wamid.OK"}]}

            def raise_for_status(self):
                raise httpx.HTTPStatusError("bad", request=self.request, response=self)

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json, headers):
                calls.append(url)
                outcome = outcomes[min(len(calls), len(outcomes)) - 1]
                if isinstance(outcome, Exception):
                    raise outcome
                return _Resp(outcome)

        class _Cache:
            async def check_and_set_idempotency(self, *a, **k):
                return True

            async def release_idempotency(self, *a, **k):
                released.append(a)

        monkeypatch.setattr(wa.httpx, "AsyncClient", lambda **kw: _Client())
        monkeypatch.setattr(wa.asyncio, "sleep", AsyncMock())
        return wa.WhatsAppService(_Cache()), calls, released

    async def test_a_connect_timeout_is_retried_and_the_second_attempt_lands(self, monkeypatch):
        import httpx
        service, calls, released = self._service(monkeypatch, [httpx.ConnectTimeout("t"), 200])
        await service.dispatch_message(
            thread_id="91", webhook_msg_id="wamid.IN", node_name="sales", msg_index=0, text="hi",
        )
        assert len(calls) == 2
        assert released == [], "it succeeded, so the claim must stand"

    async def test_metas_own_5xx_is_retried(self, monkeypatch):
        service, calls, released = self._service(monkeypatch, [503, 200])
        await service.dispatch_message(
            thread_id="91", webhook_msg_id="wamid.IN", node_name="sales", msg_index=0, text="hi",
        )
        assert len(calls) == 2

    async def test_a_bad_payload_is_not_retried(self, monkeypatch):
        # 400 means the payload or the token is wrong; it will be exactly as wrong next time, and
        # three copies of the same error in the log hide the real reason.
        import httpx
        service, calls, released = self._service(monkeypatch, [400])
        with pytest.raises(httpx.HTTPStatusError):
            await service.dispatch_message(
                thread_id="91", webhook_msg_id="wamid.IN", node_name="sales", msg_index=0, text="hi",
            )
        assert len(calls) == 1

    async def test_giving_up_hands_the_claim_back_so_a_retry_can_re_send(self, monkeypatch):
        import httpx
        from src.services import whatsapp as wa
        service, calls, released = self._service(monkeypatch, [httpx.ConnectTimeout("t")])
        with pytest.raises(httpx.ConnectTimeout):
            await service.dispatch_message(
                thread_id="91", webhook_msg_id="wamid.IN", node_name="sales", msg_index=0, text="hi",
            )
        assert len(calls) == wa._SEND_ATTEMPTS
        assert released, "without this the bubble is skipped as already-sent and lost for good"

    async def test_an_already_claimed_bubble_is_still_skipped(self, monkeypatch):
        from src.services import whatsapp as wa

        class _Cache:
            async def check_and_set_idempotency(self, *a, **k):
                return False

        sent = []
        monkeypatch.setattr(wa.httpx, "AsyncClient", lambda **kw: sent.append(1))
        service = wa.WhatsAppService(_Cache())
        await service.dispatch_message(
            thread_id="91", webhook_msg_id="wamid.IN", node_name="sales", msg_index=0, text="hi",
        )
        assert sent == []
