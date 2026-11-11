"""Builder stage E-C: HTTP skills (E10) against a real local skill server.

Done criterion: a test skill server answers, fails and times out; the bot uses the answer,
falls back to its error response otherwise, and every failure is in the skill error log.
"""

import copy
import json
import shutil
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chatcore import skills as skills_module
from chatcore.api import create_app
from chatcore.bot.loader import read_document, validate
from chatcore.bot.repository import SKILL_ERRORS_KEPT, InMemoryRepository
from chatcore.bot.schema import BotDefinition, Skill
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.values import CURRENT_TURN, ActionError, ActionRegistry, TurnContext
from chatcore.host import BotHost
from chatcore.skills import MAX_RESPONSE, SkillClient, SkillFailure, _public, post_json, resolve, user_key
from chatcore.store.memory import InMemoryStore

from .conftest import SAMPLE
from .test_builder import B, draft, finish, postgres_repository, save


class SkillServer:
    """A local skill server: each path is a behaviour; requests are recorded."""

    def __init__(self):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def do_POST(self):  # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if self.path == "/slow":
                    time.sleep(1.5)
                if self.path == "/drip":  # every read is quick, the whole answer is not
                    self.send_response(200)
                    self.send_header("Content-Length", "40")
                    self.end_headers()
                    try:
                        for _ in range(40):
                            self.wfile.write(b" ")
                            self.wfile.flush()
                            time.sleep(0.1)
                    except OSError:
                        pass
                    return
                answers = {
                    "/ok": (200, json.dumps({"order": "A-1", "echo": body["action"]["params"]})),
                    "/slow": (200, "{}"),
                    "/kakao": (200, json.dumps({"version": "2.0", "data": {"order": "K-2"}})),
                    "/fail": (500, "{}"),
                    "/notjson": (200, "<html>"),
                    "/array": (200, "[1, 2]"),
                    "/big": (200, json.dumps({"x": "a" * (MAX_RESPONSE + 10)})),
                    "/redirect": (302, ""),
                }
                status, text = answers.get(self.path.split("?")[0], (404, "{}"))
                payload = text.encode("utf-8")
                self.send_response(status)
                if status == 302:
                    self.send_header("Location", "http://169.254.169.254/")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server():
    s = SkillServer()
    yield s
    s.close()


def client(errors, secrets=None, allow_private=True, **kw):
    secrets = secrets or {}
    return SkillClient("shop", "v1", secrets.get, errors.append, allow_private=allow_private, **kw)


def skill(url, **kw):
    return Skill.model_validate({"url": url, **kw})


# ---------------------------------------------------------------- the client


def test_a_call_sends_a_kakao_style_request_and_returns_the_answer(server):
    errors = []
    sk = skill(f"{server.url}/ok", label="주문 조회", headers={"Authorization": "Bearer {{ secret.KEY }}", "X-Shop": "1"})
    out = client(errors, {"KEY": "s3cret"}).call("get_order", sk, {"no": "7"}, "주문 7번 어디야", user_key("shop:u1"))
    assert out.ok and out.status == 200 and out.result == {"order": "A-1", "echo": {"no": "7"}} and out.error is None
    request = server.requests[0]
    assert request["body"] == {"bot": {"id": "shop", "version": "v1"},
                               "userRequest": {"utterance": "주문 7번 어디야", "user": {"id": user_key("shop:u1")}},
                               "action": {"id": "get_order", "name": "주문 조회", "params": {"no": "7"}}}
    assert request["headers"]["Authorization"] == "Bearer s3cret" and request["headers"]["X-Shop"] == "1"
    assert "shop:u1" not in json.dumps(request)  # the session id itself is never sent
    assert errors == []


def test_kakao_skill_data_is_unwrapped(server):
    assert client([]).call("k", skill(f"{server.url}/kakao"), {}).result == {"order": "K-2"}


@pytest.mark.parametrize("path, kind, status", [
    ("/fail", "http_status", 500), ("/notjson", "invalid_response", 200), ("/array", "invalid_response", 200),
    ("/big", "too_large", 200), ("/redirect", "http_status", 302), ("/missing", "http_status", 404),
])
def test_failures_are_logged_with_their_kind(server, path, kind, status):
    errors = []
    out = client(errors).call("s", skill(f"{server.url}{path}"), {"no": "1"}, test=True)
    assert not out.ok and out.result is None and out.error is not None
    assert (out.error.kind, out.error.status, out.error.test, out.error.skill) == (kind, status, True, "s")
    assert errors == [out.error]
    assert len(server.requests) == 1  # the redirect was not followed (to a metadata address)


def test_a_slow_skill_times_out(server):
    errors = []
    out = client(errors).call("s", skill(f"{server.url}/slow", timeout_ms=300), {})
    assert out.error is not None and out.error.kind == "timeout" and 250 <= out.error.ms < 1500


def test_a_missing_secret_fails_before_anything_is_sent(server):
    errors = []
    out = client(errors).call("s", skill(f"{server.url}/ok", headers={"K": "{{ secret.NOPE }}"}), {})
    assert out.error is not None and out.error.kind == "secret_missing" and "NOPE" in out.error.detail
    assert server.requests == []
    bad = client([], {"K": "a\r\nX-Evil: 1"}).call("s", skill(f"{server.url}/ok", headers={"K": "{{ secret.K }}"}), {})
    assert bad.error is not None and bad.error.kind == "secret_missing" and server.requests == []  # no header injection


def test_the_test_run_uses_the_test_url(server):
    sk = skill(f"{server.url}/fail", test_url=f"{server.url}/ok")
    assert client([]).call("s", sk, {}, test=True).ok
    assert not client([]).call("s", sk, {}).ok  # a real call uses the real URL
    assert [r["path"] for r in server.requests] == ["/ok", "/fail"]


def test_the_error_log_has_no_secrets_or_bodies(server):
    errors = []
    client(errors, {"KEY": "s3cret"}).call("s", skill(f"{server.url}/fail", headers={"A": "{{ secret.KEY }}"}), {"x": "y"})
    assert "s3cret" not in json.dumps([e.__dict__ for e in errors]) and errors[0].detail == "HTTP 500"


# ---------------------------------------------------------------- internal addresses


@pytest.mark.parametrize("address, public", [
    ("8.8.8.8", True), ("2606:4700:4700::1111", True), ("127.0.0.1", False), ("10.1.2.3", False), ("172.16.0.1", False),
    ("192.168.0.1", False), ("169.254.169.254", False), ("0.0.0.0", False), ("::1", False), ("fe80::1", False),
    ("::ffff:127.0.0.1", False), ("100.64.0.1", False), ("224.0.0.1", False),
])
def test_only_public_addresses_count_as_public(address, public):
    assert _public(address) is public


def test_internal_addresses_are_blocked_unless_allowed(server, monkeypatch):
    errors = []
    out = client(errors, allow_private=False).call("s", skill(f"{server.url}/ok"), {})
    assert out.error is not None and out.error.kind == "blocked" and server.requests == []
    # a host that resolves to a public AND an internal address is blocked too (no picking the good one)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("8.8.8.8", 80)), (0, 0, 0, "", ("10.0.0.5", 80))])
    with pytest.raises(SkillFailure, match="internal"):
        resolve("mixed.example", 80, allow_private=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(SkillFailure, match="cannot resolve"):
        resolve("nothing.example", 80, allow_private=False)


def test_the_connection_goes_to_the_address_that_was_checked(server, monkeypatch):
    # DNS rebinding: the request must reach the resolved address, with the original Host header
    port = server.server.server_address[1]
    looked_up = []

    def fake_resolve(host, p, allow_private):
        looked_up.append(host)
        return ["127.0.0.1"]

    monkeypatch.setattr(skills_module, "resolve", fake_resolve)
    status, raw = post_json(f"http://shop.example.test:{port}/ok", {}, json.dumps(
        {"action": {"params": {}}}).encode(), 2.0, allow_private=False)
    assert status == 200 and json.loads(raw)["order"] == "A-1"
    assert looked_up == ["shop.example.test"] and server.requests[0]["headers"]["Host"] == f"shop.example.test:{port}"


def test_https_verifies_the_certificate_against_the_host_name(tmp_path):
    import ssl
    import subprocess

    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    tls = SkillServer()
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(str(cert), str(key))
    tls.server.socket = context.wrap_socket(tls.server.socket, server_side=True)
    port = tls.server.server_address[1]
    try:
        out = client([]).call("s", skill(f"https://127.0.0.1:{port}/ok"), {})
        assert out.error is not None and out.error.kind == "network"  # self-signed: refused, nothing trusted blindly
        assert out.error.detail == "SSLCertVerificationError"
        assert tls.requests == []
    finally:
        tls.close()


def test_a_query_string_is_kept_and_only_http_urls_are_called(server):
    assert client([]).call("s", skill(f"{server.url}/ok?shop=1"), {}).ok
    assert server.requests[0]["path"] == "/ok?shop=1"
    with pytest.raises(SkillFailure, match="http or https"):
        post_json("ftp://example.com/x", {}, b"{}", 1.0, allow_private=True)


def test_unresolvable_and_unreachable_hosts_are_network_failures():
    out = client([]).call("s", skill("http://does-not-exist.invalid/x"), {})
    assert out.error is not None and out.error.kind == "network"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    refused = client([]).call("s", skill(f"http://127.0.0.1:{closed}/x"), {})
    assert refused.error is not None and refused.error.kind == "network"


# ---------------------------------------------------------------- schema, loader, registry


@pytest.mark.parametrize("fields, message", [
    ({"url": "ftp://x/y"}, "http"), ({"url": "https://"}, "http"), ({"url": "https://x", "timeout_ms": 50}, "greater"),
    ({"url": "https://x", "timeout_ms": 20_000}, "less"), ({"url": "https://x", "headers": {"A": "{{ slots.x }}"}}, "secret.NAME"),
    ({"url": "https://x", "headers": {"A": "a\nb"}}, "line break"), ({"url": "https://x", "headers": {"A B": "1"}}, "header name"),
])
def test_skill_definitions_are_checked(fields, message):
    with pytest.raises(ValidationError, match=message):
        Skill.model_validate(fields)


def test_scenarios_call_skills_by_id_and_a_skill_cannot_shadow_a_built_in_action():
    doc = read_document(SAMPLE)
    doc["skills"] = {"get_order": {"url": "https://x.example/order"}, "now": {"url": "https://x.example/now"}}
    doc["scenarios"]["greet"]["steps"] = [{"action": "get_order"}, {"respond": "greeting"}]
    bot = BotDefinition.model_validate({**doc, "version": "v"})
    errors = validate(bot, synthetic_actions().names()).errors
    assert errors == ["skill 'now' has the name of a built-in action; rename it"]
    doc["scenarios"]["greet"]["steps"] = [{"action": "nope"}, {"respond": "greeting"}]
    assert any("unknown action 'nope'" in e for e in validate(BotDefinition.model_validate({**doc, "version": "v"}),
                                                                   synthetic_actions().names()).errors)


def test_an_extended_registry_keeps_the_built_ins_and_runs_skills_with_their_own_timeout_and_the_turn():
    base = ActionRegistry(timeout=5)
    base.register("one", lambda a: {"v": 1})
    seen = []

    def slow(args):
        seen.append(CURRENT_TURN.get())
        time.sleep(0.3)
        return {}

    extended = base.extended({"slow": (slow, 0.05), "fast": (lambda a: {"turn": CURRENT_TURN.get().utterance}, 1.0)})
    assert extended.names() == {"one", "slow", "fast"} and base.names() == {"one"}
    token = CURRENT_TURN.set(TurnContext("안녕", "b:s"))
    try:
        assert extended.run("fast", {}) == {"turn": "안녕"}  # the worker thread sees the turn
        with pytest.raises(ActionError, match="timed out after 0.05s"):
            extended.run("slow", {})
    finally:
        CURRENT_TURN.reset(token)
    assert extended._pool_owner is base  # one shared thread pool
    with pytest.raises(ValueError, match="positive"):
        base.register("bad", lambda a: {}, timeout=0)


# ---------------------------------------------------------------- repositories


@pytest.fixture(params=["memory", "postgres"])
def repository(request):
    return InMemoryRepository() if request.param == "memory" else postgres_repository()


def test_repository_secrets_and_a_bounded_error_log(repository):
    bot = f"t_{uuid.uuid4().hex[:8]}"
    assert repository.secret(bot, "KEY") is None and repository.secrets(bot) == []
    repository.set_secret(bot, "KEY", "one")
    repository.set_secret(bot, "KEY", "two")
    repository.set_secret(bot, "A", "x")
    assert repository.secret(bot, "KEY") == "two"
    assert [s.name for s in repository.secrets(bot)] == ["A", "KEY"]
    assert repository.delete_secret(bot, "A") is True and repository.delete_secret(bot, "A") is False
    for i in range(SKILL_ERRORS_KEPT + 5):
        repository.add_skill_error(bot, {"skill": "s", "n": i})
    kept = repository.skill_errors(bot)
    assert len(kept) == SKILL_ERRORS_KEPT and kept[0]["n"] == SKILL_ERRORS_KEPT + 4  # newest first, oldest dropped
    assert len(repository.skill_errors(bot, limit=3)) == 3 and repository.skill_errors(f"{bot}x") == []


# ---------------------------------------------------------------- the bot and the API (done criterion)


@pytest.fixture
def host(fake_embedder, tmp_path, trained_sample):
    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
    h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions, keep=1,
                min_f1=0.0, skill_allow_private=True)
    h.seed(SAMPLE)
    return h


@pytest.fixture
def api(host):
    return TestClient(create_app(host))


def with_order_skill(doc, url):
    doc = copy.deepcopy(doc)
    doc["skills"] = {"get_order": {"label": "주문 조회", "url": url, "headers": {"Authorization": "Bearer {{ secret.SHOP_KEY }}"},
                                   "timeout_ms": 500}}
    doc["intents"]["order"] = {"examples": ["내 주문 어디쯤 왔어", "주문 조회해줘", "배송 어디야", "주문 상태 알려줘", "택배 언제 와"]}
    doc["scenarios"]["track_order"] = {"trigger": {"intent": "order"}, "on_error": "order_unavailable",
                                       "steps": [{"action": "get_order", "args": {"from": "chat"}}, {"respond": "order_status"}]}
    doc["responses"]["order_status"] = {"text": "주문 {{ results.get_order.order }} 확인했어요."}
    doc["responses"]["order_unavailable"] = {"text": "지금은 주문을 확인할 수 없어요."}
    return doc


def test_a_bot_uses_its_skill_falls_back_on_failure_and_logs_every_failure(api, host, server):
    assert api.put(f"{B}/secrets/SHOP_KEY", json={"value": "k-123"}).status_code == 204
    assert save(api, with_order_skill(draft(api), f"{server.url}/ok")).json()["valid"]
    build = finish(api, api.post(f"{B}/test-builds"))["result"]["build"]
    chat = f"{B}/test/{build}/sessions/s1/messages"
    ok = api.post(chat, json={"text": "주문 조회해줘"}).json()
    assert [m["text"] for m in ok["messages"]] == ["주문 A-1 확인했어요."]
    assert ok["actions"] == [{"scenario": "track_order", "action": "get_order", "ok": True}]
    sent = server.requests[-1]
    assert sent["headers"]["Authorization"] == "Bearer k-123" and sent["body"]["userRequest"]["utterance"] == "주문 조회해줘"
    assert sent["body"]["action"]["params"] == {"from": "chat"}

    for path, kind in (("/fail", "http_status"), ("/slow", "timeout")):
        assert save(api, with_order_skill(draft(api), f"{server.url}{path}")).json()["valid"]
        build = finish(api, api.post(f"{B}/test-builds"))["result"]["build"]
        out = api.post(f"{B}/test/{build}/sessions/s-{kind}/messages", json={"text": "주문 조회해줘"}).json()
        assert [m["response"] for m in out["messages"]] == ["order_unavailable"], kind
        assert out["actions"] == [{"scenario": "track_order", "action": "get_order", "ok": False}]
    errors = api.get(f"{B}/skills/errors").json()["items"]
    assert [(e["skill"], e["kind"], e["test"]) for e in errors] == [("get_order", "timeout", False), ("get_order", "http_status", False)]
    assert errors[1]["status"] == 500 and "k-123" not in json.dumps(errors)


def test_the_builder_can_test_run_a_skill_of_the_saved_draft(api, server):
    doc = with_order_skill(draft(api), f"{server.url}/fail")
    doc["skills"]["get_order"]["test_url"] = f"{server.url}/kakao"
    assert save(api, doc).status_code == 200
    ran = api.post(f"{B}/skills/get_order/test", json={"params": {"no": "1"}, "utterance": "주문"})
    assert ran.status_code == 200 and ran.json()["ok"] is False and ran.json()["error"]["kind"] == "secret_missing"
    api.put(f"{B}/secrets/SHOP_KEY", json={"value": "k"})
    ran = api.post(f"{B}/skills/get_order/test", json={"params": {"no": "1"}}).json()
    assert ran["ok"] and ran["status"] == 200 and ran["result"] == {"order": "K-2"} and ran["error"] is None
    assert api.post(f"{B}/skills/nope/test", json={}).status_code == 404
    assert api.get(f"{B}/skills/errors").json()["items"][0]["test"] is True  # the failed test run is logged
    assert api.post("/v1/builder/bots/nobot/skills/get_order/test", json={}).status_code == 404


def test_secrets_are_write_only(api):
    assert api.get(f"{B}/secrets").json() == {"bot": "sample", "items": []}
    assert api.put(f"{B}/secrets/API_KEY", json={"value": "abc"}).status_code == 204
    listed = api.get(f"{B}/secrets").json()["items"]
    assert [s["name"] for s in listed] == ["API_KEY"] and "abc" not in json.dumps(listed)
    assert api.put(f"{B}/secrets/API_KEY", json={"value": "a\nb"}).status_code == 422
    assert api.put(f"{B}/secrets/API_KEY", json={"value": ""}).status_code == 422
    assert api.put(f"{B}/secrets/bad-name", json={"value": "x"}).status_code == 422
    assert api.delete(f"{B}/secrets/API_KEY").status_code == 204
    assert api.delete(f"{B}/secrets/API_KEY").status_code == 404
    assert "abc" not in json.dumps(api.get(f"{B}/draft").json())


def test_the_default_server_blocks_internal_skill_addresses(fake_embedder, tmp_path, trained_sample, server):
    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
    h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions, min_f1=0.0)
    h.seed(SAMPLE)
    api = TestClient(create_app(h))
    save(api, with_order_skill(draft(api), f"{server.url}/ok"))
    api.put(f"{B}/secrets/SHOP_KEY", json={"value": "k"})
    ran = api.post(f"{B}/skills/get_order/test", json={}).json()
    assert ran["error"]["kind"] == "blocked" and server.requests == []


# ---------------------------------------------------------------- review fixes


def test_the_timeout_covers_the_whole_call_not_each_read(server):
    errors = []
    start = time.monotonic()
    out = client(errors).call("s", skill(f"{server.url}/drip", timeout_ms=500), {})
    assert time.monotonic() - start < 1.5  # the drip would take 4 s
    assert out.error is not None and out.error.kind == "timeout" and errors == [out.error]


@pytest.mark.parametrize("fields, message", [
    ({"url": "https://x.example/주문"}, "ASCII"), ({"url": "https://x.example:99999/a"}, "port"),
    ({"url": "https://x.example:abc/a"}, "port"), ({"url": "https://[::1/a"}, "host"),
    ({"url": "https://x.example/a", "headers": {"X-Name": "주문"}}, "ASCII"),
])
def test_urls_and_headers_http_cannot_send_are_refused(fields, message):
    with pytest.raises(ValidationError, match=message):
        Skill.model_validate(fields)


def test_a_call_never_raises_even_on_unexpected_errors(server):
    errors = []

    def broken(*args):
        raise ValueError("odd")

    out = client(errors, transport=broken).call("s", skill(f"{server.url}/ok"), {})
    assert out.error is not None and out.error.kind == "network" and errors == [out.error]
    bad_secret = client([], {"K": "한글값"}).call("s", skill(f"{server.url}/ok", headers={"K": "{{ secret.K }}"}), {})
    assert bad_secret.error is not None and bad_secret.error.kind == "secret_missing" and server.requests == []


def test_a_failing_error_log_does_not_break_the_call(server):
    def log(error):
        raise OSError("database down")

    out = SkillClient("shop", "v1", {}.get, log, allow_private=True).call("s", skill(f"{server.url}/fail"), {})
    assert out.error is not None and out.error.kind == "http_status"


@pytest.mark.parametrize("address", ["64:ff9b::a9fe:a9fe", "64:ff9b::a00:1", "64:ff9b::7f00:1"])
def test_nat64_addresses_are_not_public(address):
    assert _public(address) is False  # 169.254.169.254, 10.0.0.1, 127.0.0.1 behind a NAT64 gateway


def test_every_checked_address_is_tried_in_turn(server, monkeypatch):
    port = server.server.server_address[1]
    real = socket.getaddrinfo
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: [
        (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", port, 0, 0)),  # nothing listens there
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", port))] if host == "dual.example.test" else real(host, *a, **k))
    out = client([]).call("s", skill(f"http://dual.example.test:{port}/ok"), {})
    assert out.ok, out.error


def test_secret_values_http_cannot_send_are_refused(api):
    assert api.put(f"{B}/secrets/KEY", json={"value": "한글"}).status_code == 422


def test_skill_changes_are_in_the_edit_history(api, server):
    doc = with_order_skill(draft(api), f"{server.url}/ok")
    save(api, doc)
    doc["skills"]["get_order"]["url"] = f"{server.url}/kakao"
    save(api, doc)
    latest = api.get(f"{B}/edits").json()["items"][0]["changes"]
    assert {"area": "skill", "id": "get_order", "label": "주문 조회", "action": "changed",
            "examples_added": None, "examples_removed": None} in latest
