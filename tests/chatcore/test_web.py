"""E-G: web chat settings, the widget's public settings and CORS for the chat endpoints."""

import shutil
import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chatcore.api import create_app
from chatcore.bot.repository import InMemoryRepository
from chatcore.conversations import synthetic_actions
from chatcore.host import BotHost, StaticHost
from chatcore.store.memory import InMemoryStore
from chatcore.web import (
    DEFAULT_COLOR,
    DEFAULT_PLACEHOLDER,
    MAX_ORIGINS,
    OriginCache,
    WebSettings,
    normalize_origin,
    public,
)

from .conftest import SAMPLE
from .test_bot_lifecycle import postgres_repository

SHOP = "https://shop.example"
WEB = "/v1/builder/bots/sample/web"


# ---------------------------------------------------------------- origins and settings


@pytest.mark.parametrize("given, stored", [
    ("https://Shop.Example", "https://shop.example"),
    ("http://localhost:5173", "http://localhost:5173"),
    ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
    (" https://a-b.example.co.kr ", "https://a-b.example.co.kr"),
    ("http://[::1]:3000", "http://[::1]:3000"),
])
def test_an_origin_is_kept_as_a_browser_sends_it(given, stored) -> None:
    assert normalize_origin(given) == stored


@pytest.mark.parametrize("bad", ["https://shop.example/", "https://shop.example/chat", "ftp://shop.example", "shop.example",
                                 "https://", "https://-shop.example", "https://shop.example:0", "https://shop.example:65536",
                                 "null", "", "https://shop.example?x=1", "https://user@shop.example"])
def test_anything_else_is_not_an_origin(bad) -> None:
    with pytest.raises(ValueError):
        normalize_origin(bad)


def test_settings_have_defaults_and_are_normalized() -> None:
    assert WebSettings().model_dump() == {"allowed_origins": [], "title": None, "color": DEFAULT_COLOR,
                                          "placeholder": DEFAULT_PLACEHOLDER}
    s = WebSettings(allowed_origins=["HTTPS://SHOP.EXAMPLE"], title=" 상담봇 ", color="#abcdef", placeholder="질문하세요")
    assert (s.allowed_origins, s.title, s.color) == (["https://shop.example"], "상담봇", "#ABCDEF")


@pytest.mark.parametrize("bad", [
    {"allowed_origins": [SHOP, "https://SHOP.example"]},  # the same origin twice
    {"allowed_origins": [f"https://s{i}.example" for i in range(MAX_ORIGINS + 1)]},
    {"title": "   "}, {"title": "가" * 41}, {"placeholder": ""}, {"color": "blue"}, {"color": "#12345"},
    {"unknown": 1},
])
def test_settings_refuse(bad) -> None:
    with pytest.raises(ValidationError):
        WebSettings(**bad)


def test_public_settings_never_name_the_allowed_sites() -> None:
    out = public(WebSettings(allowed_origins=[SHOP]), "sample", "")
    assert out == {"bot": "sample", "title": "sample", "color": DEFAULT_COLOR, "placeholder": DEFAULT_PLACEHOLDER}


def test_the_origin_cache_reads_again_after_its_time_or_when_told() -> None:
    reads, now = [], [0.0]
    stored = {"value": WebSettings(allowed_origins=[SHOP])}

    def load(bot):
        reads.append(bot)
        return stored["value"]

    cache = OriginCache(load, clock=lambda: now[0], seconds=10)
    assert cache.allowed("sample", SHOP) and cache.allowed("sample", "HTTPS://SHOP.EXAMPLE") and len(reads) == 1
    stored["value"] = None  # removed elsewhere: still allowed until the cache expires
    now[0] = 9.9
    assert cache.allowed("sample", SHOP) and len(reads) == 1
    now[0] = 10.0
    assert not cache.allowed("sample", SHOP) and len(reads) == 2
    stored["value"] = WebSettings(allowed_origins=[SHOP])
    cache.forget("sample")
    assert cache.allowed("sample", SHOP) and len(reads) == 3


@pytest.fixture(params=["memory", "postgres"])
def repository(request):
    return InMemoryRepository() if request.param == "memory" else postgres_repository()


def test_repository_keeps_web_settings_per_bot(repository) -> None:
    bot = f"t_{uuid.uuid4().hex[:8]}"
    assert repository.web_settings(bot) is None
    repository.save_web_settings(bot, {"allowed_origins": [SHOP], "title": "가"})
    repository.save_web_settings(bot, {"allowed_origins": [], "title": "나"})
    assert repository.web_settings(bot) == {"allowed_origins": [], "title": "나"}
    assert repository.web_settings(bot + "_other") is None


# ---------------------------------------------------------------- API


@pytest.fixture
def host(fake_embedder, tmp_path, trained_sample):
    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
    h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions, keep=1,
                min_f1=0.0)
    h.seed(SAMPLE)
    return h


@pytest.fixture
def client(host):
    return TestClient(create_app(host))


def allow(client, *origins, **more):
    r = client.put(WEB, json={"allowed_origins": list(origins), **more})
    assert r.status_code == 200, r.text
    return r.json()


def test_builder_reads_defaults_then_what_was_saved(client) -> None:
    assert client.get(WEB).json() == {"bot": "sample", "allowed_origins": [], "title": None, "color": DEFAULT_COLOR,
                                      "placeholder": DEFAULT_PLACEHOLDER}
    saved = allow(client, "HTTPS://SHOP.EXAMPLE", title="날씨봇", color="#10b981")
    assert saved["allowed_origins"] == [SHOP] and saved["color"] == "#10B981"
    assert client.get(WEB).json() == saved


def test_builder_refuses_bad_settings_unknown_bots_and_bots_being_deleted(client, host) -> None:
    assert client.put(WEB, json={"allowed_origins": ["https://shop.example/path"]}).status_code == 422
    assert client.get("/v1/builder/bots/nobody/web").status_code == 404
    assert client.put("/v1/builder/bots/nobody/web", json={}).status_code == 404
    host.delete_bot("sample")
    assert client.get(WEB).status_code == 409 and client.put(WEB, json={}).status_code == 409


def test_builder_web_settings_need_the_admin_token(host) -> None:
    client = TestClient(create_app(host, "secret-token-for-tests"))
    assert client.get(WEB).status_code == 401 and client.put(WEB, json={}).status_code == 401


def test_the_widget_reads_title_color_and_placeholder_never_the_allowed_sites(client) -> None:
    assert client.get("/v1/bots/sample/web").json() == {"bot": "sample", "title": "샘플 봇", "color": DEFAULT_COLOR,
                                                        "placeholder": DEFAULT_PLACEHOLDER}
    allow(client, SHOP, title="날씨봇", placeholder="날씨를 물어보세요")
    body = client.get("/v1/bots/sample/web", headers={"Origin": SHOP}).json()
    assert body == {"bot": "sample", "title": "날씨봇", "color": DEFAULT_COLOR, "placeholder": "날씨를 물어보세요"}
    assert client.get("/v1/bots/nobody/web").status_code == 404


# ---------------------------------------------------------------- CORS


def open_chat(client, origin=None, session="w1"):
    headers = {"Origin": origin} if origin else {}
    return client.post(f"/v1/bots/sample/sessions/{session}/open", headers=headers)


def say(client, text, origin=None, session="w1"):
    headers = {"Origin": origin} if origin else {}
    return client.post(f"/v1/bots/sample/sessions/{session}/messages", json={"text": text}, headers=headers)


def test_without_an_origin_nothing_changes(client) -> None:
    r = say(client, "안녕")
    assert r.status_code == 200 and "access-control-allow-origin" not in r.headers


def test_an_allowed_site_talks_and_reads_the_answer(client) -> None:
    allow(client, SHOP)
    for r in (open_chat(client, SHOP), say(client, "안녕", SHOP), client.get("/v1/bots/sample/web", headers={"Origin": SHOP})):
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == SHOP and r.headers["vary"] == "Origin"
    assert say(client, "안녕", SHOP).json()["messages"]


def test_any_other_site_is_refused_before_the_turn(client, host) -> None:
    allow(client, SHOP)
    for origin in ("https://evil.example", "https://shop.example:8443", "http://shop.example", "null"):
        r = say(client, "안녕", origin, session="refused")
        assert r.status_code == 403 and r.json() == {"detail": "this site may not talk to this bot"}
        assert "access-control-allow-origin" not in r.headers
    assert list(host.store.events("sample:refused")) == []  # no turn ran


def test_no_site_is_allowed_until_one_is_saved(client) -> None:
    assert say(client, "안녕", SHOP).status_code == 403
    assert client.options("/v1/bots/sample/web", headers={"Origin": SHOP, "Access-Control-Request-Method": "GET"}).status_code == 403


def test_preflight_answers_for_allowed_sites_only(client) -> None:
    allow(client, SHOP)
    path = "/v1/bots/sample/sessions/w1/messages"
    ok = client.options(path, headers={"Origin": SHOP, "Access-Control-Request-Method": "POST",
                                       "Access-Control-Request-Headers": "content-type"})
    assert ok.status_code == 204
    assert {k: ok.headers[k] for k in ("access-control-allow-origin", "access-control-allow-methods",
                                       "access-control-allow-headers", "access-control-max-age")} == {
        "access-control-allow-origin": SHOP, "access-control-allow-methods": "GET, POST",
        "access-control-allow-headers": "Content-Type", "access-control-max-age": "600"}
    assert client.options(path, headers={"Origin": "https://evil.example"}).status_code == 403


def test_a_save_applies_at_once(client) -> None:
    allow(client, SHOP)
    assert say(client, "안녕", SHOP).status_code == 200
    allow(client)  # nobody
    assert say(client, "안녕", SHOP).status_code == 403


def test_other_endpoints_get_no_cors_headers(client) -> None:
    allow(client, SHOP)
    parse = client.post("/v1/bots/sample/nlu/parse", json={"text": "안녕"}, headers={"Origin": SHOP})
    builder = client.get(WEB, headers={"Origin": SHOP})
    for r in (parse, builder):
        assert r.status_code == 200 and "access-control-allow-origin" not in r.headers


def test_another_bots_list_does_not_count(client, host) -> None:
    host.repository.save_web_settings("other", {"allowed_origins": [SHOP]})
    assert say(client, "안녕", SHOP).status_code == 403


def test_unreadable_settings_answer_503_not_an_open_door(client, host, monkeypatch) -> None:
    def down(bot):
        raise OSError("database down")

    monkeypatch.setattr(host.repository, "web_settings", down)
    assert say(client, "안녕", SHOP).status_code == 503
    assert say(client, "안녕").status_code == 200  # no origin: not affected


def test_a_server_without_a_repository_allows_no_site(runtime_client) -> None:
    client, _ = runtime_client
    assert say(client, "안녕", SHOP).status_code == 403
    assert say(client, "안녕").status_code == 200


@pytest.fixture
def runtime_client(host):
    rt = host.runtime("sample")
    return TestClient(create_app(StaticHost({"sample": rt}, host.store))), rt


# ---------------------------------------------------------------- review (E-G)


def test_behind_a_path_prefix_the_check_still_runs(host) -> None:
    client = TestClient(create_app(host), root_path="/api")
    allow(client, SHOP)
    refused = client.post("/api/v1/bots/sample/sessions/rp/open", headers={"Origin": "https://evil.example"})
    assert refused.status_code == 403 and list(host.store.events("sample:rp")) == []
    ok = client.post("/api/v1/bots/sample/sessions/rp/open", headers={"Origin": SHOP})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == SHOP


def test_a_bot_this_server_does_not_serve_is_404_without_reading_settings(client, host, monkeypatch) -> None:
    reads = []
    real = host.repository.web_settings
    monkeypatch.setattr(host.repository, "web_settings", lambda bot: reads.append(bot) or real(bot))
    for n in range(3):
        r = client.options(f"/v1/bots/nobody{n}/web", headers={"Origin": SHOP})
        assert r.status_code == 404 and r.headers["vary"] == "Origin"
    assert reads == []


def test_a_stored_row_from_another_version_still_works(client, host) -> None:
    host.repository.save_web_settings("sample", {"allowed_origins": [SHOP], "title": "날씨봇", "added_later": True})
    assert say(client, "안녕", SHOP).status_code == 200
    assert client.get("/v1/bots/sample/web").json()["title"] == "날씨봇"
    assert client.get(WEB).json()["allowed_origins"] == [SHOP]
    host.repository.save_web_settings("sample", {"allowed_origins": ["not an origin"], "title": 3})
    client.put(WEB, json={"allowed_origins": [SHOP]})  # forget the cached list
    host.repository.save_web_settings("sample", {"allowed_origins": ["not an origin"], "title": 3})
    assert say(client, "안녕", SHOP, session="broken").status_code in (200, 403)  # cached or not: never 503
    assert client.get("/v1/bots/sample/web").status_code == 200 and client.get(WEB).status_code == 200


@pytest.mark.parametrize("given, stored", [("https://shop.example:443", SHOP), ("http://shop.example:80", "http://shop.example"),
                                           ("https://shop.example:80", "https://shop.example:80")])
def test_a_default_port_is_dropped_as_browsers_do(given, stored) -> None:
    assert normalize_origin(given) == stored


def test_refusals_vary_by_origin_and_a_crash_still_carries_the_header(client, host, monkeypatch) -> None:
    allow(client, SHOP)
    refused = say(client, "안녕", "https://evil.example")
    assert refused.status_code == 403 and refused.headers["vary"] == "Origin"

    def crash(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("chatcore.api.run_turn", crash)
    r = TestClient(create_app(host), raise_server_exceptions=False).post(
        "/v1/bots/sample/sessions/c/messages", json={"text": "안녕"}, headers={"Origin": SHOP})
    assert r.status_code == 500 and r.headers["access-control-allow-origin"] == SHOP
