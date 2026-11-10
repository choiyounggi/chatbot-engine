"""HTTP skills (builder plan E10; docs/engine-core-plan.md "빌더 단계 E-C").

A skill is an action a bot defines in its document (`skills`): the engine POSTs a Kakao-style
request to its URL and uses the JSON object it answers as the action's result. Everything that can
go wrong is a SkillFailure (an ActionError, so a scenario answers with its `on_error` response) and
is written to the bot's skill error log; secrets and response bodies never are.

Internal addresses are refused unless the server allows them: the host is resolved first, every
address must be public, and the connection goes to the address that was checked (no second DNS
lookup an attacker could answer differently); TLS still verifies the original host name.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import ipaddress
import json
import socket
import ssl
import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from chatcore.bot.schema import SECRET_REF, Skill
from chatcore.dialogue.values import CURRENT_TURN, ActionError, ActionFn

MAX_RESPONSE = 64 * 1024  # bytes of a skill's answer
ERROR_LOG_SIZE = 100  # per bot, like Kakao's 오류 내역
FAILURE_KINDS = ("blocked", "network", "timeout", "http_status", "invalid_response", "too_large", "secret_missing")


class SkillFailure(ActionError):
    def __init__(self, kind: str, detail: str, status: int | None = None) -> None:  # kind: one of FAILURE_KINDS
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail, self.status = kind, detail, status


@dataclass(frozen=True)
class SkillError:
    """One failed call in the bot's error log (no secrets, no response body)."""

    at: str  # ISO 8601, UTC
    skill: str
    kind: str
    status: int | None
    detail: str
    ms: int
    test: bool  # a test run from the builder


@dataclass(frozen=True)
class SkillOutcome:
    ok: bool
    status: int | None
    ms: int
    result: dict[str, Any] | None
    error: SkillError | None


# ---------------------------------------------------------------- transport


def _public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified or not ip.is_global)


def resolve(host: str, port: int, allow_private: bool) -> str:
    """The address to connect to: every address the host resolves to must be public (unless allowed)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise SkillFailure("network", f"cannot resolve {host}") from exc
    addresses = [str(info[4][0]) for info in infos]
    if not addresses:
        raise SkillFailure("network", f"cannot resolve {host}")
    if not allow_private and not all(_public(a) for a in addresses):
        raise SkillFailure("blocked", f"{host} is an internal address")
    return addresses[0]


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        self._tls = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=self._tls)
        self._address = address

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._tls.wrap_socket(sock, server_hostname=self.host)  # verifies the real host name


Transport = Callable[[str, dict[str, str], bytes, float, bool], tuple[int, bytes]]


def post_json(url: str, headers: dict[str, str], body: bytes, timeout: float, allow_private: bool) -> tuple[int, bytes]:
    """POST `body` to `url`; (status, at most MAX_RESPONSE + 1 bytes). Redirects are not followed."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SkillFailure("invalid_response", "the skill URL must be http or https with a host")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    address = resolve(parts.hostname, port, allow_private)
    conn_type = _PinnedHTTPS if parts.scheme == "https" else _PinnedHTTP
    conn = conn_type(parts.hostname, port, address, timeout)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json; charset=utf-8",
                                                       "Accept": "application/json", "User-Agent": "chatcore-skill/1",
                                                       **headers})
        response = conn.getresponse()
        return response.status, response.read(MAX_RESPONSE + 1)
    except TimeoutError as exc:
        raise SkillFailure("timeout", f"no answer within {timeout:g}s") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise SkillFailure("network", type(exc).__name__) from exc
    finally:
        conn.close()


# ---------------------------------------------------------------- calls


def user_key(session: str) -> str:
    """What the skill sees as the user: a stable hash, never the session id itself."""
    return hashlib.sha256(session.encode("utf-8")).hexdigest()[:24]


class SkillClient:
    """Calls one bot's skills. `secret(name)` reads the bot's secret store; `log` records failures."""

    def __init__(self, bot_id: str, version: str, secret: Callable[[str], str | None],
                 log: Callable[[SkillError], None], allow_private: bool = False, transport: Transport = post_json,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.bot_id, self.version, self.secret, self.log = bot_id, version, secret, log
        self.allow_private, self.transport, self.clock = allow_private, transport, clock

    def call(self, skill_id: str, skill: Skill, params: Mapping[str, str], utterance: str = "", user: str = "",
             test: bool = False) -> SkillOutcome:
        """One call (the test URL for a test run, when the skill has one). Never raises."""
        url = (skill.test_url or skill.url) if test else skill.url
        start = self.clock()
        status: int | None = None
        try:
            headers = self._headers(skill)
            body = json.dumps({
                "bot": {"id": self.bot_id, "version": self.version},
                "userRequest": {"utterance": utterance, "user": {"id": user}},
                "action": {"id": skill_id, "name": skill.label or skill_id, "params": dict(params)},
            }, ensure_ascii=False).encode("utf-8")
            status, raw = self.transport(url, headers, body, skill.timeout_ms / 1000, self.allow_private)
            result = self._result(status, raw)
        except SkillFailure as exc:
            error = SkillError(dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), skill_id, exc.kind,
                               exc.status if exc.status is not None else status, exc.detail[:200],
                               self._ms(start), test)
            self.log(error)
            return SkillOutcome(False, error.status, error.ms, None, error)
        return SkillOutcome(True, status, self._ms(start), result, None)

    def action(self, skill_id: str, skill: Skill) -> ActionFn:
        """The skill as an action: params are the step's rendered args; the turn supplies the utterance."""

        def run(args: dict[str, str]) -> dict[str, Any]:
            turn = CURRENT_TURN.get()
            outcome = self.call(skill_id, skill, args, turn.utterance if turn else "",
                                user_key(turn.session) if turn else "")
            if outcome.error is not None:
                raise SkillFailure(outcome.error.kind, outcome.error.detail, outcome.error.status)
            assert outcome.result is not None
            return outcome.result

        return run

    def _headers(self, skill: Skill) -> dict[str, str]:
        def fill(m: Any) -> str:
            value = self.secret(m.group(1))
            if value is None:
                raise SkillFailure("secret_missing", f"secret {m.group(1)} is not set")
            return value

        headers = {name: SECRET_REF.sub(fill, value) for name, value in skill.headers.items()}
        if any(c in v for v in headers.values() for c in "\r\n\0"):
            raise SkillFailure("secret_missing", "a secret value has a line break")
        return headers

    @staticmethod
    def _result(status: int, raw: bytes) -> dict[str, Any]:
        if not 200 <= status < 300:
            raise SkillFailure("http_status", f"HTTP {status}", status)
        if len(raw) > MAX_RESPONSE:
            raise SkillFailure("too_large", f"the answer is over {MAX_RESPONSE // 1024} KB", status)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise SkillFailure("invalid_response", "the answer is not JSON", status) from exc
        if isinstance(data, dict) and data.get("version") == "2.0" and isinstance(data.get("data"), dict):
            data = data["data"]  # Kakao's skill data
        if not isinstance(data, dict):
            raise SkillFailure("invalid_response", "the answer is not a JSON object", status)
        return data

    def _ms(self, start: float) -> int:
        return max(0, round((self.clock() - start) * 1000))


def error_dict(error: SkillError) -> dict[str, Any]:
    return asdict(error)
