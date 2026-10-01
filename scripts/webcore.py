"""Shared MCP transport and result handling for the websearch and webfetch commands.

Both commands are the same client underneath: they POST JSON-RPC to a provider's
MCP endpoint, open a session with `initialize`, then reach a single tool with
`tools/call`. And both face the same second problem: providers answer in whatever
shape they please, so a command has to find the URL in a result item, and decide
whether a body is content or a status message reporting a failure.

Both of those live here, once. What still differs between the commands is which
evidence they accept, and a `ResultPolicy` states that as data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


__all__ = [
    "CLIENT_NAME",
    "DEFAULT_TIMEOUT",
    "McpClient",
    "Outcome",
    "ResultPolicy",
    "URL_KEYS",
    "clamp",
    "first_url",
    "parse_json",
    "payload_failure",
    "safe_error",
    "tool_error",
]


DEFAULT_TIMEOUT = 30.0
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_PROTOCOL_VERSION = "2024-11-05"
# Each command reports itself under its own name, so the User-Agent header and the
# clientInfo block a provider sees both follow the command that sent them.
CLIENT_NAME = "websearch.py"
_CLIENT_VERSION = "1.0"
# Matched differently on purpose: an error prefix has to open the body, while a mark
# may be preceded by blank lines.
_ERROR_PREFIX = "error:"
_MARK_PREFIX = "❌"


@dataclass
class _HttpResponse:
    status: Optional[int]
    headers: Dict[str, str]
    body: str
    elapsed_ms: int
    error: Optional[str] = None


@dataclass
class Outcome:
    """One tool call, plus room for the calling command's own rendering of it.

    `text` is the only field the caller fills in; `result` and `raw_text` are the
    provider's answer exactly as it arrived.
    """

    provider: str
    status: Optional[int]
    elapsed_ms: int
    result: Optional[Dict[str, Any]] = None
    raw_text: str = ""
    text: str = ""
    error: Optional[str] = None
    raw_429: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.raw_429


# HTTP and MCP transport
# ---------------------------------------------------------------------------

def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
    user_agent: str,
) -> _HttpResponse:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": user_agent,
            **headers,
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES)
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            response_headers = getattr(response, "headers", {}) or {}
            return _HttpResponse(
                status,
                {str(k): str(v) for k, v in response_headers.items()},
                raw.decode("utf-8", "replace"),
                _elapsed_ms(started),
            )
    except HTTPError as exc:
        try:
            raw = exc.read(_MAX_RESPONSE_BYTES)
        except Exception:
            raw = b""
        response_headers = exc.headers.items() if exc.headers else ()
        return _HttpResponse(
            exc.code,
            {str(k): str(v) for k, v in response_headers},
            raw.decode("utf-8", "replace"),
            _elapsed_ms(started),
            f"HTTP {exc.code}",
        )
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return _HttpResponse(None, {}, "", _elapsed_ms(started), f"{type(exc).__name__}: {reason}")


def parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _messages(body: str) -> List[Dict[str, Any]]:
    """Decode direct JSON-RPC, JSON batches, or MCP SSE responses."""
    collected: List[Dict[str, Any]] = []

    def add(value: Any) -> None:
        if isinstance(value, dict):
            collected.append(value)
        elif isinstance(value, list):
            collected.extend(item for item in value if isinstance(item, dict))

    add(parse_json((body or "").strip()))
    data_lines: List[str] = []
    for line in (body or "").splitlines():
        line = line.rstrip("\r")
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif not line and data_lines:
            add(parse_json("\n".join(data_lines)))
            data_lines = []
    if data_lines:
        add(parse_json("\n".join(data_lines)))

    unique: List[Dict[str, Any]] = []
    seen = set()
    for message in collected:
        marker = json.dumps(message, sort_keys=True, ensure_ascii=False)
        if marker not in seen:
            seen.add(marker)
            unique.append(message)
    return unique


def _select_message(raw_messages: List[Dict[str, Any]], request_id: int) -> Optional[Dict[str, Any]]:
    for message in reversed(raw_messages):
        if message.get("id") == request_id:
            return message
    for message in reversed(raw_messages):
        if "result" in message or "error" in message:
            return message
    return None


def safe_error(value: Any, limit: int = 240) -> str:
    """One line of readable text for a provider's failure, for a diagnostic."""
    return str(value or "unknown error").replace("\r", " ").replace("\n", " ").strip()[:limit]


def _message_error(message: Optional[Dict[str, Any]]) -> Optional[str]:
    if not message:
        return "MCP response did not contain a JSON-RPC result"
    error = message.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        return safe_error(error.get("message") or error.get("code") or error)
    return safe_error(error)


def _header(headers: Dict[str, str], name: str) -> Optional[str]:
    wanted = name.lower()
    return next((value for key, value in headers.items() if key.lower() == wanted), None)


def _http_ok(response: _HttpResponse) -> bool:
    return response.error is None and response.status is not None and 200 <= response.status < 300


def _rpc(
    endpoint: str,
    request_id: Optional[int],
    method: str,
    params: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
    user_agent: str,
) -> Tuple[_HttpResponse, Optional[Dict[str, Any]]]:
    """One JSON-RPC exchange. A request_id of None sends a notification, which by
    definition carries no id and gets no reply."""
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        payload["id"] = request_id
    response = _post_json(endpoint, payload, headers, timeout, user_agent)
    if request_id is None:
        return response, None
    return response, _select_message(_messages(response.body), request_id)


def _content_text(result: Dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    return "\n\n".join(
        block["text"].strip()
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip()
    )


def _transport_failure(provider: str, response: _HttpResponse) -> Optional[Outcome]:
    """Turn a non-2xx answer into an Outcome, or None when the exchange succeeded.

    A rate limit is flagged separately from other failures because a caller
    rotating between providers needs to tell "this one is busy, try the next" from
    "this one is broken", and only reports the two differently.
    """
    if response.status == 429:
        return Outcome(provider, response.status, response.elapsed_ms, error="HTTP 429", raw_429=True)
    if not _http_ok(response):
        return Outcome(
            provider,
            response.status,
            response.elapsed_ms,
            error=response.error or f"HTTP {response.status}",
        )
    return None


# Tool calls
# ---------------------------------------------------------------------------

class McpClient:
    """One tool on one MCP endpoint, with the session it needs to reach it.

    The session opens on the first call rather than at construction, so a caller that
    never calls pays nothing for the handshake, and every later call reuses the
    session id the endpoint handed back. Callers pass positional arguments to `call`,
    which forwards them to the tool's argument builder, so the transport never needs
    to know a query from a list of URLs. `elapsed_ms` here accumulates the handshake
    and every call, where the same field on an Outcome covers only that one call.
    """

    def __init__(
        self,
        provider: str,
        endpoint: str,
        tool: str,
        args_builder: Callable[..., Dict[str, Any]],
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        client_name: str = CLIENT_NAME,
    ):
        self.provider = provider
        self.endpoint = endpoint
        self.tool = tool
        self.args_builder = args_builder
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.client_name = client_name
        self.elapsed_ms = 0
        self._ready = False
        self._next_id = 1

    def call(self, *args: Any) -> Outcome:
        """Reach the tool, opening the session first if this is the first call.

        A tool result is returned unclassified: a transport failure is already an
        error on the Outcome, but content is the caller's to judge.
        """
        if not self._ready:
            failure = self._start()
            if failure is not None:
                return failure
        request_id = self._next_id
        self._next_id += 1
        response, message = _rpc(
            self.endpoint,
            request_id,
            "tools/call",
            {"name": self.tool, "arguments": self.args_builder(*args)},
            self.headers,
            self.timeout,
            f"{self.client_name}/{_CLIENT_VERSION}",
        )
        self.elapsed_ms += response.elapsed_ms
        failure = _transport_failure(self.provider, response)
        if failure:
            return failure
        error = _message_error(message)
        if error or message is None or not isinstance(message.get("result"), dict):
            return Outcome(
                self.provider,
                response.status,
                response.elapsed_ms,
                error=error or "MCP response did not contain a tool result",
            )
        result = message["result"]
        return Outcome(
            self.provider,
            response.status,
            response.elapsed_ms,
            result=result,
            raw_text=_content_text(result),
        )

    def _start(self) -> Optional[Outcome]:
        """Open the session, returning an Outcome if the handshake failed."""
        request_id = self._next_id
        self._next_id += 1
        response, message = _rpc(
            self.endpoint,
            request_id,
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": _CLIENT_VERSION},
            },
            self.headers,
            self.timeout,
            f"{self.client_name}/{_CLIENT_VERSION}",
        )
        self.elapsed_ms += response.elapsed_ms
        failure = _transport_failure(self.provider, response)
        if failure:
            return failure
        error = _message_error(message)
        if error or message is None:
            return Outcome(self.provider, response.status, response.elapsed_ms, error=error)
        session_id = _header(response.headers, "mcp-session-id")
        if session_id:
            self.headers["Mcp-session-id"] = session_id
        # The spec has the client confirm the handshake before its first call, and a
        # server that enforces it answers the call with "session not initialized".
        # Best-effort: a server that does not want the notification still serves the
        # call, so a failure here is not worth abandoning the provider over.
        confirm, _ = _rpc(
            self.endpoint,
            None,
            "notifications/initialized",
            {},
            self.headers,
            self.timeout,
            f"{self.client_name}/{_CLIENT_VERSION}",
        )
        self.elapsed_ms += confirm.elapsed_ms
        self._ready = True
        return None


# Provider results
# ---------------------------------------------------------------------------

# The key a provider's result item carries its URL under. No two agree: the same
# endpoint spells it `url` on a search hit and `sourceURL` on a scrape, and mirrors
# add `link` or `href`. The list is ordered so the most specific spellings win,
# which only matters for an item carrying more than one of them.
URL_KEYS = ("url", "sourceURL", "source_url", "id", "link", "href")


def first_url(*sources: Any) -> Optional[str]:
    """The first usable URL in the given result items, or None if none of them has one.

    Several sources are searched in order, because a provider may put the URL on the
    item or inside its `metadata`. Only http and https count: a value such as a
    fragment or a file path is not something a caller could fetch, and treating it
    as a URL would pair content with an address that cannot be re-fetched.
    """
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in URL_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
    return None


def clamp(text: str, limit: int, marker: str, *, rstrip: bool = False) -> str:
    """Cut `text` to `limit` characters and say so, or return it whole.

    A limit of zero or less means no limit. `marker` names the limit that was hit, so
    a per-result cut is told from a whole-page one.
    """
    if limit > 0 and len(text) > limit:
        head = text[:limit]
        return (head.rstrip() if rstrip else head) + marker
    return text


@dataclass(frozen=True)
class ResultPolicy:
    """What counts as evidence that a provider's answer is a failure, not content.

    `has_content` and `envelope` are the command's own judgement: only it knows what a
    result looks like, and only it knows which providers use a failure envelope.
    """

    # Matched case-insensitively within `probe_chars`.
    markers: Tuple[str, ...] = ()
    # Zero scans the whole body, which is right for a short report and wrong for a
    # long page, where a marker quoted deep in the content is the content.
    probe_chars: int = 0
    has_content: Optional[Callable[[Any], bool]] = None
    failure_keys: Tuple[str, ...] = ("error", "message")
    # Consulted only to fill in a failure already established by a `code`, so a status
    # code is reported with whatever explanation came with it.
    report_fallbacks: Tuple[str, ...] = ()
    # `code` values that mean the provider gave no code.
    code_sentinels: Tuple[Any, ...] = (None, "")
    report_code: bool = False
    flags_override_content: bool = False
    flags_alone_count: bool = False
    envelope: Optional[Callable[[Any], Optional[str]]] = None


def _flagged(payload: Dict[str, Any]) -> bool:
    """Whether a provider set an explicit flag saying the request failed."""
    return payload.get("success") is False or payload.get("ok") is False


def payload_failure(payload: Dict[str, Any], policy: ResultPolicy) -> Optional[str]:
    """The failure a structured body names, or None when it names none or carries content."""
    if policy.has_content is not None and policy.has_content(payload) and not (
        policy.flags_override_content and _flagged(payload)
    ):
        return None
    for key in policy.failure_keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return value
    code = payload.get("code")
    if policy.report_code and code not in policy.code_sentinels:
        for key in policy.report_fallbacks:
            value = payload.get(key)
            if value not in (None, "", [], {}):
                return value
        return code
    if policy.flags_alone_count and _flagged(payload):
        return "provider reported the request failed"
    return None


def _status_message(text: str, policy: ResultPolicy) -> Optional[str]:
    """Whether a prose body is reporting a failure rather than returning content.

    Only reached for a body that yielded no results, so a marker anywhere in it is
    worth reading as evidence rather than as prose that happens to quote the phrase.
    """
    lower = (text or "").lower()
    probe = lower[: policy.probe_chars] if policy.probe_chars > 0 else lower
    if (
        lower.startswith(_ERROR_PREFIX)
        or lower.lstrip().startswith(_MARK_PREFIX)
        or any(marker in probe for marker in policy.markers)
    ):
        return safe_error(text)
    return None


def tool_error(
    result: Dict[str, Any],
    text: str,
    policy: ResultPolicy,
    found: int = 0,
) -> Optional[str]:
    """Report a provider's answer as a failure, or None when it is usable content.

    Checked in order of how explicit the evidence is: a flag the provider set itself,
    then a structured body that names a failure, then a body that reads like a status
    message. The first that fires wins.

    `found` is how many results or pages the caller extracted. A body that produced
    something is content, so the prose scan is skipped: a page titled "Token signature
    failed: verify the secret" quotes a marker and is still the right answer, and
    rejecting it would discard a correct result to fix a rare false positive. Only a
    body that produced nothing has a marker to be evidence of anything.
    """
    if result.get("isError") is True:
        return safe_error(text or "provider returned isError=true")
    parsed = parse_json(text)
    if policy.envelope is not None:
        reported = policy.envelope(parsed)
        if reported:
            return reported
    if isinstance(parsed, dict):
        reported = payload_failure(parsed, policy)
        if reported is not None:
            return safe_error(reported)
    if found > 0:
        return None
    return _status_message(text, policy)

