#!/usr/bin/env python3
"""Fetch one or more URLs with sequential, partial-result provider fallback."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

from webcore import (
    DEFAULT_TIMEOUT,
    McpClient,
    ResultPolicy,
    clamp,
    first_url,
    parse_json,
    safe_error,
    tool_error,
)


CLIENT_NAME = "webfetch.py"
DEFAULT_MAX_CHARS = 3000
# Extractions shorter than this are treated as failed rather than returned. Providers
# answer some unhandled pages with a title-only stub or a rate-limit notice under a 200
# status; accepting those ends the fallback chain and reports a blank page as content.
MIN_USEFUL_CHARS = 200
# A provider that cannot parse a format may answer with the file's own bytes. Both patterns
# allow for the leading "[truncated: N chars retrieved, showing first M]" notice such a
# response carries, and neither can begin real prose.
_RAW_PDF_MAGIC = re.compile(r"\A\s*(?:\[[^\]]*truncated[^\]]*\]\s*)?%PDF-\d", re.IGNORECASE)
# The magic number is definitive; a leading object header on its own is not, since prose
# can open with "1 0 obj". Requiring a second structural token keeps that case as content.
_RAW_PDF_OBJECT = re.compile(r"\A\s*(?:\[[^\]]*truncated[^\]]*\]\s*)?\d+\s+\d+\s+obj\b")
_PDF_STRUCTURE = re.compile(r"\b(?:endobj|endstream|xref|trailer)\b")
ERROR_MARKERS = (
    "unauthorized action", "monthly_cap_reached", "invalid api key",
    "invalid ser papi api key", "token signature failed", "payment required",
    "valid oauth bearer token required", "failed to retrieve content",
    "you've hit", "you reached", "rate limit", "quota exceeded",
)
# A provider that fails reports a structured envelope rather than raising: a declared
# result collection that is empty alongside a populated error list. These key sets let
# that shape be told apart from a page whose own content happens to be a JSON object.
ENVELOPE_ERROR_KEYS = (
    "errors", "error", "error_type", "failures", "failed_results", "failed", "unfetched_results",
)
ENVELOPE_COLLECTION_KEYS = ("results", "data", "pages", "items", "documents", "outputs")


@dataclass
class FetchPage:
    url: str
    text: str
    provider: str


@dataclass(frozen=True)
class FetchProvider:
    name: str
    endpoint: str
    tool: str
    args_builder: Callable[[List[str], int], Dict[str, Any]]
    headers: Dict[str, str] = field(default_factory=dict)
    max_batch: int = 1


# Result extraction
# ---------------------------------------------------------------------------

def _url_from_item(item: Dict[str, Any]) -> Optional[str]:
    return first_url(item, item.get("metadata"))


def _content_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, list):
        values = [_content_from_item(value) for value in item]
        return "\n\n".join(value for value in values if value)
    if not isinstance(item, dict):
        return ""
    for key in ("markdown", "raw_content", "text", "body"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("content", "excerpts", "highlights"):
        value = item.get(key)
        text = _content_from_item(value)
        if text:
            return text
    return ""


def _collect_pages(value: Any, inherited_url: Optional[str] = None, found: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    found = {} if found is None else found
    if isinstance(value, list):
        for item in value:
            _collect_pages(item, inherited_url, found)
        return found
    if not isinstance(value, dict):
        return found
    url = _url_from_item(value) or inherited_url
    content = _content_from_item(value)
    if url and content:
        found[url] = content
    for key in ("results", "data", "pages", "items", "documents", "outputs", "content"):
        child = value.get(key)
        if isinstance(child, (dict, list)):
            _collect_pages(child, url, found)
    return found


def _normal_url(url: str) -> str:
    return url.strip().rstrip("/")


def _assign_plain_urls(text: str, urls: Sequence[str]) -> Dict[str, str]:
    if len(urls) == 1:
        return {urls[0]: text.strip()} if text.strip() else {}
    positions: List[Tuple[int, str]] = []
    for url in urls:
        marker = f"URL: {url}"
        position = text.find(marker)
        if position < 0:
            position = text.find(url)
        if position >= 0:
            positions.append((position, url))
    if len(positions) != len(urls):
        return {}
    positions.sort()
    pages: Dict[str, str] = {}
    for index, (position, url) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        chunk = text[position:end].strip()
        for marker in (f"URL: {url}\n", f"URL: {url}"):
            if chunk.startswith(marker):
                chunk = chunk[len(marker):].lstrip()
                break
        if chunk:
            pages[url] = chunk
    return pages


def _unparsed_reason(text: str) -> Optional[str]:
    """Name the format when a provider handed back the file's bytes instead of its text.

    A provider asked to extract a format it cannot parse still answers HTTP 200 with a
    plausible-looking body: for a PDF that is the file's own source, opening with the
    "%PDF-1.7" magic number and continuing through object definitions and stream markers.
    The length threshold accepts that happily, so the caller is told it fetched a page when
    it received file internals, and the fallback chain stops on the first such answer.

    Only the head of the body is inspected, and only against markers that cannot begin real
    prose, so an article *about* PDFs is not mistaken for one.
    """
    head = (text or "")[:2000]
    if _RAW_PDF_MAGIC.match(head):
        return "provider returned raw PDF bytes rather than extracted text"
    if _RAW_PDF_OBJECT.match(head) and _PDF_STRUCTURE.search(head):
        return "provider returned raw PDF object data rather than extracted text"
    return None


def _extract_pages(
    result: Dict[str, Any],
    raw_text: str,
    urls: Sequence[str],
    provider: str,
    max_chars: int,
) -> Tuple[Dict[str, FetchPage], Optional[str]]:
    structured = result.get("structuredContent")
    found = _collect_pages(structured) if structured is not None else {}
    parsed = parse_json(raw_text) if raw_text else None
    if not found and parsed is not None:
        found = _collect_pages(parsed)
    if not found and len(urls) == 1:
        candidate = _content_from_item(structured) or _content_from_item(parsed)
        if candidate:
            found = {urls[0]: candidate}
    if not found and raw_text:
        found = _assign_plain_urls(raw_text, urls)
    # Judge an extraction against what was actually asked for: when the caller caps
    # pages below MIN_USEFUL_CHARS, a short page is the requested outcome, not a failure.
    threshold = max(1, min(MIN_USEFUL_CHARS, max_chars))
    pages: Dict[str, FetchPage] = {}
    unparsed: Optional[str] = None
    for url in urls:
        text = found.get(url) or found.get(_normal_url(url))
        if not text or len(text.strip()) < threshold:
            continue
        reason = _unparsed_reason(text)
        if reason:
            # Not a page. Treating the bytes as content would spend the caller's budget on
            # file internals, so it is dropped and the chain moves on to a provider that
            # can actually read the format.
            unparsed = unparsed or reason
            continue
        pages[url] = FetchPage(url, text.strip(), provider)
    return pages, unparsed


def _envelope_error(value: Any) -> Optional[str]:
    """Describe a provider's structured failure envelope, or None if this is content.

    Without this, a fetch failure is indistinguishable from a successful fetch of a
    page: the single-URL path assigns whatever text came back to the requested URL, so
    a JSON error body is rendered as the page body and the command reports success.
    """
    if not isinstance(value, dict) or not any(key in value for key in ENVELOPE_ERROR_KEYS):
        return None
    for key in ENVELOPE_COLLECTION_KEYS:
        if not (isinstance(value.get(key), list) and not value[key]):
            continue
        errors = value.get("errors") or value.get("failed_results") or value.get("failures")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                detail = first.get("error_type") or first.get("error") or first.get("message")
                status = first.get("http_status_code")
            else:
                detail, status = first, None
            message = safe_error(detail or "provider reported errors and returned no content")
            if status not in (None, "", 0, "0"):
                return f"{message} (HTTP {status})"
            return message
        return safe_error(value.get("error") or "provider returned an error envelope with no content")
    return None


# The keys a fetch provider uses to carry an actual page back. A structured body with
# none of them describes a failure however well formed it is, which is what separates
# an error envelope from a page whose own content happens to be JSON.
PAYLOAD_KEYS = (
    "data", "results", "content", "items", "pages", "documents", "outputs",
    "markdown", "text", "raw_content", "excerpts", "highlights", "url", "sourceURL",
)


FETCH_POLICY = ResultPolicy(
    markers=ERROR_MARKERS,
    # A page may quote any marker while explaining it, so only the head is scanned.
    probe_chars=1000,
    # A body carrying a page key is content even when it is a JSON document in its
    # own right, unless the provider also says it failed.
    has_content=lambda payload: any(key in payload for key in PAYLOAD_KEYS),
    failure_keys=("error",),
    report_fallbacks=("message",),
    # A page that happens to carry a status field of its own is not a failed fetch.
    code_sentinels=(None, "", 0, "0", False),
    report_code=True,
    flags_override_content=True,
    envelope=_envelope_error,
)


# Provider definitions
# ---------------------------------------------------------------------------

def _exa_args(urls: List[str], max_chars: int) -> Dict[str, Any]:
    return {"urls": urls, "maxCharacters": max_chars}


def _parallel_args(urls: List[str], max_chars: int) -> Dict[str, Any]:
    return {"urls": urls, "full_content": True}


def _tavily_args(urls: List[str], max_chars: int) -> Dict[str, Any]:
    return {"urls": urls, "extract_depth": "basic", "format": "markdown"}


def _firecrawl_args(urls: List[str], max_chars: int) -> Dict[str, Any]:
    # onlyMainContent drops the cookie banner and nav chrome, which is most of what
    # a page returns otherwise: an AWS doc fetched without it comes back as nothing
    # but the consent dialog. The search path sets it too.
    return {"url": urls[0], "formats": ["markdown"], "onlyMainContent": True}


def _anysearch_args(urls: List[str], max_chars: int) -> Dict[str, Any]:
    return {"url": urls[0]}


def _rival_args(urls: List[str], max_chars: int) -> Dict[str, Any]:
    return {"operation": "retrieve", "url": urls[0], "extraction_method": "markdown"}


EXA_PROVIDER = FetchProvider("Exa", "https://mcp.exa.ai/mcp", "web_fetch_exa", _exa_args, max_batch=100)
PARALLEL_PROVIDER = FetchProvider("Parallel", "https://search.parallel.ai/mcp", "web_fetch", _parallel_args, max_batch=20)
TAVILY_PROVIDER = FetchProvider("Tavily keyless", "https://mcp.tavily.com/mcp/", "tavily_extract", _tavily_args, {"X-Tavily-Access-Mode": "keyless"}, 20)
FIRECRAWL_PROVIDER = FetchProvider("Firecrawl", "https://mcp.firecrawl.dev/v2/mcp", "firecrawl_scrape", _firecrawl_args)
ANYSEARCH_PROVIDER = FetchProvider("AnySearch", "https://api.anysearch.com/mcp", "extract", _anysearch_args, {"X-Anysearch-Client": "mcp/1.0.0"})
RIVAL_PROVIDER = FetchProvider("RivalSearchMCP", "https://rivalsearchmcp.fastmcp.app/mcp", "content_operations", _rival_args)

SINGLE_PROVIDERS = (FIRECRAWL_PROVIDER, EXA_PROVIDER, PARALLEL_PROVIDER, TAVILY_PROVIDER, ANYSEARCH_PROVIDER, RIVAL_PROVIDER)
MULTI_PROVIDERS = (EXA_PROVIDER, PARALLEL_PROVIDER, TAVILY_PROVIDER, FIRECRAWL_PROVIDER, ANYSEARCH_PROVIDER, RIVAL_PROVIDER)


# Provider execution
# ---------------------------------------------------------------------------

def _chunks(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), max(1, size)):
        yield list(values[start:start + max(1, size)])


def _run_mcp_provider(
    provider: FetchProvider,
    pending: Sequence[str],
    max_chars: int,
    timeout: float,
) -> Tuple[Dict[str, FetchPage], Optional[str], int]:
    """Fetch every pending URL from one provider, one batch of URLs per call.

    Splitting the URLs into batches the endpoint accepts, judging the extracted
    pages, and stopping the chain at the first batch that fails outright.
    """
    client = McpClient(
        provider.name,
        provider.endpoint,
        provider.tool,
        provider.args_builder,
        provider.headers,
        timeout,
        CLIENT_NAME,
    )
    pages: Dict[str, FetchPage] = {}
    note: Optional[str] = None
    for chunk in _chunks(pending, provider.max_batch):
        outcome = client.call(chunk, max_chars)
        if not outcome.ok:
            return pages, outcome.error, client.elapsed_ms
        result = outcome.result or {}
        extracted, unparsed = _extract_pages(
            result, outcome.raw_text, chunk, provider.name, max_chars
        )
        # Judged after extraction, so a page that quotes a marker it found while
        # being read is content rather than a provider reporting a failure.
        error = tool_error(result, outcome.raw_text, FETCH_POLICY, len(extracted))
        if error:
            return pages, error, client.elapsed_ms
        if not extracted and note is None:
            note = unparsed or "no extractable content"
        pages.update(extracted)
    return pages, note, client.elapsed_ms


# Secrets, CLI, and output
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch one or more URLs with provider fallback.")
    parser.add_argument("urls", nargs="*", help="HTTP or HTTPS URLs")
    parser.add_argument("--max", "--max-chars", dest="max_chars", type=int, default=DEFAULT_MAX_CHARS, help=f"maximum characters per URL (default: {DEFAULT_MAX_CHARS})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT:g})")
    parser.add_argument("--debug", action="store_true", help="write provider status diagnostics to stderr")
    return parser.parse_args(argv)


def _split_urls(values: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return every distinct argument in input order, with a reason for each rejected one.

    One malformed argument must not discard the rest of the batch. Callers typically pass a
    list assembled from search results, where a single bad entry should cost that entry and
    nothing else, so a bad URL is reported and skipped instead of failing the whole run.
    """
    ordered: List[str] = []
    rejected: Dict[str, str] = {}
    seen: Set[str] = set()
    for value in values:
        url = value.strip()
        if not url:
            continue
        reason = ""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            reason = f"invalid URL: {url}"
        else:
            key = _normal_url(url)
            if key in seen:
                continue
            seen.add(key)
        if url not in ordered:
            ordered.append(url)
        if reason:
            rejected[url] = reason
    return ordered, rejected


def _truncate(text: str, max_chars: int) -> str:
    return clamp(text.strip(), max_chars, "\n[truncated]")


def _print_debug(enabled: bool, provider: str, before: int, after: int, elapsed: int, error: Optional[str]) -> None:
    if not enabled:
        return
    detail = f" ({error})" if error else ""
    print(f"[webfetch] {provider}: {before - after}/{before} URLs; {elapsed}ms{detail}", file=sys.stderr)


def _render(
    urls: Sequence[str],
    pages: Dict[str, FetchPage],
    failures: Dict[str, List[str]],
    max_chars: int,
    skipped: Optional[Dict[str, str]] = None,
) -> None:
    for url in urls:
        print(f"=== {url} ===")
        page = pages.get(url)
        if page:
            print(_truncate(page.text, max_chars))
        elif skipped and url in skipped:
            # Never attempted, so it is not a provider failure and must not be reported as
            # one: no provider was ever asked for this URL.
            print(f"[skipped: {skipped[url]}]")
        else:
            # Report what each provider actually said. Naming only the first failure
            # misattributes the cause, since providers fail for unrelated reasons: a
            # rate-limited first provider would otherwise explain a dead domain.
            reasons = failures.get(url) or ["no provider returned content"]
            print("[fetch failed: " + "; ".join(reasons[:3]) + "]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.urls:
        print('Usage: webfetch.py "url" [url ...] [--max N]', file=sys.stderr)
        return 2
    if args.max_chars <= 0:
        print("error: --max must be greater than zero", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("error: --timeout must be greater than zero", file=sys.stderr)
        return 2
    ordered, rejected = _split_urls(args.urls)
    urls = [url for url in ordered if url not in rejected]
    if not urls:
        print("error: no valid URL to fetch", file=sys.stderr)
        for reason in rejected.values():
            print(f"  - {reason}", file=sys.stderr)
        return 2

    pages: Dict[str, FetchPage] = {}
    failures: Dict[str, List[str]] = {}
    pending = list(urls)
    providers = SINGLE_PROVIDERS if len(urls) == 1 else MULTI_PROVIDERS

    for provider in providers:
        if not pending:
            break
        before = len(pending)
        fetched, error, elapsed = _run_mcp_provider(provider, pending, args.max_chars, args.timeout)
        pages.update(fetched)
        for url in fetched:
            failures.pop(url, None)
        pending = [url for url in pending if url not in pages]
        for url in pending:
            failures.setdefault(url, []).append(f"{provider.name}: {safe_error(error or 'no content', 100)}")
        _print_debug(args.debug, provider.name, before, len(pending), elapsed, error)

    _render(ordered, pages, failures, args.max_chars, rejected)
    return 0 if pages else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        raise SystemExit(130)
