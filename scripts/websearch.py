#!/usr/bin/env python3
"""Sequential, rate-limit-aware web search with plain-text output."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from webcore import (
    CLIENT_NAME,
    DEFAULT_TIMEOUT,
    McpClient,
    Outcome,
    ResultPolicy,
    clamp,
    first_url,
    parse_json,
    safe_error,
    tool_error,
)


DEFAULT_LIMIT = 5
# Total output budget. Treated as a budget divided across results rather than a hard
# cut-off, so a large result set keeps every result instead of losing the tail.
DEFAULT_MAX_OUTPUT = 120000
# Exa serves 30 results cleanly and deterministically (measured 5/10/20/30 ->
# 27k/56k/128k/152k characters). 40 still works but 50 hangs, so 30 is the ceiling.
MAX_LIMIT = 30
# Above this result count, Firecrawl returns result metadata only; see _firecrawl_search_args.
SCRAPE_WITH_SEARCH_MAX_RESULTS = 5
RESULT_KEYS = ("results", "web", "items", "hits", "documents", "data")
# A marker is only evidence when it opens the body or a line, so these name a
# provider's own wording for a failure. "rate limit" is here because the free tiers
# answer HTTP 200 with a plain sentence saying so, which is otherwise indistinguishable
# from a result.
ERROR_MARKERS = (
    "unauthorized action", "monthly_cap_reached", "invalid api key",
    "invalid ser papi api key", "token signature failed", "search failed for query",
    "unknown tool:", "payment required", "missing required mcp-session-id",
    "valid oauth bearer token required", "you've hit", "you reached",
    "rate limit", "quota exceeded",
)


SEARCH_POLICY = ResultPolicy(
    markers=ERROR_MARKERS,
    # A payload with no result list is a status report, so a bare code in it is worth
    # surfacing: it says the query was refused, which the caller can act on.
    has_content=lambda payload: bool(_result_lists(payload)),
    failure_keys=("error", "message"),
    report_fallbacks=("error", "message"),
    code_sentinels=(None, ""),
    report_code=True,
)


@dataclass(frozen=True)
class SearchProvider:
    name: str
    endpoint: str
    tool: str
    args_builder: Callable[[str, int], Dict[str, Any]]
    headers: Dict[str, str] = field(default_factory=dict)
    max_results: int = 10


# Text rendering and error classification
# ---------------------------------------------------------------------------

def _stringify(value: Any, limit: int = 5000) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return clamp(str(value).strip(), limit, "\n[truncated]")


def _first(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _result_lists(value: Any, depth: int = 0) -> List[List[Dict[str, Any]]]:
    if depth > 4:
        return []
    if isinstance(value, list):
        return [value] if all(isinstance(item, dict) for item in value) else []
    if not isinstance(value, dict):
        return []
    found: List[List[Dict[str, Any]]] = []
    for key in RESULT_KEYS:
        child = value.get(key)
        if isinstance(child, list) and all(isinstance(item, dict) for item in child):
            found.append(child)
        elif isinstance(child, dict):
            found.extend(_result_lists(child, depth + 1))
    return found


def _render_item(item: Dict[str, Any], item_chars: int = 0) -> str:
    title = _first(item, ("title", "name", "heading", "page_title"))
    url = first_url(item)
    parts: List[str] = []
    if isinstance(title, (str, int, float)):
        parts.append(str(title))
    if isinstance(url, str) and url:
        parts.append(f"URL: {url}")
    for key, label in (
        ("description", "Description"), ("snippet", "Snippet"),
        ("summary", "Summary"), ("content", "Content"),
        ("text", "Text"), ("markdown", "Markdown"),
    ):
        rendered = _stringify(item.get(key))
        if rendered:
            parts.append(f"{label}: {rendered}")
    for key in ("excerpts", "highlights", "snippets"):
        values = item.get(key)
        if isinstance(values, list):
            entries = [_stringify(value, 1200) for value in values]
            entries = [entry for entry in entries if entry]
            if entries:
                parts.append("\n".join(f"- {entry}" for entry in entries))
    return clamp(
        "\n".join(parts) if parts else _stringify(item, 3000),
        item_chars,
        "\n[result truncated]",
        rstrip=True,
    )


def _flatten_results(lists: Sequence[Sequence[Dict[str, Any]]], max_items: int) -> List[Dict[str, Any]]:
    """Order every result a provider returned into one list and cap the total.

    The cap is applied here rather than left to the request because providers disagree on
    both the envelope key and whether the count is honoured at all: Parallel and
    RivalSearchMCP ignore it, and some providers repeat the same hits under two keys. The
    output budget is divided by the requested count, so an over-delivering provider would
    otherwise overrun the budget and lose its tail to the global cut-off. Results are
    de-duplicated by URL, falling back to the item itself when a hit has none, so the same
    page listed twice cannot consume two slots of the budget.
    """
    items: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for group in lists:
        for item in group:
            url = first_url(item)
            key = url or json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    return items[:max_items] if max_items > 0 else items


def _render_payload(value: Any, item_chars: int = 0, max_items: int = 0) -> Tuple[int, str]:
    """Render a payload, and say how many results it produced.

    The count is what tells a result apart from a status message, so it is returned
    alongside the text rather than recounted from it.
    """
    if isinstance(value, str):
        parsed = parse_json(value)
        return _render_payload(parsed, item_chars, max_items) if parsed is not None else (0, value.strip())
    lists = _result_lists(value)
    if lists:
        if all(not items for items in lists):
            return 0, ""
        items = _flatten_results(lists, max_items)
        if not items:
            return 0, ""
        return len(items), "\n\n".join(
            f"{index}. {_render_item(item, item_chars)}" for index, item in enumerate(items, 1)
        )
    return 0, _stringify(value, 8000)


_RECORD_SEPARATOR = re.compile(r"^---\s*$")
_RECORD_START = ("Title:", "URL:")


def _split_records(text: str) -> List[str]:
    """Split a pre-rendered plain-text response into per-result records.

    Exa returns formatted text rather than JSON. Applying the output budget to that
    whole blob would truncate the response and discard every result after the first
    few, so records are separated first and capped individually.
    """
    lines = (text or "").split("\n")
    records: List[str] = []
    current: List[str] = []
    for index, line in enumerate(lines):
        if _RECORD_SEPARATOR.match(line):
            following = next((later for later in lines[index + 1:] if later.strip()), "")
            if following.startswith(_RECORD_START):
                if current:
                    records.append("\n".join(current).strip("\n"))
                current = []
                continue
        current.append(line)
    if current:
        records.append("\n".join(current).strip("\n"))
    return [record for record in records if record.strip()]


def _render_result(result: Dict[str, Any], raw_text: str, item_chars: int = 0, max_items: int = 0) -> Tuple[int, str]:
    """Render a provider's answer, and say how many results it produced.

    A provider that returns prose rather than results counts as zero, which is what
    lets a status message be told from a result: only a body with nothing in it has a
    marker to be evidence of anything.
    """
    if raw_text:
        parsed = parse_json(raw_text)
        if parsed is not None:
            found, text = _render_payload(parsed, item_chars, max_items)
            return found, text.strip()
        records = _split_records(raw_text)
        if len(records) > 1:
            if max_items > 0:
                records = records[:max_items]
            if item_chars > 0:
                records = [
                    clamp(record, item_chars, "\n[result truncated]", rstrip=True)
                    for record in records
                ]
            return len(records), "\n\n---\n\n".join(records)
        return 0, raw_text.strip()
    structured = result.get("structuredContent")
    if structured is None:
        return 0, ""
    found, text = _render_payload(structured, item_chars, max_items)
    return found, text.strip()


# Provider calls and fallback
# ---------------------------------------------------------------------------

def call_provider(
    provider: SearchProvider,
    query: str,
    limit: int,
    timeout: float,
    item_chars: int = 0,
    max_items: int = 0,
) -> Outcome:
    """Run one query against one provider and render the result.

    Capping the count to what the provider serves, rendering the answer, and judging
    whether the rendered text is usable.
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
    outcome = client.call(query, max(1, min(limit, provider.max_results)))
    if not outcome.ok:
        return outcome
    result = outcome.result or {}
    found, text = _render_result(result, outcome.raw_text, item_chars, max_items)
    return Outcome(
        outcome.provider,
        outcome.status,
        outcome.elapsed_ms,
        result=result,
        raw_text=outcome.raw_text,
        text=text,
        error=tool_error(result, outcome.raw_text or text, SEARCH_POLICY, found),
    )


# Provider registry
# ---------------------------------------------------------------------------

def _query_limit(key: str) -> Callable[[str, int], Dict[str, Any]]:
    return lambda query, limit: {"query": query, key: limit}


def _parallel_args(query: str, limit: int) -> Dict[str, Any]:
    return {"objective": query, "search_queries": [query]}


def _tavily_args(query: str, limit: int) -> Dict[str, Any]:
    return {"query": query, "max_results": limit, "search_depth": "basic"}


def _query_only(query: str, limit: int) -> Dict[str, Any]:
    return {"query": query}


def _firecrawl_search_args(query: str, limit: int) -> Dict[str, Any]:
    """Ask Firecrawl to return page content alongside the result list.

    Measured on the same query: a plain search returns roughly 6.4k characters of
    title/description, while adding scrapeOptions returns roughly 29k because each hit
    carries its page body. That is a large quality gain for one extra parameter, but it
    costs latency and tokens, so it is applied only for narrow result sets.
    """
    args: Dict[str, Any] = {"query": query, "limit": limit}
    if limit <= SCRAPE_WITH_SEARCH_MAX_RESULTS:
        args["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}
    return args


SEARCH_PROVIDERS: Tuple[SearchProvider, ...] = (
    SearchProvider(
        "Exa",
        "https://mcp.exa.ai/mcp",
        "web_search_exa",
        _query_limit("numResults"),
        max_results=30,
    ),
    SearchProvider(
        "Parallel",
        "https://search.parallel.ai/mcp",
        "web_search",
        _parallel_args,
    ),
    SearchProvider(
        "Firecrawl",
        "https://mcp.firecrawl.dev/v2/mcp",
        "firecrawl_search",
        _firecrawl_search_args,
    ),
    SearchProvider(
        "Tavily keyless",
        "https://mcp.tavily.com/mcp/",
        "tavily_search",
        _tavily_args,
        {"X-Tavily-Access-Mode": "keyless"},
        20,
    ),
    SearchProvider(
        "AnySearch anonymous",
        "https://api.anysearch.com/mcp",
        "search",
        _query_limit("max_results"),
        {"X-Anysearch-Client": "mcp/1.0.0"},
    ),
    SearchProvider(
        "RivalSearchMCP",
        "https://rivalsearchmcp.fastmcp.app/mcp",
        "web_search",
        _query_only,
    ),
)


# Vertical search modes. Each flag selects one specialised corpus that general web search
# does not cover, and each is served by a single tool on a single endpoint: no other provider
# declares a vertical-search tool, so there is nothing to rotate to when it is unavailable.
# They are deliberately outside SEARCH_PROVIDERS, which is a rotation for queries every
# provider can answer.
VERTICAL_ENDPOINT = "https://rivalsearchmcp.fastmcp.app/mcp"
VERTICAL_MAX_RESULTS = 25


@dataclass(frozen=True)
class VerticalMode:
    flag: str
    tool: str
    args_builder: Callable[[str, int], Dict[str, Any]]
    label: str


def _vertical_args(extra: Dict[str, Any]) -> Callable[[str, int], Dict[str, Any]]:
    """Build a vertical tool's arguments, letting the caller's result count through."""
    def build(query: str, limit: int) -> Dict[str, Any]:
        return {"query": query, "max_results": limit, **extra}
    return build


VERTICAL_MODES: Tuple[VerticalMode, ...] = (
    VerticalMode("--github", "github_search", _vertical_args({}), "GitHub repository search"),
    VerticalMode("--papers", "scientific_research", _vertical_args({"operation": "academic_search"}), "academic paper search"),
)

# Vertical reports carry a "Next Steps" block naming follow-up tools, several of which the
# endpoint does not declare, so following the advice yields a tool-not-found error. It is also
# provider output instructing the caller, which this skill otherwise treats as untrusted data.
# The block sits near the top of the report, directly under the summary line, so the pattern
# takes the marker and the bullet list beneath it and nothing else: results follow after a
# separator, and the run stops at the first line that is not a list item.
_ADVICE_MARKER = re.compile(
    r"\n?[^\n]*\*\*Next Steps:\*\*[^\n]*(?:\n[ \t]*[-*•][^\n]*)*", re.IGNORECASE
)


def strip_advice(text: str) -> str:
    cleaned = _ADVICE_MARKER.sub("", text or "")
    # Removing the block leaves the blank lines around it adjacent, and leaves the separator
    # that followed it doubled against the one before it. Single separators, which is how
    # results are delimited, are left alone.
    cleaned = re.sub(r"---\s*\n\s*---", "---", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def vertical_provider(mode: VerticalMode) -> SearchProvider:
    return SearchProvider(
        mode.label, VERTICAL_ENDPOINT, mode.tool, mode.args_builder, max_results=VERTICAL_MAX_RESULTS
    )


# CLI and entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search through keyless providers, falling back on empty results, errors, and rate limits.",
    )
    parser.add_argument("query", help="search query")
    parser.add_argument(
        "num_results_positional",
        nargs="?",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="NUM_RESULTS",
        help=f"number of results, up to {MAX_LIMIT} (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "-n",
        "--limit",
        "--num-results",
        dest="limit",
        type=int,
        default=None,
        help="override the number of results",
    )
    vertical = parser.add_argument_group(
        "vertical search",
        "Search one specialised corpus instead of the web. Served by a single provider, so "
        "unlike a general query these do not rotate when it is unavailable.",
    )
    vertical_modes = vertical.add_mutually_exclusive_group()
    for mode in VERTICAL_MODES:
        vertical_modes.add_argument(
            mode.flag,
            dest="vertical",
            action="store_const",
            const=mode.flag.lstrip("-"),
            help=f"{mode.label} instead of a general web search",
        )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_OUTPUT,
        help=f"total output budget in characters, divided across results (default: {DEFAULT_MAX_OUTPUT})",
    )
    parser.add_argument("--debug", action="store_true", help="write provider status diagnostics to stderr")
    return parser.parse_args(argv)


def _debug_log(enabled: bool, attempt: Outcome) -> None:
    if enabled:
        state = "ok" if attempt.ok else "failed"
        detail = f" ({attempt.error})" if attempt.error else ""
        print(
            f"[websearch] {attempt.provider}: {state}; HTTP {attempt.status}; "
            f"{attempt.elapsed_ms}ms{detail}",
            file=sys.stderr,
        )


def _output_text(text: str, max_chars: int) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "No results found."
    return clamp(cleaned, max_chars, "\n[output truncated]")


# A vertical report states its own count in a bolded summary line, so a report
# claiming zero hits is a definite answer rather than an absent one, and reads as
# success to a caller. Anchored to the bolded form, which is where the count goes,
# so a count quoted in a result title does not match.
_VERTICAL_NO_HITS = re.compile(r"(?im)^\*\*found 0\s+(?:repositories|repos|papers)\b")

# A vertical report is a markdown document: a summary, then one `## N. title` section
# per hit. The endpoint answers with three times the count it was asked for, so the
# caller's number has to be enforced on the way out or a request for 25 emits 75.
_VERTICAL_ITEM = re.compile(r"(?m)^##[ \t]+\d+\.[ \t]")
_VERTICAL_COUNT = re.compile(r"(?im)^(\*\*found\s+)\d+(\s+\w+\*\*)")


def _clamp_vertical(text: str, limit: int) -> str:
    """Cut a vertical report to the requested number of hits.

    The summary count is corrected along with the items, because a header that
    claims more hits than the report holds is the one number a reader is likely to
    trust without checking.
    """
    if limit <= 0:
        return text
    starts = [match.start() for match in _VERTICAL_ITEM.finditer(text)]
    if len(starts) <= limit:
        return text
    kept = re.sub(r"[ \t]*-{3,}[ \t]*$", "", text[: starts[limit]].rstrip()).rstrip()
    return _VERTICAL_COUNT.sub(lambda m: f"{m.group(1)}{limit}{m.group(2)}", kept, count=1)


def run_vertical(mode: VerticalMode, query: str, limit: int, args: argparse.Namespace) -> int:
    """Run one vertical search.

    The result is a single provider-formatted report rather than the rendered
    title/URL/passage form a general query produces, so the count is enforced on the
    report and the character budget, not a per-item cap, bounds the output.
    """
    attempt = call_provider(vertical_provider(mode), query, limit, args.timeout)
    _debug_log(args.debug, attempt)
    if not attempt.ok:
        print(f"error: {mode.label} failed", file=sys.stderr)
        print(f"  - {safe_error(attempt.error or 'failed')}", file=sys.stderr)
        return 1
    text = _clamp_vertical(strip_advice(attempt.text), limit)
    if _VERTICAL_NO_HITS.search(text):
        print(f"error: no {mode.label} results", file=sys.stderr)
        return 1
    if not text.strip():
        print(f"error: no {mode.label} results", file=sys.stderr)
        return 1
    print(_output_text(text, args.max_chars))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.query.strip():
        print("error: query must not be empty", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("error: --timeout must be greater than zero", file=sys.stderr)
        return 2
    if args.max_chars < 0:
        print("error: --max-chars must not be negative", file=sys.stderr)
        return 2
    limit = args.limit if args.limit is not None else args.num_results_positional
    if limit <= 0:
        print("error: number of results must be greater than zero", file=sys.stderr)
        return 2

    query = args.query.strip()
    limit = min(limit, MAX_LIMIT)
    if args.vertical:
        mode = next(mode for mode in VERTICAL_MODES if mode.flag.lstrip("-") == args.vertical)
        return run_vertical(mode, query, min(limit, VERTICAL_MAX_RESULTS), args)
    # Split the output budget across results so that every requested result survives.
    # A global cut-off would silently drop the tail of a large result set, which is
    # where the least-similar and often most specific sources live. This holds only while
    # no more than `limit` results are rendered, which is why the count is enforced
    # downstream in _flatten_results rather than assumed from the request.
    item_chars = args.max_chars // limit if args.max_chars > 0 else 0
    # Tracks whether every provider answered successfully with an empty result set.
    # When that is the case the query itself has no matches, which is worth saying
    # differently from a broken search path: an agent can retry or rephrase the
    # former, but has to report the latter.
    all_empty = True
    failures: List[str] = []
    for provider in SEARCH_PROVIDERS:
        attempt = call_provider(
            provider, query, limit, args.timeout, item_chars=item_chars, max_items=limit
        )
        _debug_log(args.debug, attempt)
        if attempt.ok and attempt.text.strip():
            print(_output_text(attempt.text, args.max_chars))
            return 0
        if attempt.ok:
            failures.append(f"{attempt.provider}: no results")
        else:
            all_empty = False
            failures.append(f"{attempt.provider}: {attempt.error or 'failed'}")
        if attempt.raw_429 and args.debug:
            print(f"[websearch] {attempt.provider}: rate limited; trying next provider", file=sys.stderr)

    # Distinguish "the query matched nothing" from "the search path broke": an agent can
    # retry or rephrase the former, but needs to report the latter.
    if all_empty:
        print("error: no search results", file=sys.stderr)
    else:
        print("error: all search providers failed", file=sys.stderr)
    for failure in failures:
        print(f"  - {safe_error(failure)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        raise SystemExit(130)
