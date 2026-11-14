#!/usr/bin/env python3
"""
Context retrieval from a Redis Iris Context Gateway (the context surface).

For a user question, the model picks the Gateway tools that can answer it, the tools are
called, and their results are returned as text for the LLM prompt. This is the logic of the
standalone Context Retriever CLI, returning text instead of printing panels.

- A `filter_{entity}` tool (structured constraints) is always paired with its
  `search_{entity}_by_text` partner (free-text search) when the Gateway lists one; the two
  result sets are intersected. With nothing in common the filter's records are returned.
- A search-mode Gateway lists only `search_tools` and `run_tool`, so the real tools are found
  with a wildcard search.

Settings (see .env): CONTEXT_GATEWAY_URL, the Gateway's MCP endpoint, IRIS_API_KEY, and
LLM_TOOL_SELECTOR_PROMPT, the file holding the tool selector's system prompt.
When CONTEXT_GATEWAY_URL is empty the retriever is disabled and returns no context.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from redis_iris import ContextGateway
from redis_iris.context_gateway.values import GatewayTool, ToolResult
from redis_iris.errors import ConfigurationError, ToolError

import utils

logger = logging.getLogger(__name__)

DEFAULT_SELECTOR_PROMPT_FILE = "llm-prompt-tool-selector"  # override with LLM_TOOL_SELECTOR_PROMPT

DEFAULT_ACCOUNT_PATTERN = r"ACC\d{6}"  # override with CONTEXT_UID_PATTERN
DEFAULT_ACCOUNT_FIELD = "account_num"  # the Gateway field holding the account; override with CONTEXT_ACCOUNT_FIELD

PERSONAL_PROMPT = """Decide whether the user's message is about the user's own information or about any entity related to the user: their account, positions, holdings, balances, orders, or sign-in and access problems, typically phrased with "my", "me" or "I". General, company-wide or policy questions are not personal. The conversation is given only to resolve references."""
PERSONAL_TOOL = {
    "type": "function",
    "function": {
        "name": "classify",
        "description": "Record whether the message is about the user's own information.",
        "parameters": {
            "type": "object",
            "properties": {"personal": {"type": "boolean"}},
            "required": ["personal"],
        },
    },
}

SEARCH_TOOLS, RUN_TOOL = "search_tools", "run_tool"  # a search-mode Gateway lists only these
FILTER_RE = re.compile(r"^(?P<prefix>.*?__)?filter_(?P<entity>.+)$")
TEXT_SEARCH_RE = re.compile(r"^(?:.*?__)?search_.+_by_text$")
AGGREGATE_PREFIXES = ("summarize_", "count_")  # tools that run over every record and take no filter
ENTITY_TOOL_RE = re.compile(r"^(?:filter|list|count|summarize|search)_(?P<entity>.+?)(?:_by_text)?$")
TEXT_FIELDS_RE = re.compile(r"TEXT-indexed fields:\s*(.*)", re.IGNORECASE | re.DOTALL)
NO_TEXT_FIELDS_RE = re.compile(r"\b(?:no|without)\s+TEXT-indexed fields\b", re.IGNORECASE)
ID_FIELDS = ("id", "_id", "key", "doc_id")
PAGE_SIZE = 100  # server maximum per call
MAX_PAGES = 10  # safety cap: at most 1000 records per tool
MAX_TOOLS = 100  # Gateway search limit; OpenAI accepts up to 128 tools
MAX_CONTEXT_CHARS = 8000  # cap on the context text added to the LLM prompt
DEFAULT_SELECTOR_MODEL = "gpt-4o-mini"  # tool selection is a small task; override with CONTEXT_GATEWAY_MODEL
HISTORY_MESSAGES = 6  # most recent conversation messages shown to the tool selector
HISTORY_MESSAGE_CHARS = 500  # per message
MEMORY_CHARS = 2000  # cap on the long-term memory / summary text shown to the tool selector

# OpenAI rejects these at the top level of a function's parameter schema.
UNSUPPORTED_TOP_LEVEL = ("oneOf", "anyOf", "allOf", "enum", "const", "not")


def load_selector_prompt(name: str) -> str:
    """The tool selector's system prompt, read from a file next to this module (or an absolute
    path). <prompt>file</prompt> tags in it are replaced with that file's content."""
    return utils.load_prompt(Path(__file__).parent / name)


def to_openai_tools(tools: list[GatewayTool]) -> list[dict[str, Any]]:
    """Convert Gateway tool definitions into OpenAI function-calling tool specs.

    Top-level keywords OpenAI refuses are removed from the schema and restated in the
    description so the model still sees the constraint (the Gateway enforces it anyway)."""
    specs = []
    for t in tools:
        schema = dict(t.input_schema or {"type": "object", "properties": {}})
        description = t.description
        for key in UNSUPPORTED_TOP_LEVEL:
            if key in schema:
                description += f"\nArgument constraint ({key}): {json.dumps(schema.pop(key))}"
        specs.append(
            {"type": "function", "function": {"name": t.name, "description": description, "parameters": schema}}
        )
    return specs


def render_result(result: ToolResult) -> str:
    """Printable text for a tool result."""
    if result.data is not None:
        return json.dumps(result.data, indent=2, default=str)
    return result.text


@dataclass
class Retrieval:
    """What the Gateway step produced for one message.

    context: text for the LLM prompt ("" when nothing was fetched).
    needs_account: the question is about the user's own data and no account number is known, so
        the LLM should ask for it instead of answering.
    account: the account number a personal question was restricted to, else None.
    """

    context: str = ""
    needs_account: bool = False
    account: str | None = None

    @property
    def cacheable(self) -> bool:
        """An answer that asked for the account number, or is about one account, must not be
        reused for a later, similar question."""
        return not self.needs_account and self.account is None


def find_account(
    question: str, history: list[dict[str, Any]] | None, pattern: re.Pattern[str]
) -> str | None:
    """The user's account number: the latest one the user typed (this question first), else one
    in the recalled memories or session summary. Assistant messages are ignored, because they
    may quote other customers' accounts."""
    history = history or []
    sources = [question]
    sources += [str(m.get("content", "")) for m in reversed(history) if m.get("role") == "user"]
    sources += [str(m.get("content", "")) for m in history if m.get("role") == "system"]
    for text in sources:
        found = [m.group(0) for m in pattern.finditer(text)]
        if found:
            return found[-1].upper()
    return None


def clean_args(tool: GatewayTool | None, args: dict[str, Any]) -> dict[str, Any]:
    """Drop optional arguments the tool would reject, so one bad guess does not lose the whole call.

    The model sometimes invents an argument or a value outside an enum, for example sort_by="sector"
    when "group it by sector" was asked (sort_by only takes the sortable numeric fields). The
    Gateway then refuses the call and nothing is returned. A required argument is never dropped."""
    if tool is None:
        return args
    schema = tool.input_schema or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    cleaned = {}
    for key, value in args.items():
        spec = props.get(key)
        if spec is None and props:
            reason = "is not a parameter"
        elif spec and spec.get("enum") and value not in spec["enum"]:
            reason = f"must be one of {spec['enum']}"
        else:
            cleaned[key] = value
            continue
        if key in required:
            cleaned[key] = value
        else:
            logger.info("Dropping argument %s=%r of %s: it %s", key, value, tool.name, reason)
    return cleaned


def filters_by_account(tool: GatewayTool | None, field: str) -> bool:
    """True when the tool takes tag_conditions and its description lists the account field."""
    if tool is None:
        return False
    props = (tool.input_schema or {}).get("properties") or {}
    return "tag_conditions" in props and f"{field}:" in (tool.description or "")


def scope_calls(
    calls: list[tuple[str, dict[str, Any]]],
    account: str,
    tools_by_name: dict[str, GatewayTool],
    field: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Restrict every call to one account, whatever the model asked for.

    - A tool that can filter by the account field gets that condition set (replacing any account
      condition the model wrote).
    - list_{entity} becomes filter_{entity}, which can filter.
    - An entity that is keyed by the account number (its get_{entity}_by_id takes an account
      number, like a customer profile) cannot be filtered by account, but its one record can be
      fetched: any filter, search, list or count on it becomes get_{entity}_by_id(account).
    - summarize_{entity} and count_{entity} take no filter and cover every customer, so they are
      replaced by the user's own filter_{entity} records.
    - get_{entity}_by_id is kept only when the id is the user's own account number.
    - A search_{entity}_by_text call is kept only together with a scoped filter_{entity} call, so
      its results are intersected with that account's records.
    - Any other call cannot be scoped, so it is dropped rather than risk another customer's data."""

    def scoped(args: dict[str, Any]) -> dict[str, Any]:
        others = [c for c in args.get("tag_conditions") or [] if c.get("field") != field]
        return {**args, "tag_conditions": [*others, {"field": field, "value": account}]}

    def own_record(prefix: str, entity: str | None) -> tuple[str, dict[str, Any]] | None:
        """get_{entity}_by_id(account), when that entity is keyed by the account number."""
        tool = tools_by_name.get(f"{prefix}get_{entity}_by_id") if entity else None
        props = ((tool.input_schema or {}).get("properties") or {}) if tool else {}
        text = " ".join([tool.description or "", *(str(p.get("description", "")) for p in props.values())]) if tool else ""
        if tool is None or not props or "account number" not in text.lower():
            return None
        id_param = "id" if "id" in props else next(iter(props))
        return tool.name, {id_param: account}

    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def add(call: tuple[str, dict[str, Any]]) -> None:
        if call[0] not in seen:
            seen.add(call[0])
            out.append(call)

    for name, args in calls:
        prefix, _, base = name.rpartition("__")
        prefix = f"{prefix}__" if prefix else ""
        m = ENTITY_TOOL_RE.match(base)
        entity = m.group("entity") if m else None

        if TEXT_SEARCH_RE.match(name):
            filt = f"{prefix}filter_{entity}"
            if filters_by_account(tools_by_name.get(filt), field):
                if filt not in {n for n, _ in calls}:
                    add((filt, scoped({})))
                out.append((name, args))
            elif own := own_record(prefix, entity):
                add(own)
            continue

        if base.startswith("get_") and base.endswith("_by_id"):
            # Looking up one record by its id is safe when the id is the user's own account number.
            if any(str(v).strip().upper() == account.upper() for v in args.values()):
                add((name, args))
            else:
                logger.info("Dropping %s: it is not a lookup of the user's own account", name)
            continue

        target = name
        if base.startswith("list_") and filters_by_account(tools_by_name.get(f"{prefix}filter_{base[5:]}"), field):
            target = f"{prefix}filter_{base[5:]}"
            args = {k: v for k, v in args.items() if k in ("limit", "offset", "sort_by", "sort_order")}
        if filters_by_account(tools_by_name.get(target), field):
            add((target, scoped(args)))
        elif own := own_record(prefix, entity):
            add(own)
        elif base.startswith(AGGREGATE_PREFIXES) and filters_by_account(tools_by_name.get(f"{prefix}filter_{entity}"), field):
            # summarize_* / count_* take no filter and run over every customer's records. Fetch the
            # user's own records instead, and let the answer do the grouping, summing or counting.
            logger.info("Replacing %s with the user's own %s records", name, entity)
            add((f"{prefix}filter_{entity}", scoped({})))
        else:
            logger.info("Dropping %s: it cannot be restricted to the user's account", name)
    return out


def selector_input(
    question: str, history: list[dict[str, Any]] | None, account: str | None = None
) -> str:
    """The user message for the tool selector: the question, preceded by what is known about
    the user and the recent conversation when there is any.

    history is the memory prompt: system messages (recalled long-term memories, a session
    summary) and the transcript turns. The current question is usually the transcript's last
    message because it was recorded before retrieval; it is left out so it appears only once."""
    history = list(history or [])
    system_text = "\n".join(str(m.get("content", "")) for m in history if m.get("role") == "system")
    turns = [m for m in history if m.get("role") in ("user", "assistant")]
    if turns and turns[-1].get("role") == "user" and str(turns[-1].get("content", "")).strip() == question.strip():
        turns.pop()

    parts = []
    if account:
        parts.append(f"The user's account number: {account}")
    if system_text.strip():
        parts.append("Known about the user:\n" + system_text.strip()[:MEMORY_CHARS])
    if turns:
        lines = [f"{m['role']}: {str(m.get('content', ''))[:HISTORY_MESSAGE_CHARS]}" for m in turns[-HISTORY_MESSAGES:]]
        parts.append("Recent conversation:\n" + "\n".join(lines))
    if not parts:
        return question
    return "\n\n".join(parts) + f"\n\nCurrent question: {question}"


def has_text_fields(tool: GatewayTool) -> bool:
    """False when a search_{entity}_by_text tool says its entity has no TEXT-indexed fields.

    The Gateway describes the searchable fields as "... over only these TEXT-indexed fields:
    name: ...". An empty list, or a statement that there are none, means a text search can
    never match, so the tool is not offered. A description that does not mention TEXT-indexed
    fields at all is taken to be searchable (unknown means keep)."""
    description = tool.description or ""
    if NO_TEXT_FIELDS_RE.search(description):
        return False
    m = TEXT_FIELDS_RE.search(description)
    return m is None or bool(m.group(1).strip(" .;:"))


def search_partner(name: str, tools_by_name: dict[str, Any]) -> str | None:
    """[prefix__]filter_{entity} -> [prefix__]search_{entity}_by_text, if the Gateway lists it."""
    m = FILTER_RE.match(name)
    if not m:
        return None
    partner = f"{m.group('prefix') or ''}search_{m.group('entity')}_by_text"
    return partner if partner in tools_by_name else None


def text_args(tool: GatewayTool, question: str) -> dict[str, Any]:
    """Build arguments for a text-search tool from the user's question."""
    schema = tool.input_schema or {}
    props = schema.get("properties") or {}
    if "query" in props:
        return {"query": question}
    strings = [p for p, s in props.items() if s.get("type") == "string"]
    required = [p for p in strings if p in (schema.get("required") or [])]
    target = (required or strings or ["query"])[0]
    return {target: question}


def parse_records(result: ToolResult) -> list[Any] | None:
    """Extract the list of records from a tool result, or None if it has none."""
    data = result.data
    if result.is_error or data is None:
        return None
    if isinstance(data, dict):
        found = data.get("results")
        data = found if isinstance(found, list) else next(
            (v for v in data.values() if isinstance(v, list)), [data] if data else []
        )
    return data if isinstance(data, list) else None


def record_key(record: Any) -> str:
    """Identity of a record: its id-like field, else its canonical JSON."""
    if isinstance(record, dict):
        for f in ID_FIELDS:
            if record.get(f) is not None:
                return f"{f}:{record[f]}"
    return json.dumps(record, sort_keys=True, default=str)


class ContextRetriever:
    """Fetches context for a question from the Context Gateway."""

    def __init__(
        self,
        gateway: ContextGateway | None,
        llm: AsyncOpenAI | None,
        model: str = DEFAULT_SELECTOR_MODEL,
        account_pattern: str = DEFAULT_ACCOUNT_PATTERN,
        account_field: str = DEFAULT_ACCOUNT_FIELD,
        selector_prompt: str = "",
    ):
        self.selector_prompt = selector_prompt
        self.gateway = gateway
        self.llm = llm
        self.model = model
        self.account_re = re.compile(account_pattern, re.IGNORECASE)
        self.account_field = account_field
        self._tools: list[GatewayTool] | None = None
        self._openai_tools: list[dict[str, Any]] = []
        self._tools_by_name: dict[str, GatewayTool] = {}
        self._load_lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "ContextRetriever":
        """Build from .env. Without CONTEXT_GATEWAY_URL the retriever is disabled."""
        url = utils.resolve_env("CONTEXT_GATEWAY_URL")
        if not url:
            logger.info("CONTEXT_GATEWAY_URL is not set; context retrieval is disabled")
            return cls(None, None)
        gateway = ContextGateway.from_env(
            {
                "IRIS_CONTEXT_GATEWAY_URL": url,
                "IRIS_CONTEXT_GATEWAY_API_KEY": utils.resolve_env("IRIS_API_KEY"),
            }
        )
        model = utils.resolve_env("CONTEXT_GATEWAY_MODEL") or DEFAULT_SELECTOR_MODEL
        pattern = utils.resolve_env("CONTEXT_UID_PATTERN") or DEFAULT_ACCOUNT_PATTERN
        field = utils.resolve_env("CONTEXT_ACCOUNT_FIELD") or DEFAULT_ACCOUNT_FIELD
        prompt = load_selector_prompt(utils.resolve_env("LLM_TOOL_SELECTOR_PROMPT") or DEFAULT_SELECTOR_PROMPT_FILE)
        return cls(gateway, AsyncOpenAI(api_key=utils.resolve_env("OPENAI_API_KEY"), max_retries=4), model, pattern, field, prompt)

    @property
    def enabled(self) -> bool:
        return self.gateway is not None

    # ---- tools ----

    async def _load_tools(self) -> None:
        """Every tool the Gateway offers, loaded once. A search-mode Gateway lists only
        search_tools and run_tool, so the real tools are found with a wildcard search."""
        async with self._load_lock:
            if self._tools is not None:
                return
            tools = await utils.aretry(self.gateway.atools, "Context Gateway: list tools")
            if {SEARCH_TOOLS, RUN_TOOL} <= {t.name for t in tools}:
                tools = await utils.aretry(
                    lambda: self.gateway.asearch("*", limit=MAX_TOOLS), "Context Gateway: list tools"
                )
                if len(tools) >= MAX_TOOLS:
                    logger.warning("Only the first %d Context Gateway tools were loaded", MAX_TOOLS)
            # An entity with no TEXT-indexed fields cannot be text-searched: do not offer its
            # search_{entity}_by_text tool, so it is neither chosen nor paired with a filter.
            searchless = [t.name for t in tools if TEXT_SEARCH_RE.match(t.name) and not has_text_fields(t)]
            if searchless:
                logger.info("Skipping text search for entities without TEXT fields: %s", ", ".join(searchless))
                tools = [t for t in tools if t.name not in searchless]
            self._tools = tools
            self._openai_tools = to_openai_tools(tools)
            self._tools_by_name = {t.name: t for t in tools}
            logger.info("Loaded %d Context Gateway tools", len(tools))

    async def _run_tool(self, name: str, args: dict[str, Any]) -> ToolResult | None:
        try:
            args = clean_args(self._tools_by_name.get(name), args)
            return await utils.aretry(lambda: self.gateway.acall(name, args), f"Context Gateway: {name}")
        except (ConfigurationError, ToolError) as e:
            logger.warning("Context Gateway tool call failed (%s): %s", name, e)
            return None

    async def _fetch_all(
        self, name: str, args: dict[str, Any]
    ) -> tuple[list[Any] | None, ToolResult | None]:
        """Call a tool repeatedly (limit/offset paging) until has_more is false.

        Returns (records, last_result). records is None when the call failed or the
        result has no record list; last_result is then what to show instead."""
        combined: list[Any] = []
        result: ToolResult | None = None
        for page in range(MAX_PAGES):
            result = await self._run_tool(name, {**args, "limit": PAGE_SIZE, "offset": page * PAGE_SIZE})
            if result is None:
                return (combined or None), None
            records = parse_records(result)
            if records is None:
                return None, result
            combined.extend(records)
            if not (isinstance(result.data, dict) and result.data.get("has_more")):
                break
        return combined, result

    # ---- tool selection ----

    async def _select_tools(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        account: str | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        prompt = selector_input(question, history, account)
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.selector_prompt},
                {"role": "user", "content": prompt},
            ],
            tools=self._openai_tools,
            tool_choice="auto",
        )
        calls: list[tuple[str, dict[str, Any]]] = []
        for call in response.choices[0].message.tool_calls or []:
            try:
                calls.append((call.function.name, json.loads(call.function.arguments or "{}")))
            except json.JSONDecodeError:
                logger.warning("LLM returned invalid arguments for %s: %s", call.function.name, call.function.arguments)
        return await self._complete_pairs(prompt, calls)

    async def _is_personal(self, question: str, history: list[dict[str, Any]] | None) -> bool:
        """Whether the question is about the user's own data. A forced yes/no call, so the model
        cannot skip the decision; on any doubt or failure the answer is yes (ask for the account,
        never risk showing another customer's data)."""
        try:
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PERSONAL_PROMPT},
                    {"role": "user", "content": selector_input(question, history)},
                ],
                tools=[PERSONAL_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify"}},
            )
            call = response.choices[0].message.tool_calls[0]
            return bool(json.loads(call.function.arguments or "{}").get("personal", True))
        except Exception:
            logger.exception("Could not classify the question as personal; treating it as personal")
            return True

    async def _complete_pairs(
        self, question: str, calls: list[tuple[str, dict[str, Any]]]
    ) -> list[tuple[str, dict[str, Any]]]:
        """If the LLM picked only one half of a filter_{e}/search_{e}_by_text pair,
        make a forced follow-up call so the other half gets proper arguments."""
        tools_by_name = self._tools_by_name
        chosen = {n for n, _ in calls}
        for filt in [n for n in tools_by_name if FILTER_RE.match(n)]:
            search = search_partner(filt, tools_by_name)
            if not search or (filt in chosen) == (search in chosen) or not (chosen & {filt, search}):
                continue
            missing = search if filt in chosen else filt
            role = (
                "the descriptive / thematic / free-text part of the question (topics, plot, mood, keywords), "
                "as a short search phrase"
                if missing == search
                else "the structured constraints in the question (people, dates, ranges, categories, numbers)"
            )
            resp = await self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.selector_prompt},
                    {
                        "role": "user",
                        "content": f"{question}\n\nCall {missing} using only {role}. Leave out anything that "
                        "belongs to the other tool.",
                    },
                ],
                tools=to_openai_tools([tools_by_name[missing]]),
                tool_choice={"type": "function", "function": {"name": missing}},
            )
            for call in resp.choices[0].message.tool_calls or []:
                try:
                    calls.append((call.function.name, json.loads(call.function.arguments or "{}")))
                except json.JSONDecodeError:
                    pass
        return calls

    # ---- retrieval ----

    async def retrieve(self, question: str, history: list[dict[str, Any]] | None = None) -> Retrieval:
        """Context for the question: empty when disabled, nothing is relevant, or the Gateway fails.

        history (the memory prompt: recalled memories and the session transcript) helps the tool
        selector resolve follow-up questions and supplies the user's account number; it is never
        sent to the Gateway. When the question is about the user's own data and no account number
        is known, needs_account is set and nothing is fetched.

        A Gateway outage must not stop the assistant from answering, so every failure is logged
        and returns an empty Retrieval."""
        if not self.enabled:
            return Retrieval()
        try:
            return await self._retrieve(question, history)
        except Exception:
            logger.exception("Context retrieval failed; answering without it")
            return Retrieval()

    async def _retrieve(self, question: str, history: list[dict[str, Any]] | None = None) -> Retrieval:
        await self._load_tools()
        if not self._tools:
            return Retrieval()

        account = find_account(question, history, self.account_re)
        calls, personal = await asyncio.gather(
            self._select_tools(question, history, account), self._is_personal(question, history)
        )
        if personal:
            if account is None:
                logger.info("The question is about the user's own data and no account number is known")
                return Retrieval(needs_account=True)
            # Do not rely on the model to scope the query: force every call to the user's account.
            calls = scope_calls(calls, account, self._tools_by_name, self.account_field)
        scoped_account = account if personal else None
        if not calls:
            return Retrieval(account=scoped_account)
        logger.info("Context Gateway tools selected: %s", ", ".join(n for n, _ in calls))
        llm_args = dict(calls)

        # Search tools that will run as the partner of a filter_{entity} call.
        paired = {p for name, _ in calls if (p := search_partner(name, self._tools_by_name))}
        done: set[str] = set()
        sections: list[str] = []

        for name, args in calls:
            if name in done or name in paired:
                continue
            done.add(name)

            partner = search_partner(name, self._tools_by_name)
            if not partner:
                result = await self._run_tool(name, args)
                if result is not None and not result.is_error:
                    sections.append(f"## {name}\n{render_result(result)}")
                continue

            # filter_{entity} -> always follow with search_{entity}_by_text.
            done.add(partner)
            if scoped_account and partner not in llm_args:
                # A personal question whose filter was set up by the account scoping, not chosen together
                # with a text search: the user's own records are wanted whole, so do not narrow them by
                # searching for the words of the question.
                records, raw = await self._fetch_all(name, args)
                if records is None:
                    if raw is not None and not raw.is_error:
                        sections.append(f"## {name}\n{render_result(raw)}")
                elif records:
                    sections.append(f"## {name}\n{json.dumps(records, indent=2, default=str)}")
                else:
                    sections.append(f"## {name}\nNo matching records for account {scoped_account}.")
                continue
            search_args = llm_args.get(partner) or text_args(self._tools_by_name[partner], question)
            (filter_records, filter_raw), (search_records, search_raw) = await asyncio.gather(
                self._fetch_all(name, args), self._fetch_all(partner, search_args)
            )

            # The text search is not restricted to an account, so for a personal question only the
            # (scoped) filter may supply records.
            if search_records is None:
                pair = ((name, filter_raw),) if scoped_account else ((name, filter_raw), (partner, search_raw))
                for title, res in pair:
                    if res is not None and not res.is_error:
                        sections.append(f"## {title}\n{render_result(res)}")
            elif not filter_records:
                if scoped_account:
                    sections.append(f"## {name}\nNo matching records for account {scoped_account}.")
                else:
                    sections.append(f"## {partner}\n{json.dumps(search_records, indent=2, default=str)}")
            else:
                wanted = {record_key(r) for r in filter_records}
                merged = [r for r in search_records if record_key(r) in wanted]
                if merged:
                    sections.append(f"## {name} and {partner} (in common)\n{json.dumps(merged, indent=2, default=str)}")
                else:
                    # Nothing in common: the structured filter is the precise part, so return its records.
                    sections.append(f"## {name}\n{json.dumps(filter_records, indent=2, default=str)}")

        context = "\n\n".join(sections)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n... (truncated)"
        return Retrieval(context=context, account=scoped_account)
