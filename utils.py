import asyncio
import logging
import os
import re
import socket
import sys
import threading
import time

import httpx
import redis
import json

from datetime import datetime
from pathlib import Path

from playbook_faq import SUBJECT_ID_ATTRIBUTE, sync_knowledge_base
from redis_iris import AgentPlaybooks
from redis_iris.agent_playbooks import eq
from redisvl.extensions.llmcache import SemanticCache
from redisvl.extensions.router import  SemanticRouter
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import HFTextVectorizer

_APP_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def resolve_env(var_name: str, default: str = "") -> str:
    """
    Read var_name from the environment. If the resulting value is itself the
    name of another environment variable that is currently set, use that
    variable's value instead of the literal value.

    This lets .env entries indirect to a real environment variable, e.g.
    REDIS_DB_PASSWORD=REDIS_PASSWORD looks up REDIS_PASSWORD first and falls
    back to the literal string "REDIS_PASSWORD" if it isn't set.
    """
    raw_value = os.getenv(var_name, default)
    if raw_value and raw_value in os.environ:
        return os.environ[raw_value]
    return raw_value


# Delays between attempts when a connection cannot be opened. A DNS lookup that fails
# ("getaddrinfo failed") on this network usually succeeds a moment later.
CONNECT_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0)


def is_connect_failure(exc: BaseException | None) -> bool:
    """True when exc, anything it was raised from, or any member of an exception group is a
    failure to open the connection (DNS lookup, refused, connect timeout). The request never
    reached the server, so it is safe to repeat, even a write."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout, socket.gaierror)):
            return True
        stack += [e.__cause__, e.__context__, *getattr(e, "exceptions", ())]
    return False


async def aretry(make_call, what: str, delays=CONNECT_RETRY_DELAYS):
    """await make_call(), repeating it after a short wait when the connection cannot be opened.

    make_call is a function that returns a fresh awaitable each time. Only connection failures
    are retried (see is_connect_failure); every other error is raised at once. Each retry is one
    warning line instead of a traceback. After the last attempt the original error is raised."""
    for delay in (*delays, None):
        try:
            return await make_call()
        except Exception as exc:
            if delay is None or not is_connect_failure(exc):
                raise
            logger.warning("%s: could not connect (%s); retrying in %.1fs", what, _root_message(exc), delay)
            await asyncio.sleep(delay)


def retry(make_call, what: str, delays=(0.3, 0.6)):
    """Blocking version of aretry for synchronous calls. The default waits are short because
    the caller blocks while it waits."""
    for delay in (*delays, None):
        try:
            return make_call()
        except Exception as exc:
            if delay is None or not is_connect_failure(exc):
                raise
            logger.warning("%s: could not connect (%s); retrying in %.1fs", what, _root_message(exc), delay)
            time.sleep(delay)


def _root_message(exc: BaseException) -> str:
    while exc.__cause__ is not None:
        exc = exc.__cause__
    return str(exc)[:80]


DEFAULT_LOG_LEVEL = "CRITICAL"  # the most severe level: with nothing set, the application logs next to nothing
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
}


def log_level() -> int:
    """The level named by LOG_LEVEL (DEBUG, INFO, WARNING, ERROR or CRITICAL, any case), or
    CRITICAL, the most severe, when it is unset, empty or not a level name."""
    raw = os.getenv("LOG_LEVEL", "").strip().upper()
    if raw and raw not in _LOG_LEVELS:
        print(f"LOG_LEVEL={raw!r} is not a log level; using {DEFAULT_LOG_LEVEL}", file=sys.stderr)
    return _LOG_LEVELS.get(raw, _LOG_LEVELS[DEFAULT_LOG_LEVEL])


def configure_logging() -> int:
    """Set the log level for the application from LOG_LEVEL and return it.

    The HTTP and OpenAI client libraries log every request at INFO, which would bury the
    application's own messages, so they are kept at WARNING or the chosen level, whichever is
    more severe. The web server's loggers (uvicorn) are set to the same level."""
    level = log_level()
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))
    # The web server's own lines (startup, and one access line per request) follow it too. uvicorn
    # sets its loggers up before it imports the app, so setting their levels here takes effect.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)
    return level


def env_flag(var_name: str, default: bool = True) -> bool:
    """A true/false setting from the environment. false, 0, no and off (any case) mean False,
    true, 1, yes and on mean True, and anything else, or an unset value, gives the default."""
    value = resolve_env(var_name).strip().lower()
    if value in ("false", "0", "no", "off"):
        return False
    if value in ("true", "1", "yes", "on"):
        return True
    return default


_PROMPT_TAG = re.compile(r"<prompt>\s*(.*?)\s*</prompt>", re.DOTALL)
_MAX_PROMPT_DEPTH = 5


def load_prompt(path: str | Path, _chain: tuple[Path, ...] = ()) -> str:
    """Read a prompt file, replacing every <prompt>file_name</prompt> with that file's content.

    A file name is relative to the file that names it (or absolute). Included files may include
    others, up to a depth of five. A missing file, a circular include or an empty result raises a
    RuntimeError that names the file, so a bad prompt stops the app at startup instead of
    silently dropping instructions."""
    path = Path(path).resolve()
    if path in _chain:
        raise RuntimeError("Circular <prompt> include: " + " -> ".join(p.name for p in (*_chain, path)))
    if len(_chain) >= _MAX_PROMPT_DEPTH:
        raise RuntimeError(f"<prompt> includes are nested more than {_MAX_PROMPT_DEPTH} deep at {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        origin = f" (included from {_chain[-1]})" if _chain else ""
        raise RuntimeError(f"Cannot read the prompt file {path}{origin}: {e}") from e

    def include(match: re.Match[str]) -> str:
        return load_prompt(path.parent / match.group(1), (*_chain, path)).strip()

    text = _PROMPT_TAG.sub(include, text).strip()
    if not text:
        raise RuntimeError(f"The prompt file {path} is empty")
    return text


_INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9-]")


def sanitize_id(value: str, max_length: int = 64) -> str:
    """Coerce a value into the [a-zA-Z0-9-]{1,64} shape Agent Memory requires for IDs.

    Kept identical to the derivation the app used before moving to the Iris SDK, so existing
    sessions and owners ("demo_user" -> "demo-user") continue unchanged.
    """
    return _INVALID_ID_CHARS.sub("-", value)[:max_length]


def agent_memory_environ() -> dict[str, str]:
    """Agent Memory settings for AgentMemory.from_env(), honouring resolve_env indirection.

    The app's .env uses AGENT_MEMORY_URL and AGENT_MEMORY_STORE_ID; the SDK expects the
    IRIS_AGENT_MEMORY_* names, so map them here. Every Iris service uses the one IRIS_API_KEY.
    """
    env = {
        "IRIS_AGENT_MEMORY_URL": resolve_env("AGENT_MEMORY_URL"),
        "IRIS_AGENT_MEMORY_STORE_ID": resolve_env("AGENT_MEMORY_STORE_ID"),
        "IRIS_AGENT_MEMORY_API_KEY": resolve_env("IRIS_API_KEY"),
        "IRIS_AGENT_MEMORY_TIMEOUT": resolve_env("AGENT_MEMORY_TIMEOUT"),
    }
    return {name: value for name, value in env.items() if value}


def _playbook_environ() -> dict[str, str]:
    """Agent Playbooks settings for AgentPlaybooks.from_env(), honouring resolve_env indirection.

    The app's .env uses AGENT_PLAYBOOKS_URL and AGENT_PLAYBOOKS_PLAYBOOK_ID; the SDK expects the
    IRIS_AGENT_PLAYBOOKS_* names, so map them here. Every Iris service uses the one IRIS_API_KEY.
    """
    env = {
        "IRIS_AGENT_PLAYBOOKS_URL": resolve_env("AGENT_PLAYBOOKS_URL"),
        "IRIS_AGENT_PLAYBOOKS_PLAYBOOK_ID": resolve_env("AGENT_PLAYBOOKS_PLAYBOOK_ID"),
        "IRIS_AGENT_PLAYBOOKS_API_KEY": resolve_env("IRIS_API_KEY"),
    }
    return {name: value for name, value in env.items() if value}


class RateLimiter:
    def __init__(self, redis_connection : redis.Redis, key_prefix="rag:limiter:", window=20):
        self.rds = redis_connection
        self.key_prefix = key_prefix
        self.window = window

    def is_allowed(self, user_key):
        allow_call = True

        key = f"{self.key_prefix}{user_key}"

        current_limit = self.rds.hincrby(key, "call_count", 1)
        self.rds.hexpire(key, self.window, "call_count", nx = True)

        user_limit = self.rds.hget(user_key, "call_limit")

        if current_limit > int(user_limit):
            allow_call = False

        return allow_call


class SemanticWrapper:
    def __init__(self, redis_connection : redis.Redis):

        #CONNECT TO REDIS DB
        if not redis_connection:
            self.rds = get_redis_connection()

        #LOAD EMBEDDING MODEL
        em = HFTextVectorizer("sentence-transformers/all-MiniLM-L6-v2") 

        llm_router_file = os.getenv("LLM_ROUTER_FILE", "llm-router.json")
        llm_routes = load_dict_from_file(_APP_DIR / llm_router_file)

        #SEMANTIC ROUTER
        logger.info("Setting Up Semantic Router...")

        # SemanticRouter's overwrite flag only recreates the index definition;
        # it does NOT drop existing route vectors (it calls index.create with
        # drop=False internally). So a route renamed/removed in
        # llm-router.json would still be indexed from a previous run and get
        # returned as a stale match, raising ValueError. Drop any existing
        # index and its data first so the router always reflects the current
        # file exactly.
        try:
            self.rds.ft("semantic:router").dropindex(delete_documents=True)
        except redis.exceptions.ResponseError:
            pass  # index did not exist yet

        self.semantic_router = SemanticRouter(
            name="semantic:router",
            vectorizer = em,
            routes = llm_routes,
            redis_client = self.rds,
            overwrite=False
        )

    

        #SEMANTIC CACHE (ENABLE_SEMANTIC_CACHE=false turns it off: no index is created, nothing is
        #looked up or stored, and every question that reaches the LLM step is answered fresh)
        self.cache_enabled = env_flag("ENABLE_SEMANTIC_CACHE", True)
        self.semantic_cache = None
        if self.cache_enabled:
            logger.info("Setting Up Semantic Cache...")

            # Entries are scoped per user and session via tag fields. An index
            # created before these fields existed can't filter on them, so drop it
            # (cache entries are short-lived anyway) and let it be recreated.
            try:
                existing_fields = [
                    attr[1] for attr in self.rds.ft("semantic:cache").info()["attributes"]
                ]
                if not {"user_id", "session_id"} <= set(existing_fields):
                    self.rds.ft("semantic:cache").dropindex(delete_documents=True)
            except redis.exceptions.ResponseError:
                pass  # index did not exist yet

            self.semantic_cache = SemanticCache(
                name="semantic:cache",
                redis_client=self.rds,
                ttl = 600,
                distance_threshold=0.2,
                filterable_fields=[
                    {"name": "user_id", "type": "tag"},
                    {"name": "session_id", "type": "tag"},
                ],
            )
        else:
            logger.info("Semantic Cache is disabled (ENABLE_SEMANTIC_CACHE=false)")

        #FAQ KNOWLEDGE BASE (Redis Iris Agent Playbooks)
        logger.info("Connecting to Agent Playbooks FAQ Knowledge Base...")

        playbook_environ = _playbook_environ()
        self.playbook = AgentPlaybooks.from_env(playbook_environ)

        # SUBJECT_ID selects which subject's FAQs this application answers from. The knowledge base may
        # hold entries for several subjects (each names its own attributes.subject_id) and all of them
        # are loaded; only FAQs whose subject_id equals SUBJECT_ID are looked up. Empty means every
        # FAQ in the Playbook is used.
        self.subject_id = resolve_env("SUBJECT_ID").strip()
        logger.info("FAQ subject: %s", self.subject_id or "(none: using every FAQ in the Playbook)")

        # knowledge-base.json is the source of truth: reconcile it with the Playbook on every
        # startup (unchanged entries are left alone). This runs in the background so the app
        # starts serving at once: the sync can take a minute or more when the Playbook host is
        # slow to resolve, and previously published FAQs keep serving meanwhile. Lookups degrade
        # to the LLM if the service is down. A failed sync is logged and kept in faq_sync_error.
        self.faq_sync_error: Exception | None = None
        self._faq_sync_thread = threading.Thread(
            target=self._sync_faqs,
            args=(
                _APP_DIR / os.getenv("KNOWLEDGE_BASE_FILE", "knowledge-base.json"),
                playbook_environ,
                resolve_env("PLAYBOOK_FAQ_PRUNE", "false").lower() == "true",
                self.subject_id,
            ),
            name="faq-sync",
            daemon=True,
        )
        self._faq_sync_thread.start()

    def _sync_faqs(self, knowledge_base_file, playbook_environ, prune, subject_id=""):
        try:
            sync_knowledge_base(knowledge_base_file, playbook_environ, prune=prune, subject_id=subject_id)
        except Exception as exc:
            self.faq_sync_error = exc
            logger.error("FAQ knowledge base was not synced to Agent Playbooks: %s", exc)

    def get_faq_answer(self, user_query, min_score=None):
        """Return the best published FAQ answer for the query, or "NONE".

        The service scores semantic similarity; the application decides whether
        the score is high enough to serve the approved answer directly. If
        Agent Playbooks is unreachable the lookup is treated as a miss so the
        question falls through to the semantic cache and the LLM.
        """
        if min_score is None:
            min_score = float(resolve_env("FAQ_SEMANTIC_THRESHOLD", "0.75"))
        try:
            scope = eq(SUBJECT_ID_ATTRIBUTE, self.subject_id) if self.subject_id else None
            found = retry(
                lambda: self.playbook.discover(user_query, filter_=scope), "Agent Playbooks FAQ lookup"
            )
        except Exception as exc:
            logger.warning("Agent Playbooks FAQ lookup failed: %s", exc)
            return "NONE"
        best = found.best_faq
        if best is None or best.score < min_score:
            return "NONE"
        return best.faq.answer

    def get_route(self, user_query):
        #CHECK THE ROUTE
        route_match = self.semantic_router(user_query, distance_threshold= 2.0)
        return route_match.name
    
    def get_similar(self, user_query, user_id, session_id):
        if not self.cache_enabled:
            return "NONE"
        scope_filter = (Tag("user_id") == user_id) & (Tag("session_id") == session_id)
        cached_result_list = self.semantic_cache.check(
            prompt = user_query,
            filter_expression = scope_filter,
        )
        if cached_result_list:
            return cached_result_list[0]['response']

        return "NONE"

    def set_cache(self, query, response, user_id, session_id):
        if not self.cache_enabled:
            return
        # filters are part of the entry key, so the same prompt from a
        # different user/session is stored separately, not overwritten.
        self.semantic_cache.store(
                prompt= query,
                response= response,
                filters={"user_id": user_id, "session_id": session_id}
            )

    
    def clear_cache(self) :
        if self.cache_enabled:
            self.semantic_cache.clear()



def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_redis_connection():
    redis_db_host = resolve_env("REDIS_DB_HOST", "localhost")
    redis_db_port = resolve_env("REDIS_DB_PORT", "6379")
    redis_db_user = resolve_env("REDIS_DB_USER", "default")
    redis_db_password = resolve_env("REDIS_DB_PASSWORD", "")

    return redis.Redis(
        host=redis_db_host,
        port=int(redis_db_port),
        username=redis_db_user,
        password=redis_db_password,
        decode_responses=True,
    )


def time_difference(ts1, ts2):
    # Convert Unix timestamps to datetime objects
    dt1 = datetime.fromtimestamp(ts1)
    dt2 = datetime.fromtimestamp(ts2)
    delta = abs(dt2 - dt1)
    minutes, seconds = divmod(delta.total_seconds(), 60) 
    return f"{int(minutes)}:{int(seconds)}"

def main():
    redis_connection = get_redis_connection()

    rate_limiter = RateLimiter(redis_connection, "user:", 10)
    rate_limiter.is_allowed("jay")

if __name__ == "__main__":
    main()
