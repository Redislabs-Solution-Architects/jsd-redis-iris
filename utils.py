import logging
import os
import re
import threading

import redis
import json

from datetime import datetime
from pathlib import Path

from playbook_faq import sync_knowledge_base
from redis_iris import AgentPlaybooks
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
        print("Setting Up Semantic Router...")

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

    

        #SEMANTIC CACHE
        print("Setting Up Semantic Cache...")

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

        #FAQ KNOWLEDGE BASE (Redis Iris Agent Playbooks)
        print("Connecting to Agent Playbooks FAQ Knowledge Base...")

        playbook_environ = _playbook_environ()
        self.playbook = AgentPlaybooks.from_env(playbook_environ)

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
            ),
            name="faq-sync",
            daemon=True,
        )
        self._faq_sync_thread.start()

    def _sync_faqs(self, knowledge_base_file, playbook_environ, prune):
        try:
            sync_knowledge_base(knowledge_base_file, playbook_environ, prune=prune)
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
            found = self.playbook.discover(user_query)
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
        scope_filter = (Tag("user_id") == user_id) & (Tag("session_id") == session_id)
        cached_result_list = self.semantic_cache.check(
            prompt = user_query,
            filter_expression = scope_filter,
        )
        if cached_result_list:
            return cached_result_list[0]['response']

        return "NONE"

    def set_cache(self, query, response, user_id, session_id):
        # filters are part of the entry key, so the same prompt from a
        # different user/session is stored separately, not overwritten.
        self.semantic_cache.store(
                prompt= query,
                response= response,
                filters={"user_id": user_id, "session_id": session_id}
            )

    
    def clear_cache(self) :
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
