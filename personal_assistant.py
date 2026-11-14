#!/usr/bin/env python3
"""
Memory Prompt Agent Example

This agent talks to the Redis Iris Agent Memory service through the redis-iris
SDK. Instead of manually managing conversation history, this agent:

1. Records every turn as a session event (user message before the model call,
   assistant answer after it)
2. For each turn, recalls the user's long-term memories for the current input
   and reads the session transcript (summary plus recent turns)
3. Combines those results with the system prompt
4. Sends the enriched context to the LLM

Environment variables (see .env):
- OPENAI_API_KEY: Required for OpenAI ChatGPT
- AGENT_MEMORY_URL: Agent Memory data-plane URL
- AGENT_MEMORY_STORE_ID: Agent Memory store ID
- IRIS_API_KEY: API key shared by all Iris services
"""

import asyncio
import logging
import os
import re
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from redis_iris import AgentMemory

import utils
from context_retriever import DEFAULT_ACCOUNT_PATTERN, ContextRetriever
from utils import RateLimiter
from utils import SemanticWrapper
from utils import sanitize_id

load_dotenv()


# Configure logging: the level comes from LOG_LEVEL in .env (CRITICAL, the most severe, when unset)
utils.configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_USER = "demo_user"

REJECT_MESSAGE = "This question violates the terms of our user agreement."

NEEDS_ACCOUNT_INSTRUCTION = (
    "The user's message is about their own account or information, and their account number is "
    "not known. Ask them to provide their account number, and say you need it to look up their "
    "details. Do not guess it, do not answer from other customers' data, and do not give "
    "account-specific details until they provide it."
)


MAX_WELCOME_MEMORIES = 10  # facts listed when the model cannot write the welcome-back sentence
MAX_WELCOME_FACTS = 30  # facts given to the model that writes it


@dataclass(frozen=True)
class WelcomeMessages:
    """The texts of the welcome flow, each read from a file named in .env.

    greeting: sent automatically when a chat starts; it asks for the user's UID (account number).
    back: sent when Agent Memory holds memories for the user name; {memories} is replaced by a short
        sentence or two, written by the model from those memories (see back_prompt).
    back_prompt: the instructions for the model that writes that sentence.
    new: sent when it holds none; it asks for preferences to remember. {account_num} and {user_id}
        are available.
    An empty WELCOME_MESSAGE_FILE turns the whole flow off."""

    greeting: str = ""
    back: str = ""
    new: str = ""
    back_prompt: str = ""

    @classmethod
    def from_env(cls) -> "WelcomeMessages":
        greeting_file = os.getenv("WELCOME_MESSAGE_FILE", "welcome-message").strip()
        if not greeting_file:
            return cls()

        def read(var: str, default: str) -> str:
            return utils.load_prompt(Path(__file__).parent / (os.getenv(var, default).strip() or default))

        return cls(
            greeting=utils.load_prompt(Path(__file__).parent / greeting_file),
            back=read("WELCOME_BACK_MESSAGE_FILE", "welcome-back-message"),
            new=read("WELCOME_NEW_MESSAGE_FILE", "welcome-new-message"),
            back_prompt=read("WELCOME_BACK_PROMPT_FILE", "welcome-back-prompt"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.greeting)


@dataclass(frozen=True)
class Identification:
    """The result of the user giving their UID: the reply to send."""

    response: str
    returning: bool  # True when memories were found


class Reply(str):
    """The assistant's reply text. `cacheable` is False for a reply that must not be reused for a
    later, similar question: one that asks for the account number, one built from a single
    account's data, or an error."""

    cacheable: bool = True

    def __new__(cls, text: str, cacheable: bool = True) -> "Reply":
        reply = super().__new__(cls, text)
        reply.cacheable = cacheable
        return reply

LLM_PROMPT_PATH = Path(__file__).parent / os.getenv("LLM_PROMPT", "llm_prompt")
SYSTEM_PROMPT = utils.load_prompt(LLM_PROMPT_PATH)  # <prompt>file</prompt> tags are expanded


class MemoryPromptAgent:
    """
    A conversational agent that demonstrates the memory prompt feature.

    This agent uses the memory prompt to automatically retrieve and format
    relevant memories for each conversation turn, providing rich context
    to the LLM without manual history management.
    """

    def __init__(self):
        # One configured memory object, created once. The client connects lazily, so a missing
        # URL / store ID / key raises ConfigurationError naming it on the first memory call.
        self.memory = AgentMemory.from_env(
            utils.agent_memory_environ(), assistant_id="memory-prompt-agent"
        )
        # Context surface, reached through the Context Gateway (disabled when CONTEXT_GATEWAY_URL is unset).
        self.context = ContextRetriever.from_env()
        # Welcome flow: a greeting that asks for the UID, then a reply built from the user's memories.
        self.welcome = WelcomeMessages.from_env()
        # Only memories of this type are listed in the welcome-back message (empty: every type).
        self.welcome_memory_type = utils.resolve_env("WELCOME_MEMORY_TYPE", "semantic").strip().lower()
        self.account_re = re.compile(
            utils.resolve_env("CONTEXT_UID_PATTERN") or DEFAULT_ACCOUNT_PATTERN, re.IGNORECASE
        )
        self._setup_llm()

    def _conversation(self, session_id: str, user_id: str):
        """The user's session. Ids keep the app's original sanitized derivation."""
        return self.memory.conversation(
            owner=sanitize_id(user_id), session_id=sanitize_id(session_id)
        )

    def _setup_llm(self):
        """Set up the LLM instance."""
        api_key = utils.resolve_env("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=api_key, max_retries=4)

    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.memory.__aexit__(None, None, None)
        except Exception:
            logger.debug("Agent Memory client was not open", exc_info=True)

    async def _record_turn(
        self, session_id: str, user_id: str, role: str, content: str, source: str | None = None
    ) -> None:
        """Record one conversation turn as a session event.

        `source` says which route produced the answer ("llm", "faq" or "cache"); it is
        stored as event metadata, so it is visible in the transcript but never sent to the model.
        """
        conversation = self._conversation(session_id, user_id)
        metadata = {"source": source} if source else None
        write = conversation.auser if role == "user" else conversation.aassistant
        await utils.aretry(lambda: write(content, metadata=metadata), "Agent Memory: record turn")

    # ---- welcome flow ----

    def account_in(self, message: str) -> str | None:
        """The account number the user typed in reply to the welcome message, or None.

        It may stand alone ("ACC000007") or sit in a sentence ("hi my account number is ACC000007").
        It must not be part of a longer word or number."""
        match = re.search(rf"(?<![A-Za-z0-9])(?:{self.account_re.pattern})(?![A-Za-z0-9])", message, re.IGNORECASE)
        return match.group(0).upper() if match else None

    async def _describe_user(self, facts: list[str]) -> str | None:
        """A sentence or two on what is known about the user, written by the model from their
        memories (the welcome-back prompt sets the style), or None when none of the memories is a
        lasting fact about the user. If the model cannot be reached the facts are listed instead, so
        a returning user still gets their welcome."""
        listing = "\n".join(f"- {fact}" for fact in facts[:MAX_WELCOME_FACTS])
        try:
            answer = await asyncio.wait_for(
                self.llm.bind(temperature=0.3).ainvoke(
                    [
                        {"role": "system", "content": self.welcome.back_prompt},
                        {"role": "user", "content": f"Today's date: {date.today():%B %d, %Y}\n\nMemories:\n{listing}"},
                    ]
                ),
                timeout=30,
            )
        except Exception:
            logger.exception("Could not write the welcome-back sentence; listing the memories instead")
            shown = [f"- {fact}" for fact in facts[:MAX_WELCOME_MEMORIES]]
            return "Here is what I know about you:\n" + "\n".join(shown)
        text = " ".join(str(answer.content).split()).strip().strip('"')
        return None if not text or text.upper().rstrip(".") == "NONE" else text

    async def identify(
        self, session_id: str, user_id: str, account: str, message: str | None = None
    ) -> Identification:
        """The user gave their UID (account number): greet them from what Agent Memory holds.

        Only long-term memories of type WELCOME_MEMORY_TYPE (default "semantic") are used, and the
        model turns them into a sentence about what is known about the user.
        The memories are looked up under the user name entered in the UI (`user_id`), which is the
        owner every other turn is recorded and recalled under. The account number is not used as a
        lookup key, and nothing is read from the context surface, so the user's profile there is
        not consulted. If memories exist, the reply is the welcome-back message listing them;
        otherwise it asks for preferences and other things to remember.

        The exchange (including the account number, which later questions pick up from the session)
        is recorded in Agent Memory, best effort. A memory outage is logged and treated as "nothing
        found", so the user is never left without a reply."""
        owner = sanitize_id(user_id)
        try:
            memories = await utils.aretry(lambda: self.memory.amemories(owner), "Agent Memory: list memories")
        except Exception:
            logger.exception("Could not read the memories of %s for the welcome message", owner)
            memories = []

        facts: list[str] = []
        for memory in memories:
            # Long-term memories of the configured type only: semantic memories are lasting facts and
            # preferences; episodic ones are events from past conversations ("On <date>, ...").
            if self.welcome_memory_type and (memory.memory_type or "").lower() != self.welcome_memory_type:
                continue
            text = " ".join(str(memory.text).split())
            if text and text not in facts:
                facts.append(text)
        knowledge = await self._describe_user(facts) if facts else None
        if knowledge:
            template = self.welcome.back.replace("{memories}", knowledge)
        else:
            facts = []
            template = self.welcome.new
        reply = template.replace("{account_num}", account).replace("{user_id}", user_id)

        await self.record_exchange(session_id, user_id, message or account, reply, "welcome")
        return Identification(response=reply, returning=bool(facts))

    async def record_exchange(
        self, session_id: str, user_id: str, user_input: str, response: str, source: str
    ) -> None:
        """Record a question and the reply of the FAQ or cache route in Agent Memory.

        The LLM route records its own turns in process_user_input. Blocked ("reject") messages are
        never recorded. A memory outage must not stop the reply the user is already owed, so
        failures are logged and swallowed here.
        """
        try:
            await self._record_turn(session_id, user_id, "user", user_input, source)
            await self._record_turn(session_id, user_id, "assistant", response, source)
        except Exception:
            logger.exception("Could not record the %s exchange in Agent Memory", source)

    async def _get_memory_prompt(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        show_memories: bool = False,
    ) -> list[dict[str, Any]]:
        """Build memory-enriched context: recalled long-term memories plus the session transcript."""
        # RECALL WHAT IS KNOWN ABOUT THE USER FOR THIS QUESTION
        known = await utils.aretry(
            lambda: self.memory.arecall(user_input, owner=sanitize_id(user_id), limit=30),
            "Agent Memory: recall",
        )

        messages: list[dict[str, Any]] = []

        if known:
            messages.append(
                {
                    "role": "system",
                    "content": known.as_prompt("Long term memories related to the user's message"),
                }
            )

        # The recalled memories are logged at INFO (LOG_LEVEL=INFO to see them), not printed.
        if show_memories and known:
            for i, memory in enumerate(list(known)[:5], 1):
                logger.info("Recalled memory [%d] id=%s :: %s", i, memory.id, memory.text)

        # SESSION SO FAR: the summary (if the service wrote one) first, then recent turns in order
        conversation = self._conversation(session_id, user_id)
        transcript = await utils.aretry(conversation.atranscript, "Agent Memory: transcript")
        messages.extend(transcript.as_messages())

        return messages





    async def run_demo_conversation(
        self, session_id: str = "memory_prompt_demo", user_id: str = DEFAULT_USER
    ):
        """Run a demonstration conversation showing memory prompt capabilities."""
        print("🧠 Memory Prompt Agent Demo")
        print("=" * 50)
        print("This demo shows how the memory prompt feature automatically retrieves")
        print("relevant memories to provide contextual responses.")
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print()

        # First, we need to create some long-term memories to demonstrate the feature
        print("🔧 Setting up demo by checking for existing background memories...")

        owner = sanitize_id(user_id)

        # Check if we already have demo memories for this user
        should_create_memories = True
        try:
            existing_memories = await self.memory.arecall("Alice", owner=owner, limit=10)

            if len(existing_memories) >= 5:
                print("✅ Found existing background memories about Alice")
                print()
                should_create_memories = False
        except Exception:
            # Recall failed, proceed with memory creation
            pass

        if should_create_memories:
            print("🔧 Creating new background memories...")

            # Create some background memories that the prompt agent can use
            demo_memories = [
                ("User Alice loves Italian food, especially pasta and pizza", ["food", "preferences"]),
                (
                    "Alice works as a software engineer at a tech startup in San Francisco",
                    ["work", "job", "location"],
                ),
                (
                    "Alice enjoys hiking on weekends and has climbed Mount Tamalpais several times",
                    ["hobbies", "outdoors", "hiking"],
                ),
                (
                    "Alice is planning a trip to Italy next summer to visit Rome and Florence",
                    ["travel", "plans", "Italy"],
                ),
                (
                    "Alice mentioned she's learning Italian using Duolingo and taking evening classes",
                    ["learning", "languages", "education"],
                ),
            ]

            for text, topics in demo_memories:
                await self.memory.aremember(text, owner=owner, topics=topics)
            print("✅ Created background memories about Alice")
            print()

        # Demo conversation scenarios that should trigger memory retrieval
        demo_inputs = [
            "I love Italian food. What's a good Italian restaurant recommendation?",
            "I'm planning a trip to Italy next summer to visit Rome and Florence. Any tips?",
            "I enjoy hiking on weekends. What should I do this weekend for some outdoor activity?",
            "I'm learning Italian. Any suggestions to speed up my progress?",
            "I'm a software engineer in San Francisco. Can you suggest some programming projects?",
            "What do you know about me from our previous conversations?",
        ]

        try:
            for user_input in demo_inputs:
                print(f"👤 User: {user_input}")
                print("🤔 Assistant is thinking... (retrieving relevant memories)")

                response = await self.process_user_input(
                    user_input, session_id, user_id
                )
                print(f"🤖 Assistant: {response}")
                print("-" * 70)
                print()

                # Add a small delay for better demo flow
                await asyncio.sleep(1)

        finally:
            await self.cleanup()

    async def run_interactive(
        self, session_id: str = "memory_prompt_session", user_id: str = DEFAULT_USER
    ):
        """Main async interaction loop for the memory prompt agent."""
        print("🧠 Memory Prompt Agent - Interactive Mode")
        print("=" * 50)
        print("This agent uses memory prompts to provide contextual responses.")
        print("Try mentioning your preferences, interests, or past conversations!")
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print("Type 'exit' to quit")
        print()

        try:

            #SEMANTIC CACHE AND SEMANTIC ROUTER WRAPPER
            sw = SemanticWrapper(redis_connection=None)

            #WELCOME MESSAGE: asks for the UID; the first reply may be the account number
            awaiting_uid = self.welcome.enabled
            if awaiting_uid:
                print(f"[Welcome]: {self.welcome.greeting}")
                print()

            while True:
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("👋 Thank you for using the Memory Prompt Agent!")
                    break

                if awaiting_uid:
                    awaiting_uid = False
                    account = self.account_in(user_input)
                    if account:
                        who = await self.identify(session_id, user_id, account, user_input)
                        print(f"[Welcome]: {who.response}")
                        print()
                        continue

                #SEMANTIC ROUTER FOR GUARDRAILS
                sugested_route = sw.get_route(user_input)

                if sugested_route == "reject":
                    # Blocked messages are deliberately not recorded in Agent Memory.
                    print(f"[Semantic Router]: {REJECT_MESSAGE}")
                    print()
                    continue

                #ROUTE FAQ QUESTIONS TO THE AGENT PLAYBOOKS KNOWLEDGE BASE
                if sugested_route == "FAQ":
                    faq_answer = sw.get_faq_answer(user_input)
                    if faq_answer != "NONE":
                        print(f"[FAQ Knowledge Base]:")
                        print(f"{faq_answer}")
                        print()
                        await self.record_exchange(session_id, user_id, user_input, faq_answer, "faq")
                        continue

                #CHECK SEMANTIC CACHE FOR SIMILAR QUESTIONS
                cached_result = sw.get_similar(user_input, user_id = user_id, session_id = session_id)
                if cached_result != "NONE":
                    print(f"[Semantic Cache]:")
                    print(f"{cached_result}")
                    print()
                    await self.record_exchange(session_id, user_id, user_input, cached_result, "cache")
                    continue

                # Process input and get response
                print("🤔 Thinking...")
                response = await self.process_user_input(
                    user_input, session_id, user_id
                )
                print(f"🤖 Assistant: {response}")
                print()

                #ADD TO SEMANTIC CACHE
                if response.cacheable and response.find("error processing your request") == -1:
                    sw.set_cache(query = user_input, response= response, user_id = user_id, session_id = session_id)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            #sw.clear_cache()
        finally:
            await self.cleanup()


    async def process_user_input(
        self, user_input: str, session_id: str, user_id: str
    ) -> "Reply":
        """Process user input and return assistant response (a str; see Reply.cacheable)."""
        try:
            # Add user message to working memory first
            await self._record_turn(
                session_id, user_id, "user", user_input, "llm"
            )

            # Generate response using memory prompt (with memory visibility in demo mode)
            response = await self._generate_response(
                session_id, user_id, user_input, show_memories=True
            )

            # Add assistant response to working memory
            await self._record_turn(
                session_id, user_id, "assistant", response, "llm"
            )

            return response

        except Exception as e:
            logger.exception(f"Error processing user input: {e}")
            return Reply("I'm sorry, I encountered an error processing your request.", cacheable=False)


    async def _generate_response(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        show_memories: bool = False,
    ) -> str:
        """Generate a response using the LLM with memory and context-surface context."""
        # The tool selector sees the recalled memories and recent turns, so it can resolve
        # follow-ups ("what about their positions?"); memory is read first, then the Gateway.
        # retrieve() returns "" when the gateway is disabled, finds nothing relevant, or fails.
        memory_messages = await self._get_memory_prompt(session_id, user_id, user_input, show_memories)
        retrieval = await self.context.retrieve(user_input, memory_messages)

        # Add system prompt to the beginning
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if retrieval.needs_account:
            messages.append({"role": "system", "content": NEEDS_ACCOUNT_INSTRUCTION})
        elif retrieval.context:
            messages.append(
                {
                    "role": "system",
                    "content": "Context retrieved from the context surface for the user's message. "
                    "Use it to answer when it is relevant, and do not invent details beyond it:\n"
                    + retrieval.context,
                }
            )

        # Add memory messages (these already include context and the user query)
        for msg in memory_messages:
            # Extract text content from the message structure
            content = msg.get("content", "")
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            messages.append({"role": msg["role"], "content": str(content)})

        # Generate response
        response = self.llm.invoke(messages)
        return Reply(str(response.content), cacheable=retrieval.cacheable)
    

    def run_demo(self, session_id: str = "memory_prompt_demo", user_id: str = DEFAULT_USER):
        """Synchronous wrapper for the async demo method."""
        asyncio.run(self.run_demo_conversation(session_id, user_id))

    def run(self, session_id: str = "memory_prompt_session", user_id: str = DEFAULT_USER):
        """Synchronous wrapper for the async interactive method."""
        asyncio.run(self.run_interactive(session_id, user_id))


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Memory Prompt Agent Example")
    parser.add_argument("--user-id", default=DEFAULT_USER, help="User ID")
    parser.add_argument(
        "--session-id", default="demo_memory_prompt_session", help="Session ID"
    )
    parser.add_argument(
        "--agent-memory-service-url",
        default=None,
        help="Agent Memory data-plane URL (overrides AGENT_MEMORY_URL)",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run automated demo conversation"
    )

    args = parser.parse_args()

    # Check for required API keys / configuration
    required_env_vars = [
        "OPENAI_API_KEY",
        "AGENT_MEMORY_URL",
        "AGENT_MEMORY_STORE_ID",
        "IRIS_API_KEY",
    ]
    missing = [name for name in required_env_vars if not utils.resolve_env(name)]
    if missing:
        print(f"Error: missing required environment variable(s): {', '.join(missing)}")
        print("See .env for the full list of required settings.")
        return

    # Set memory service URL from argument if provided
    if args.agent_memory_service_url:
        os.environ["AGENT_MEMORY_URL"] = args.agent_memory_service_url

    try:
        agent = MemoryPromptAgent()



        if args.demo:
            # Run automated demo
            agent.run_demo(session_id=args.session_id, user_id=args.user_id)
        else:
            # Run interactive session
            agent.run(session_id=args.session_id, user_id=args.user_id)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Error running memory prompt agent: {e}")
        raise


if __name__ == "__main__":
    main()
