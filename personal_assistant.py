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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from redis_iris import AgentMemory

import utils
from utils import RateLimiter
from utils import SemanticWrapper
from utils import sanitize_id

load_dotenv()


# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce third-party logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DEFAULT_USER = "demo_user"

LLM_PROMPT_PATH = Path(__file__).parent / os.getenv("LLM_PROMPT", "llm_prompt")
SYSTEM_PROMPT = LLM_PROMPT_PATH.read_text(encoding="utf-8").strip()


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
        self._setup_llm()

    def _conversation(self, session_id: str, user_id: str):
        """The user's session. Ids keep the app's original sanitized derivation."""
        return self.memory.conversation(
            owner=sanitize_id(user_id), session_id=sanitize_id(session_id)
        )

    def _setup_llm(self):
        """Set up the LLM instance."""
        api_key = utils.resolve_env("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=api_key)

    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.memory.__aexit__(None, None, None)
        except Exception:
            logger.debug("Agent Memory client was not open", exc_info=True)

    async def _record_turn(self, session_id: str, user_id: str, role: str, content: str) -> None:
        """Record one conversation turn as a session event."""
        conversation = self._conversation(session_id, user_id)
        if role == "user":
            await conversation.auser(content)
        else:
            await conversation.aassistant(content)

    async def _get_memory_prompt(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        show_memories: bool = False,
    ) -> list[dict[str, Any]]:
        """Build memory-enriched context: recalled long-term memories plus the session transcript."""
        # RECALL WHAT IS KNOWN ABOUT THE USER FOR THIS QUESTION
        known = await self.memory.arecall(user_input, owner=sanitize_id(user_id), limit=30)

        messages: list[dict[str, Any]] = []

        if known:
            messages.append(
                {
                    "role": "system",
                    "content": known.as_prompt("Long term memories related to the user's message"),
                }
            )

        if show_memories:
            if known:
                for i, memory in enumerate(list(known)[:5], 1):
                    print(f"   [{i}] id={memory.id} :: {memory.text}")
            else:
                print()

        # SESSION SO FAR: the summary (if the service wrote one) first, then recent turns in order
        transcript = await self._conversation(session_id, user_id).atranscript()
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

            while True:
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("👋 Thank you for using the Memory Prompt Agent!")
                    break

                #SEMANTIC ROUTER FOR GUARDRAILS
                sugested_route = sw.get_route(user_input)

                if sugested_route == "reject":
                    print(f"[Semantic Router]: This question violates the terms of our user agreement.")
                    print()
                    continue

                #ROUTE FAQ QUESTIONS TO THE AGENT PLAYBOOKS KNOWLEDGE BASE
                if sugested_route == "FAQ":
                    faq_answer = sw.get_faq_answer(user_input)
                    if faq_answer != "NONE":
                        print(f"[FAQ Knowledge Base]:")
                        print(f"{faq_answer}")
                        print()
                        continue

                #CHECK SEMANTIC CACHE FOR SIMILAR QUESTIONS
                cached_result = sw.get_similar(user_input, user_id = user_id, session_id = session_id)
                if cached_result != "NONE":
                    print(f"[Semantic Cache]:")
                    print(f"{cached_result}")
                    print()
                    continue                  

                # Process input and get response
                print("🤔 Thinking...")
                response = await self.process_user_input(
                    user_input, session_id, user_id
                )
                print(f"🤖 Assistant: {response}")
                print()

                #ADD TO SEMANTIC CACHE
                if response.find("error processing your request") == -1:
                    sw.set_cache(query = user_input, response= response, user_id = user_id, session_id = session_id)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            #sw.clear_cache()
        finally:
            await self.cleanup()


    async def process_user_input(
        self, user_input: str, session_id: str, user_id: str
    ) -> str:
        """Process user input and return assistant response."""
        try:
            # Add user message to working memory first
            await self._record_turn(
                session_id, user_id, "user", user_input
            )

            # Generate response using memory prompt (with memory visibility in demo mode)
            response = await self._generate_response(
                session_id, user_id, user_input, show_memories=True
            )

            # Add assistant response to working memory
            await self._record_turn(
                session_id, user_id, "assistant", response
            )

            return response

        except Exception as e:
            logger.exception(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error processing your request."
        

    async def _generate_response(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        show_memories: bool = False,
    ) -> str:
        """Generate a response using the LLM with memory-enriched context."""
        # Get memory prompt with relevant context
        memory_messages = await self._get_memory_prompt(
            session_id, user_id, user_input, show_memories
        )

        # Add system prompt to the beginning
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add memory messages (these already include context and the user query)
        for msg in memory_messages:
            # Extract text content from the message structure
            content = msg.get("content", "")
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            messages.append({"role": msg["role"], "content": str(content)})

        # Generate response
        response = self.llm.invoke(messages)
        return str(response.content)
    

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
