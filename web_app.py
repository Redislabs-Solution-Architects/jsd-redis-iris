#!/usr/bin/env python3
"""
FastAPI web server wrapping the MemoryPromptAgent for a chat UI.
Run with: uvicorn web_app:app --reload
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import utils
from personal_assistant import REJECT_MESSAGE, MemoryPromptAgent
from utils import SemanticWrapper

load_dotenv()

app = FastAPI(title="Memory Agent Chat")

# Single shared agent instance (lazy init)
_agent: MemoryPromptAgent | None = None
_semantic_wrapper: SemanticWrapper | None = None


def get_agent() -> MemoryPromptAgent:
    global _agent
    if _agent is None:
        _agent = MemoryPromptAgent()
    return _agent


def get_semantic_wrapper() -> SemanticWrapper:
    global _semantic_wrapper
    if _semantic_wrapper is None:
        _semantic_wrapper = SemanticWrapper(redis_connection=None)
    return _semantic_wrapper


class ChatRequest(BaseModel):
    message: str
    session_id: str = "web_session"
    user_id: str = "web_user"
    # True for the first message after the welcome message, which asked for the UID.
    awaiting_uid: bool = False


class ChatResponse(BaseModel):
    response: str
    source: Literal["llm", "cache", "faq", "rejected", "welcome"]


class WelcomeResponse(BaseModel):
    message: str
    awaiting_uid: bool


@app.on_event("startup")
async def startup():
    # Eagerly set up the semantic router, semantic cache, and Agent Playbooks
    # client at process start, instead of paying that cost on the first
    # /chat request.
    get_semantic_wrapper()


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path)


@app.get("/welcome", response_model=WelcomeResponse)
async def welcome():
    """The message sent automatically when a chat starts (empty when the welcome flow is off)."""
    greeting = get_agent().welcome.greeting
    return WelcomeResponse(message=greeting, awaiting_uid=bool(greeting))


async def reply(req: ChatRequest, response: str, source: str) -> ChatResponse:
    """Answer from the FAQ or the cache, recording the exchange in Agent Memory."""
    await get_agent().record_exchange(req.session_id, req.user_id, req.message, response, source)
    return ChatResponse(response=response, source=source)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # The reply to the welcome message: a message with an account number in it is answered from the
    # memories held for the user name in the UI, without going through the router, the FAQ, the cache,
    # the LLM or the context surface.
    if req.awaiting_uid:
        agent = get_agent()
        account = agent.account_in(req.message)
        if account:
            who = await agent.identify(req.session_id, req.user_id, account, req.message)
            return ChatResponse(response=who.response, source="welcome")

    if not utils.resolve_env("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    sw = get_semantic_wrapper()

    # Semantic router guardrail
    route = sw.get_route(req.message)
    if route == "reject":
        # Blocked messages are deliberately not recorded in Agent Memory.
        return ChatResponse(response=REJECT_MESSAGE, source="rejected")

    # Route FAQ questions to the Agent Playbooks knowledge base
    if route == "FAQ":
        faq_answer = sw.get_faq_answer(req.message)
        if faq_answer != "NONE":
            return await reply(req, faq_answer, "faq")

    # Semantic cache check
    cached = sw.get_similar(req.message, user_id=req.user_id, session_id=req.session_id)
    if cached != "NONE":
        return await reply(req, cached, "cache")

    # Full LLM call
    agent = get_agent()
    response = await agent.process_user_input(req.message, req.session_id, req.user_id)

    # Store in semantic cache on success
    if response.cacheable and "error processing your request" not in response:
        sw.set_cache(
            query=req.message,
            response=response,
            user_id=req.user_id,
            session_id=req.session_id,
        )

    return ChatResponse(response=response, source="llm")


@app.on_event("shutdown")
async def shutdown():
    global _agent
    if _agent:
        await _agent.cleanup()
