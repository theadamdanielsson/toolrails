"""The ASGI proxy: an OpenAI-compatible front door that repairs tool calls.

Point any agent that speaks the OpenAI API at this instead of at Ollama and
nothing else changes. Endpoints:

    POST /v1/chat/completions   the pipeline (or a plain pass-through)
    GET  /v1/models             forwarded to Ollama verbatim
    GET  /health                liveness

Fail-open is the whole safety story: if the repair pipeline raises for any
reason, we forward the original request unchanged rather than return an error.
The worst toolrails can do is nothing.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .pipeline import handle
from .upstream import Upstream


def create_app(ollama_url: str) -> Starlette:
    async def chat_completions(request: Request) -> Response:
        up: Upstream = request.app.state.upstream
        try:
            body: dict[str, Any] = await request.json()
        except (json.JSONDecodeError, ValueError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        wants_stream = bool(body.get("stream"))
        has_tools = bool(body.get("tools"))

        # With tools we run the pipeline on a non-streamed response, then
        # re-emit as a stream if the client asked for one. Guarantee first,
        # streaming second. Without tools there is nothing to repair, so we
        # forward untouched (including native streaming).
        if not has_tools:
            return await _forward(up, body, wants_stream)

        try:
            result = await handle({**body, "stream": False}, up)
        except Exception:
            # Fail open: give the client Ollama's own answer.
            return await _forward(up, body, wants_stream)

        if wants_stream:
            return StreamingResponse(
                _as_sse(result), media_type="text/event-stream"
            )
        return JSONResponse(result)

    async def models(request: Request) -> Response:
        up: Upstream = request.app.state.upstream
        r = await up.passthrough("/v1/models")
        return Response(
            r.content, status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "ollama": ollama_url})

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.upstream = Upstream(ollama_url)
        try:
            yield
        finally:
            await app.state.upstream.aclose()

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
    return app


async def _forward(up: Upstream, body: dict[str, Any], wants_stream: bool) -> Response:
    """Pass a request straight to Ollama. Used when there's nothing to repair
    and as the fail-open path."""
    if wants_stream:
        async def gen():
            async with up._client.stream(  # noqa: SLF001 - deliberate reuse
                "POST", f"{up.base_url}/v1/chat/completions", json=body
            ) as r:
                async for chunk in r.aiter_raw():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")
    r = await up.chat_raw(body)
    return Response(
        r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


def _as_sse(result: dict[str, Any]):
    """Re-emit a completed response as a single OpenAI streaming chunk.

    v1 keeps this deliberately simple: one delta carrying the full message,
    then [DONE]. Token-by-token streaming under the grammar is a later concern;
    correctness of the call comes first.
    """
    try:
        choice = result["choices"][0]
        message = choice.get("message", {})
    except (KeyError, IndexError):
        yield b"data: [DONE]\n\n"
        return

    chunk = {
        "id": result.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion.chunk",
        "created": result.get("created", int(time.time())),
        "model": result.get("model", ""),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                },
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"
