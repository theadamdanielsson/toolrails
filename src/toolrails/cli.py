"""`toolrails` command line: start the proxy."""

from __future__ import annotations

import argparse
import os

from . import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="toolrails",
        description="An OpenAI-compatible proxy that guarantees valid tool "
        "calls from local models served by Ollama.",
    )
    parser.add_argument(
        "--ollama",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Base URL of the Ollama server (default: %(default)s).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Address to bind (default: %(default)s)."
    )
    parser.add_argument(
        "--port", type=int, default=11500, help="Port to listen on (default: %(default)s)."
    )
    parser.add_argument("--version", action="version", version=f"toolrails {__version__}")
    args = parser.parse_args(argv)

    import uvicorn

    from .app import create_app

    app = create_app(args.ollama)
    print(
        f"toolrails {__version__} → proxying {args.ollama}\n"
        f"point your agent's base URL at http://{args.host}:{args.port}/v1"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
