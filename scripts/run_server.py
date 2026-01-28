#!/usr/bin/env python3
"""
Dorothy Web Server

Usage:
    python -m scripts.run_server                    # Start server on localhost:8000
    python -m scripts.run_server --port 8080        # Different port
    python -m scripts.run_server --reload           # Auto-reload on changes
    python -m scripts.run_server --host 0.0.0.0     # Accept external connections
"""

import argparse
import sys

import uvicorn
from rich.console import Console

console = Console()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dorothy Web Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        "-r",
        action="store_true",
        help="Enable auto-reload on code changes",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    console.print("[bold blue]Dorothy - Newspaper of Averages[/bold blue]")
    console.print(f"Starting server at http://{args.host}:{args.port}")
    console.print()

    if args.reload:
        console.print("[dim]Auto-reload enabled[/dim]")

    try:
        uvicorn.run(
            "src.web:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
