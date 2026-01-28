#!/usr/bin/env python3
"""
Import OpenSearch data from JSONL files.

Usage:
    python -m scripts.import_data
    python -m scripts.import_data --input-dir ./data
    python -m scripts.import_data --clear  # Clear existing indices first
"""

import argparse
import json
from pathlib import Path

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
from rich.console import Console
from rich.progress import Progress

from src.config import config

console = Console()


def import_index(client: OpenSearch, index_name: str, input_file: Path, clear: bool = False) -> int:
    """Import documents from JSONL file into an index."""

    # Read mapping and documents
    mapping = None
    docs = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#MAPPING#"):
                mapping_data = json.loads(line[9:])  # Skip #MAPPING# prefix
                mapping = mapping_data.get("_mapping", {})
            else:
                docs.append(json.loads(line))

    if not docs:
        return 0

    # Handle index creation/clearing
    index_exists = client.indices.exists(index=index_name)

    if index_exists and clear:
        console.print(f"    [yellow]Deleting existing index...[/yellow]")
        client.indices.delete(index=index_name)
        index_exists = False

    if not index_exists and mapping:
        console.print(f"    [dim]Creating index with mapping...[/dim]")
        # Build index settings
        body = {"mappings": mapping}

        # Add k-NN settings if this is an articles index with embeddings
        if "embedding" in str(mapping):
            body["settings"] = {
                "index": {
                    "knn": True
                }
            }

        client.indices.create(index=index_name, body=body)

    # Bulk import documents
    def generate_actions():
        for doc in docs:
            yield {
                "_index": index_name,
                "_id": doc["_id"],
                "_source": doc["_source"]
            }

    success, errors = bulk(client, generate_actions(), raise_on_error=False)

    if errors:
        console.print(f"    [yellow]Warning: {len(errors)} errors during import[/yellow]")

    return success


def main():
    parser = argparse.ArgumentParser(description="Import OpenSearch data from JSONL")
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="data",
        help="Input directory with JSONL files (default: data/)",
    )
    parser.add_argument(
        "--clear",
        "-c",
        action="store_true",
        help="Clear existing indices before importing",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        return

    # Find all JSONL files
    jsonl_files = list(input_dir.glob("dorothy-*.jsonl"))

    if not jsonl_files:
        console.print(f"[red]No dorothy-*.jsonl files found in {input_dir}[/red]")
        return

    # Connect to OpenSearch
    auth = None
    if config.opensearch.username and config.opensearch.password:
        auth = (config.opensearch.username, config.opensearch.password)

    client = OpenSearch(
        hosts=[{"host": config.opensearch.host, "port": config.opensearch.port}],
        http_auth=auth,
        use_ssl=config.opensearch.use_ssl,
        verify_certs=config.opensearch.verify_certs,
    )

    # Verify connection
    try:
        info = client.info()
        console.print(f"[bold blue]Dorothy Data Import[/bold blue]")
        console.print(f"OpenSearch: {config.opensearch.host}:{config.opensearch.port} (v{info['version']['number']})")
        console.print(f"Input directory: {input_dir.absolute()}")
        if args.clear:
            console.print("[yellow]Mode: Clear and reimport[/yellow]")
        console.print()
    except Exception as e:
        console.print(f"[red]Failed to connect to OpenSearch: {e}[/red]")
        return

    console.print(f"Found {len(jsonl_files)} files to import:\n")

    total_docs = 0
    for jsonl_file in sorted(jsonl_files):
        index_name = jsonl_file.stem  # filename without .jsonl

        # Count lines (minus mapping line)
        with open(jsonl_file) as f:
            doc_count = sum(1 for line in f if line.strip() and not line.startswith("#MAPPING#"))

        console.print(f"  [dim]{index_name}[/dim] ({doc_count} docs)...", end=" ")
        imported = import_index(client, index_name, jsonl_file, clear=args.clear)
        console.print(f"[green]✓ {imported} imported[/green]")
        total_docs += imported

    console.print(f"\n[bold green]Imported {total_docs} documents[/bold green]")


if __name__ == "__main__":
    main()
