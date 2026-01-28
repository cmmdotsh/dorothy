#!/usr/bin/env python3
"""
Export OpenSearch data to JSONL files for portability.

Usage:
    python -m scripts.export_data
    python -m scripts.export_data --output-dir ./data
"""

import argparse
import json
from pathlib import Path

from opensearchpy import OpenSearch
from rich.console import Console

from src.config import config

console = Console()


def export_index(client: OpenSearch, index_name: str, output_file: Path) -> int:
    """Export all documents from an index to JSONL file."""

    # Get index mapping first
    mapping = client.indices.get_mapping(index=index_name)

    # Use scroll API for large exports
    docs_exported = 0

    with open(output_file, 'w') as f:
        # Write mapping as first line (prefixed with #MAPPING#)
        mapping_line = {"_mapping": mapping[index_name]["mappings"]}
        f.write("#MAPPING#" + json.dumps(mapping_line) + "\n")

        # Initial search with scroll
        resp = client.search(
            index=index_name,
            scroll="2m",
            size=500,
            body={"query": {"match_all": {}}}
        )

        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

        while hits:
            for hit in hits:
                doc = {
                    "_id": hit["_id"],
                    "_source": hit["_source"]
                }
                f.write(json.dumps(doc) + "\n")
                docs_exported += 1

            # Get next batch
            resp = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = resp["_scroll_id"]
            hits = resp["hits"]["hits"]

        # Clear scroll
        client.clear_scroll(scroll_id=scroll_id)

    return docs_exported


def main():
    parser = argparse.ArgumentParser(description="Export OpenSearch data to JSONL")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data",
        help="Output directory for JSONL files (default: data/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    console.print("[bold blue]Dorothy Data Export[/bold blue]")
    console.print(f"Output directory: {output_dir.absolute()}\n")

    # Get all dorothy indices
    indices = list(client.indices.get(index="dorothy-*").keys())

    if not indices:
        console.print("[red]No dorothy-* indices found![/red]")
        return

    console.print(f"Found {len(indices)} indices to export:\n")

    total_docs = 0
    for index_name in sorted(indices):
        # Get doc count
        count = client.count(index=index_name)["count"]
        output_file = output_dir / f"{index_name}.jsonl"

        console.print(f"  [dim]{index_name}[/dim] ({count} docs)...", end=" ")
        exported = export_index(client, index_name, output_file)
        console.print(f"[green]✓ {output_file.name}[/green]")
        total_docs += exported

    console.print(f"\n[bold green]Exported {total_docs} documents[/bold green]")
    console.print(f"\nFiles ready in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
