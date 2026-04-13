"""Async concurrent chunk embedding via Ollama.

Sends individual embedding requests concurrently to sidestep Ollama's
batch context window bug (all inputs in a batch count against a single
512-token context). A semaphore caps in-flight requests.
"""

import asyncio

import httpx
import structlog

from src.claim_graph.models import Chunk

logger = structlog.get_logger(__name__)


async def _embed_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    chunk: Chunk,
    endpoint: str,
    model: str,
) -> Chunk:
    """Embed a single chunk with retry."""
    async with semaphore:
        for attempt in range(2):
            try:
                response = await client.post(
                    endpoint,
                    json={"model": model, "input": [chunk.text]},
                )
                response.raise_for_status()
                data = response.json()
                chunk.embedding = data["data"][0]["embedding"]
                return chunk
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as e:
                if attempt == 0:
                    logger.debug(
                        "chunk_embed_retry",
                        source=chunk.source_name,
                        chunk_index=chunk.chunk_index,
                        error=str(e),
                    )
                    await asyncio.sleep(0.5)
                else:
                    logger.warning(
                        "chunk_embed_failed",
                        source=chunk.source_name,
                        chunk_index=chunk.chunk_index,
                        error=str(e),
                    )
    return chunk


async def embed_chunks(
    chunks: list[Chunk],
    base_url: str,
    model: str,
    concurrency: int = 8,
    timeout: float = 60.0,
) -> list[Chunk]:
    """Embed all chunks concurrently with bounded parallelism.

    Each chunk is sent as an individual request to avoid Ollama's batch
    context window limitation. Returns the same chunks with embeddings set
    (None on failure).
    """
    if not chunks:
        return chunks

    endpoint = f"{base_url.rstrip('/')}/v1/embeddings"
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _embed_one(client, semaphore, chunk, endpoint, model)
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks)

    embedded = sum(1 for c in results if c.embedding is not None)
    logger.info(
        "chunk_embedding_complete",
        total=len(chunks),
        embedded=embedded,
        failed=len(chunks) - embedded,
    )
    return list(results)


def embed_chunks_sync(
    chunks: list[Chunk],
    base_url: str,
    model: str,
    concurrency: int = 8,
    timeout: float = 60.0,
) -> list[Chunk]:
    """Synchronous wrapper for embed_chunks."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — run in a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                embed_chunks(chunks, base_url, model, concurrency, timeout),
            )
            return future.result()
    else:
        return asyncio.run(
            embed_chunks(chunks, base_url, model, concurrency, timeout)
        )
