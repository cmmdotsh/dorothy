"""Async concurrent chunk embedding via LMStudio.

Sends batched embedding requests concurrently. A semaphore caps
in-flight requests.
"""

import asyncio

import httpx
import structlog

from src.claim_graph.models import Chunk

logger = structlog.get_logger(__name__)


async def _embed_batch(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    batch: list[Chunk],
    endpoint: str,
    model: str,
) -> list[Chunk]:
    """Embed a batch of chunks with retry."""
    async with semaphore:
        for attempt in range(2):
            try:
                response = await client.post(
                    endpoint,
                    json={"model": model, "input": [c.text for c in batch]},
                )
                response.raise_for_status()
                data = response.json()
                for chunk, item in zip(batch, data["data"]):
                    chunk.embedding = item["embedding"]
                return batch
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as e:
                if attempt == 0:
                    logger.debug("chunk_batch_embed_retry", batch_size=len(batch), error=str(e))
                    await asyncio.sleep(0.5)
                else:
                    logger.warning("chunk_batch_embed_failed", batch_size=len(batch), error=str(e))
    return batch


async def embed_chunks(
    chunks: list[Chunk],
    base_url: str,
    model: str,
    concurrency: int = 32,
    timeout: float = 60.0,
    batch_size: int = 32,
) -> list[Chunk]:
    """Embed all chunks concurrently with bounded parallelism.

    Chunks are grouped into batches and sent as concurrent requests.
    Returns the same chunks with embeddings set (None on failure).
    """
    if not chunks:
        return chunks

    endpoint = f"{base_url.rstrip('/')}/v1/embeddings"
    semaphore = asyncio.Semaphore(concurrency)

    # Split into batches
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _embed_batch(client, semaphore, batch, endpoint, model)
            for batch in batches
        ]
        await asyncio.gather(*tasks)

    embedded = sum(1 for c in chunks if c.embedding is not None)
    logger.info(
        "chunk_embedding_complete",
        total=len(chunks),
        embedded=embedded,
        failed=len(chunks) - embedded,
    )
    return chunks


def embed_chunks_sync(
    chunks: list[Chunk],
    base_url: str,
    model: str,
    concurrency: int = 32,
    timeout: float = 60.0,
    batch_size: int = 32,
) -> list[Chunk]:
    """Synchronous wrapper for embed_chunks."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = embed_chunks(chunks, base_url, model, concurrency, timeout, batch_size)

    if loop and loop.is_running():
        # Already in an async context — run in a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
