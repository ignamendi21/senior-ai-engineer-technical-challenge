import hashlib
from collections.abc import Sequence

from improved.models import DocumentChunk, SourceDocument


def chunk_documents(
    documents: Sequence[SourceDocument],
    *,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be nonnegative and lower than chunk_size")
    chunks = []
    step = chunk_size - overlap
    for document in documents:
        for chunk_index, start in enumerate(range(0, len(document.text), step)):
            text = document.text[start : start + chunk_size]
            chunks.append(
                DocumentChunk(
                    chunk_id=stable_chunk_id(document.source_id, chunk_index, text),
                    source_id=document.source_id,
                    text=text,
                    chunk_index=chunk_index,
                    metadata=document.metadata,
                )
            )
            if start + chunk_size >= len(document.text):
                break
    return chunks


def stable_chunk_id(source_id: str, chunk_index: int, text: str) -> str:
    identity = f"{source_id}\0{chunk_index}\0{text}".encode()
    return hashlib.sha256(identity).hexdigest()
