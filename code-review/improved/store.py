import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings

from improved.errors import IncompatibleIndexError, VectorStoreError
from improved.models import DocumentChunk, RetrievedChunk

INDEX_SCHEMA_VERSION = "1"


class VectorStore(Protocol):
    def upsert(
        self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]: ...


class ChromaVectorStore:
    def __init__(
        self,
        persistence_directory: Path,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        expected_metadata = {
            "embedding_model": embedding_model,
            "index_schema_version": INDEX_SCHEMA_VERSION,
        }
        try:
            self._client = chromadb.PersistentClient(
                path=persistence_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                collection_name,
                embedding_function=None,
            )
            current_metadata = self._collection.metadata or {}
            if self._collection.count() > 0 and any(
                current_metadata.get(key) != value for key, value in expected_metadata.items()
            ):
                raise IncompatibleIndexError(
                    "Persistent index embedding model or schema is incompatible"
                )
            if self._collection.count() == 0 and current_metadata != expected_metadata:
                self._collection.modify(metadata=expected_metadata)
        except IncompatibleIndexError:
            raise
        except Exception as error:
            raise VectorStoreError("Could not initialize persistent vector storage") from error

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")
        if not chunks:
            return
        metadatas = [
            {
                "source_id": chunk.source_id,
                "chunk_index": chunk.chunk_index,
                "metadata_json": json.dumps(chunk.metadata, sort_keys=True),
            }
            for chunk in chunks
        ]
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=[list(embedding) for embedding in embeddings],
                metadatas=metadatas,
            )
        except Exception as error:
            raise VectorStoreError("Could not upsert document chunks") from error

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            identifiers = results["ids"][0]
            documents = (results["documents"] or [[]])[0]
            metadatas = (results["metadatas"] or [[]])[0]
            distances = (results["distances"] or [[]])[0]
        except Exception as error:
            raise VectorStoreError("Could not query vector storage") from error

        retrieved = []
        for identifier, document, metadata, distance in zip(
            identifiers, documents, metadatas, distances, strict=True
        ):
            if document is None or metadata is None or distance is None:
                raise VectorStoreError("Vector storage returned incomplete retrieval data")
            retrieved.append(
                RetrievedChunk(
                    chunk_id=identifier,
                    source_id=str(metadata["source_id"]),
                    text=document,
                    chunk_index=int(metadata["chunk_index"]),
                    metadata=json.loads(str(metadata["metadata_json"])),
                    distance=float(distance),
                )
            )
        return retrieved
