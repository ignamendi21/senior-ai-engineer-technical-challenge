import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from magic_assistant.rules.chunker import RuleChunk, RulesChunker
from magic_assistant.rules.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from magic_assistant.rules.parser import ComprehensiveRulesParser

INDEX_SCHEMA_VERSION = 2
DEFAULT_INDEX_DIRECTORY = Path("data/index")
DEFAULT_PDF_PATH = Path("data/MagicCompRules 20260417.pdf")


class RuleIndexError(ValueError):
    pass


class RuleIndexManifest(BaseModel):
    schema_version: int
    embedding_model: str
    rules_version: str
    chunk_count: int
    embedding_dimension: int
    corpus_fingerprint: str
    embeddings_fingerprint: str


class DenseRuleIndex:
    def __init__(
        self,
        chunks: Sequence[RuleChunk],
        embeddings: NDArray[np.float32],
        manifest: RuleIndexManifest,
    ) -> None:
        self.chunks = list(chunks)
        self.embeddings = embeddings
        self.manifest = manifest

    @classmethod
    def build(cls, chunks: Sequence[RuleChunk], provider: EmbeddingProvider) -> "DenseRuleIndex":
        if not chunks:
            raise RuleIndexError("Cannot build an index without chunks")
        versions = {chunk.rules_version for chunk in chunks}
        if len(versions) != 1:
            raise RuleIndexError("All indexed chunks must have the same rules version")

        embeddings = cls._normalize_matrix(
            provider.encode_documents([chunk.text for chunk in chunks])
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise RuleIndexError("Embedding provider returned an incompatible document matrix")
        manifest = RuleIndexManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            embedding_model=provider.model_name,
            rules_version=next(iter(versions)),
            chunk_count=len(chunks),
            embedding_dimension=embeddings.shape[1],
            corpus_fingerprint=corpus_fingerprint(chunks),
            embeddings_fingerprint=embeddings_fingerprint(embeddings),
        )
        return cls(chunks, embeddings, manifest)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings, allow_pickle=False)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as file:
            for chunk in self.chunks:
                file.write(chunk.model_dump_json() + "\n")
        (directory / "manifest.json").write_text(
            self.manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_chunks: Sequence[RuleChunk],
        expected_model: str,
    ) -> "DenseRuleIndex":
        manifest_path = directory / "manifest.json"
        chunks_path = directory / "chunks.jsonl"
        embeddings_path = directory / "embeddings.npy"
        if not all(path.is_file() for path in (manifest_path, chunks_path, embeddings_path)):
            raise RuleIndexError(f"Incomplete rule index in {directory}")

        try:
            manifest = RuleIndexManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            persisted_chunks = [
                RuleChunk.model_validate_json(line)
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            embeddings = np.asarray(np.load(embeddings_path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError, ValidationError) as error:
            raise RuleIndexError(f"Could not load rule index from {directory}") from error

        expected_fingerprint = corpus_fingerprint(expected_chunks)
        expected_versions = {chunk.rules_version for chunk in expected_chunks}
        persisted_versions = {chunk.rules_version for chunk in persisted_chunks}
        mismatches = []
        if manifest.schema_version != INDEX_SCHEMA_VERSION:
            mismatches.append("schema version")
        if manifest.embedding_model != expected_model:
            mismatches.append("embedding model")
        if manifest.corpus_fingerprint != expected_fingerprint:
            mismatches.append("corpus fingerprint")
        if manifest.chunk_count != len(expected_chunks):
            mismatches.append("chunk count")
        if expected_versions != {manifest.rules_version}:
            mismatches.append("rules version")
        if persisted_versions != {manifest.rules_version}:
            mismatches.append("persisted chunk versions")
        if mismatches:
            raise RuleIndexError(
                f"Stale or incompatible rule index ({', '.join(mismatches)}); rebuild explicitly"
            )
        if corpus_fingerprint(persisted_chunks) != manifest.corpus_fingerprint:
            raise RuleIndexError("Persisted chunks do not match the index manifest")
        expected_shape = (manifest.chunk_count, manifest.embedding_dimension)
        if embeddings.shape != expected_shape:
            actual_shape = embeddings.shape
            message = (
                f"Embedding matrix shape {actual_shape} does not match manifest {expected_shape}"
            )
            raise RuleIndexError(message)
        if embeddings_fingerprint(embeddings) != manifest.embeddings_fingerprint:
            raise RuleIndexError("Embedding matrix does not match the index manifest")
        cls._validate_normalized_matrix(embeddings)
        return cls(persisted_chunks, embeddings, manifest)

    @staticmethod
    def _normalize_matrix(values: NDArray[np.float32]) -> NDArray[np.float32]:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2:
            raise RuleIndexError("Document embeddings must be a two-dimensional matrix")
        if not np.all(np.isfinite(matrix)):
            raise RuleIndexError("Document embeddings must contain only finite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuleIndexError("Document embeddings must not contain zero vectors")
        return matrix / norms

    @staticmethod
    def _validate_normalized_matrix(matrix: NDArray[np.float32]) -> None:
        if not np.all(np.isfinite(matrix)):
            raise RuleIndexError("Persisted embeddings must contain only finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise RuleIndexError("Persisted embeddings must be normalized")


def embeddings_fingerprint(embeddings: NDArray[np.float32]) -> str:
    matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def corpus_fingerprint(chunks: Sequence[RuleChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        serialized = json.dumps(
            chunk.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(serialized.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate the local rules embedding index"
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    parsed = ComprehensiveRulesParser().parse_pdf(arguments.pdf)
    chunks = RulesChunker().chunk(parsed)
    provider = SentenceTransformerEmbeddingProvider(arguments.model, show_progress=True)
    manifest_path = arguments.index_dir / "manifest.json"
    if manifest_path.is_file() and not arguments.rebuild:
        index = DenseRuleIndex.load(
            arguments.index_dir,
            expected_chunks=chunks,
            expected_model=provider.model_name,
        )
        print(f"Validated {index.manifest.chunk_count} chunks in {arguments.index_dir}")
        return 0

    index = DenseRuleIndex.build(chunks, provider)
    index.save(arguments.index_dir)
    print(
        f"Built {index.manifest.chunk_count} chunks with "
        f"{index.manifest.embedding_dimension}-dimensional embeddings in {arguments.index_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
