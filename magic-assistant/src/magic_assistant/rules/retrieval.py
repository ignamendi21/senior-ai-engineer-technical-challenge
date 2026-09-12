import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from magic_assistant.rules.chunker import RuleChunk
from magic_assistant.rules.embeddings import EmbeddingProvider
from magic_assistant.rules.index import DenseRuleIndex, RuleIndexError

DEFAULT_RRF_CONSTANT = 60
_RULE_ID_RE = re.compile(r"(?<![\d.])\d{3}(?:\.\d+)*(?:[a-z])?(?![\d.])", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\d{3}(?:\.\d+)*(?:[a-z])?|[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
RetrievalMethod = Literal["exact", "lexical", "semantic", "hybrid", "glossary_expansion"]


class RuleEvidence(BaseModel):
    chunk_id: str
    rule_ids: list[str] = Field(default_factory=list)
    root_rule_id: str | None = None
    title: str | None = None
    term: str | None = None
    text: str
    page_start: int
    page_end: int
    rules_version: str
    score: float
    retrieval_methods: list[RetrievalMethod]
    document_type: Literal["rule", "glossary"]


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, constant: int = DEFAULT_RRF_CONSTANT
) -> list[tuple[str, float]]:
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, identifier in enumerate(ranking, 1):
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] += 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class RulesKnowledgeBase:
    def __init__(
        self,
        chunks: Sequence[RuleChunk],
        dense_index: DenseRuleIndex,
        embedding_provider: EmbeddingProvider,
        *,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
    ) -> None:
        if rrf_constant <= 0:
            raise ValueError("RRF constant must be positive")
        self._chunks = list(chunks)
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        if len(self._chunk_by_id) != len(self._chunks):
            raise ValueError("Chunk identifiers must be unique")
        if [chunk.chunk_id for chunk in dense_index.chunks] != [
            chunk.chunk_id for chunk in self._chunks
        ]:
            raise RuleIndexError("Dense index chunk order does not match the retrieval corpus")
        if dense_index.manifest.embedding_model != embedding_provider.model_name:
            raise RuleIndexError("Embedding provider does not match the dense index model")

        self._dense_index = dense_index
        self._embedding_provider = embedding_provider
        self._rrf_constant = rrf_constant
        self._bm25 = BM25Okapi([tokenize(self._searchable_text(chunk)) for chunk in self._chunks])
        self._rule_index: dict[str, list[str]] = defaultdict(list)
        self._root_index: dict[str, list[str]] = defaultdict(list)
        self._section_index: dict[str, list[str]] = defaultdict(list)
        self._term_index: dict[str, list[str]] = defaultdict(list)
        for chunk in self._chunks:
            for rule_id in chunk.rule_ids:
                self._rule_index[rule_id].append(chunk.chunk_id)
            if chunk.root_rule_id:
                self._root_index[chunk.root_rule_id].append(chunk.chunk_id)
            if chunk.section_id and chunk.document_type == "rule":
                self._section_index[chunk.section_id].append(chunk.chunk_id)
            if chunk.term:
                self._term_index[self._normalize_term(chunk.term)].append(chunk.chunk_id)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def get_rule(self, rule_id: str) -> list[RuleEvidence]:
        normalized = rule_id.strip().lower().rstrip(".")
        if not _RULE_ID_RE.fullmatch(normalized):
            return []
        if "." not in normalized:
            chunk_ids = self._section_index.get(normalized, [])
        elif normalized in self._root_index:
            chunk_ids = self._root_index[normalized]
        else:
            chunk_ids = self._rule_index.get(normalized, [])
        return [
            self._to_evidence(self._chunk_by_id[chunk_id], 1.0, ["exact"])
            for chunk_id in dict.fromkeys(chunk_ids)
        ]

    def search(self, query: str, top_k: int = 5) -> list[RuleEvidence]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        exact = self._exact_matches(query)
        excluded = {evidence.chunk_id for evidence in exact}
        lexical_ids = self._lexical_ranking(query, len(self._chunks))
        semantic_ids = self._semantic_ranking(query, len(self._chunks))
        lexical_set = set(lexical_ids)
        semantic_set = set(semantic_ids)
        fused = reciprocal_rank_fusion(
            [lexical_ids, semantic_ids],
            constant=self._rrf_constant,
        )

        ranked = list(exact)
        for chunk_id, score in fused:
            if chunk_id in excluded:
                continue
            methods: list[RetrievalMethod] = []
            if chunk_id in lexical_set:
                methods.append("lexical")
            if chunk_id in semantic_set:
                methods.append("semantic")
            if len(methods) == 2:
                methods.append("hybrid")
            ranked.append(self._to_evidence(self._chunk_by_id[chunk_id], score, methods))
            excluded.add(chunk_id)

        return self._expand_glossary(ranked, top_k)

    def _exact_matches(self, query: str) -> list[RuleEvidence]:
        matches: list[RuleEvidence] = []
        seen: set[str] = set()
        for rule_id in _RULE_ID_RE.findall(query):
            for evidence in self.get_rule(rule_id):
                if evidence.chunk_id not in seen:
                    matches.append(evidence)
                    seen.add(evidence.chunk_id)
        term_chunk_ids = self._term_index.get(self._normalize_term(query), [])
        for chunk_id in term_chunk_ids:
            if chunk_id not in seen:
                matches.append(self._to_evidence(self._chunk_by_id[chunk_id], 1.0, ["exact"]))
                seen.add(chunk_id)
        return matches

    def _lexical_ranking(self, query: str, limit: int) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )
        return [self._chunks[index].chunk_id for index in ranked_indices[:limit]]

    def _semantic_ranking(self, query: str, limit: int) -> list[str]:
        vector = np.asarray(self._embedding_provider.encode_query(query), dtype=np.float32).reshape(
            -1
        )
        norm = float(np.linalg.norm(vector))
        if (
            not np.all(np.isfinite(vector))
            or not np.isfinite(norm)
            or norm == 0
            or vector.shape[0] != self._dense_index.embeddings.shape[1]
        ):
            raise RuleIndexError("Query embedding is incompatible with the dense index")
        scores = self._dense_index.embeddings @ (vector / norm)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )
        return [self._chunks[index].chunk_id for index in ranked_indices[:limit]]

    def _expand_glossary(self, ranked: Sequence[RuleEvidence], top_k: int) -> list[RuleEvidence]:
        results: list[RuleEvidence] = []
        seen: set[str] = set()
        for evidence in ranked:
            if evidence.chunk_id in seen:
                continue
            results.append(evidence)
            seen.add(evidence.chunk_id)
            if len(results) >= top_k:
                break
            chunk = self._chunk_by_id[evidence.chunk_id]
            if chunk.document_type != "glossary":
                continue
            for related_rule_id in chunk.related_rule_ids:
                for related in self.get_rule(related_rule_id):
                    if related.chunk_id in seen:
                        continue
                    related.score = evidence.score
                    related.retrieval_methods = ["glossary_expansion"]
                    results.append(related)
                    seen.add(related.chunk_id)
                    if len(results) >= top_k:
                        break
                if len(results) >= top_k:
                    break
        return results[:top_k]

    @staticmethod
    def _normalize_term(term: str) -> str:
        normalized = unicodedata.normalize("NFKC", term).casefold().strip()
        normalized = normalized.strip(
            "".join(
                character for character in normalized if unicodedata.category(character)[0] == "P"
            )
        )
        return " ".join(normalized.split())

    @staticmethod
    def _searchable_text(chunk: RuleChunk) -> str:
        metadata = " ".join(
            value
            for value in [chunk.title, chunk.term, chunk.root_rule_id, *chunk.rule_ids]
            if value
        )
        return f"{metadata} {chunk.text}"

    @staticmethod
    def _to_evidence(
        chunk: RuleChunk,
        score: float,
        methods: list[RetrievalMethod],
    ) -> RuleEvidence:
        return RuleEvidence(
            chunk_id=chunk.chunk_id,
            rule_ids=chunk.rule_ids,
            root_rule_id=chunk.root_rule_id,
            title=chunk.title,
            term=chunk.term,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            rules_version=chunk.rules_version,
            score=score,
            retrieval_methods=methods,
            document_type=chunk.document_type,
        )
