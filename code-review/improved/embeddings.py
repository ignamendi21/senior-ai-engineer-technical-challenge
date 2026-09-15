from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI, OpenAIError

from improved.errors import EmbeddingProviderError


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, client: OpenAI, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=list(texts),
            )
        except OpenAIError as error:
            raise EmbeddingProviderError("Embedding provider request failed") from error
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise EmbeddingProviderError("Embedding provider returned an unexpected result count")
        return [item.embedding for item in ordered]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0]
