from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def encode_query(self, text: str) -> NDArray[np.float32]: ...


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        show_progress: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._show_progress = show_progress
        self._model: Any = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        prefixed = [f"passage: {text}" for text in texts]
        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=self._show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> NDArray[np.float32]:
        embedding = self._model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32)
