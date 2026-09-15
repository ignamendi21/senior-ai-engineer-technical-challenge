class RagError(RuntimeError):
    pass


class ConfigurationError(RagError):
    pass


class EmbeddingProviderError(RagError):
    pass


class VectorStoreError(RagError):
    pass


class IncompatibleIndexError(VectorStoreError):
    pass


class GenerationError(RagError):
    pass
