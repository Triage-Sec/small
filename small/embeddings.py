"""Embedding provider interfaces for offline analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed_single(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def dimension(self) -> int: ...

    def max_tokens(self) -> int: ...

    def normalize(self) -> bool: ...

    def model_id(self) -> str: ...


@dataclass
class HuggingFaceEmbeddingProvider:
    model_name: str
    device: str = "auto"
    batch_size: int = 32
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "HuggingFace embedding provider requires 'sentence-transformers' and 'torch'."
            ) from exc

        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device

        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            trust_remote_code=self.trust_remote_code,
        )

    def embed_single(self, text: str) -> list[float]:
        return list(self._model.encode([text], normalize_embeddings=True)[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True
        )
        return [list(vec) for vec in vectors]

    def dimension(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 0

    def max_tokens(self) -> int:
        return 8192

    def normalize(self) -> bool:
        return True

    def model_id(self) -> str:
        return self.model_name


@dataclass
class OllamaEmbeddingProvider:
    model: str
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 30

    def _client(self):
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("Ollama embedding provider requires 'httpx'.") from exc
        return httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds)

    def embed_single(self, text: str) -> list[float]:
        with self._client() as client:
            response = client.post(
                "/api/embeddings", json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            return list(response.json()["embedding"])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_single(text) for text in texts]

    def dimension(self) -> int:
        return 0

    def max_tokens(self) -> int:
        return 8192

    def normalize(self) -> bool:
        return True

    def model_id(self) -> str:
        return self.model


@dataclass
class OpenAIEmbeddingProvider:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    dimensions: int | None = None

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise ImportError("OpenAI embedding provider requires 'openai'.") from exc
        return openai.OpenAI()

    def embed_single(self, text: str) -> list[float]:
        client = self._client()
        response = client.embeddings.create(
            model=self.model, input=[text], dimensions=self.dimensions
        )
        return list(response.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        response = client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dimensions
        )
        return [list(item.embedding) for item in response.data]

    def dimension(self) -> int:
        return 0

    def max_tokens(self) -> int:
        return 8192

    def normalize(self) -> bool:
        return False

    def model_id(self) -> str:
        return self.model


@dataclass
class VoyageEmbeddingProvider:
    model: str
    api_key_env: str = "VOYAGE_API_KEY"

    def _client(self):
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError("Voyage embedding provider requires 'voyageai'.") from exc
        return voyageai.Client()

    def embed_single(self, text: str) -> list[float]:
        client = self._client()
        response = client.embed([text], model=self.model)
        return list(response.embeddings[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        response = client.embed(texts, model=self.model)
        return [list(vec) for vec in response.embeddings]

    def dimension(self) -> int:
        return 0

    def max_tokens(self) -> int:
        return 8192

    def normalize(self) -> bool:
        return False

    def model_id(self) -> str:
        return self.model


@dataclass
class CohereEmbeddingProvider:
    model: str
    api_key_env: str = "COHERE_API_KEY"

    def _client(self):
        try:
            import cohere
        except ImportError as exc:
            raise ImportError("Cohere embedding provider requires 'cohere'.") from exc
        return cohere.Client()

    def embed_single(self, text: str) -> list[float]:
        client = self._client()
        response = client.embed(texts=[text], model=self.model)
        return list(response.embeddings[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        response = client.embed(texts=texts, model=self.model)
        return [list(vec) for vec in response.embeddings]

    def dimension(self) -> int:
        return 0

    def max_tokens(self) -> int:
        return 8192

    def normalize(self) -> bool:
        return False

    def model_id(self) -> str:
        return self.model
