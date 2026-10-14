"""벡터 저장소. 로컬 Chroma(파일 저장)를 쓴다.

파이프라인은 add·search 두 메서드만 쓰므로, 운영에서는 같은 메서드를 가진
Azure AI Search 구현으로 바꿔 끼우면 된다.
"""

from dataclasses import dataclass

import chromadb

from .chunk import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # 코사인 유사도(1에 가까울수록 비슷함)


class ChromaStore:
    def __init__(self, path: str, collection: str = "policies"):
        self._client = chromadb.PersistentClient(path=path)
        self._name = collection
        self._collection = self._open()

    def _open(self):
        # 임베딩은 직접 넣으므로 Chroma 기본 임베딩 함수는 쓰지 않는다
        return self._client.get_or_create_collection(
            self._name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._open()

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.body for c in chunks],
            metadatas=[{"title": c.title, "doc": c.doc} for c in chunks],
        )

    def count(self) -> int:
        return self._collection.count()

    def search(self, vector: list[float], k: int, doc: str | None = None) -> list[Hit]:
        """doc을 주면 그 규정 문서 안에서만 찾는다."""
        r = self._collection.query(query_embeddings=[vector], n_results=k, where={"doc": doc} if doc else None)
        return [
            Hit(Chunk(id=i, title=m["title"], body=d), 1 - dist)
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
