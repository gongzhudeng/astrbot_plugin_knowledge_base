"""Hybrid retrieval orchestration for the legacy knowledge-base plugin."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.api import logger

from ..vector_store.base import Document, VectorDBBase

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class SparseHit:
    document_id: str
    text_content: str
    metadata: dict[str, Any]
    score: float


class SparseIndex:
    """Small persistent BM25 index independent from the vector backend."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    async def index_documents(
        self,
        collection_name: str,
        documents: Sequence[Document],
        document_ids: Sequence[str],
    ) -> None:
        pairs = [
            (str(document_id), document)
            for document_id, document in zip(document_ids, documents)
            if document_id is not None
        ]
        if not pairs:
            return
        await asyncio.to_thread(self._index_documents_sync, collection_name, pairs)

    async def search(
        self,
        collection_name: str,
        query: str,
        top_k: int,
    ) -> list[SparseHit]:
        if top_k <= 0:
            return []
        return await asyncio.to_thread(
            self._search_sync,
            collection_name,
            query,
            top_k,
        )

    async def delete_collection(self, collection_name: str) -> None:
        await asyncio.to_thread(self._delete_collection_sync, collection_name)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize_sync(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sparse_documents (
                    collection_name TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    text_content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (collection_name, document_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sparse_documents_collection
                ON sparse_documents (collection_name)
                """
            )

    def _index_documents_sync(
        self,
        collection_name: str,
        pairs: Sequence[tuple[str, Document]],
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO sparse_documents
                    (collection_name, document_id, text_content, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        collection_name,
                        document_id,
                        document.text_content,
                        json.dumps(document.metadata, ensure_ascii=False),
                    )
                    for document_id, document in pairs
                ],
            )

    def _search_sync(
        self,
        collection_name: str,
        query: str,
        top_k: int,
    ) -> list[SparseHit]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, text_content, metadata_json
                FROM sparse_documents
                WHERE collection_name = ?
                """,
                (collection_name,),
            ).fetchall()

        if not rows:
            return []

        tokenized_documents = [_tokenize(row[1]) for row in rows]
        document_frequency = Counter(
            token for tokens in tokenized_documents for token in set(tokens)
        )
        average_length = sum(map(len, tokenized_documents)) / len(tokenized_documents)
        if average_length <= 0:
            return []

        query_terms = set(query_tokens)
        document_count = len(rows)
        scored_hits: list[SparseHit] = []
        for row, tokens in zip(rows, tokenized_documents):
            if not tokens:
                continue
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                frequency_in_documents = document_frequency[term]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - frequency_in_documents + 0.5)
                    / (frequency_in_documents + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * len(tokens) / average_length
                )
                score += inverse_document_frequency * (frequency * 2.5) / denominator

            if score <= 0:
                continue
            try:
                metadata = json.loads(row[2])
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            scored_hits.append(
                SparseHit(
                    document_id=row[0],
                    text_content=row[1],
                    metadata=metadata,
                    score=score,
                )
            )

        scored_hits.sort(key=lambda hit: hit.score, reverse=True)
        return scored_hits[:top_k]

    def _delete_collection_sync(self, collection_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM sparse_documents WHERE collection_name = ?",
                (collection_name,),
            )


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.lower()):
        value = match.group(0)
        if value[0] <= "\u007f":
            tokens.append(value)
            continue
        tokens.extend(value)
        tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
    return tokens


def _document_key(document: Document) -> str:
    if document.id is not None:
        return str(document.id)
    digest = hashlib.sha1(document.text_content.encode("utf-8")).hexdigest()
    return f"content:{digest}"


class HybridSearchService:
    """Preserve the vector-store API while adding optional hybrid retrieval."""

    def __init__(
        self,
        vector_db: VectorDBBase,
        index_path: str | Path,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.vector_db = vector_db
        self.config = config or {}
        self.sparse_index = SparseIndex(index_path)

    async def initialize(self) -> None:
        await self.sparse_index.initialize()

    async def add_documents(
        self,
        collection_name: str,
        documents: Sequence[Document],
    ) -> list[str]:
        document_ids = await self.vector_db.add_documents(
            collection_name,
            list(documents),
        )
        if document_ids and len(document_ids) == len(documents):
            try:
                await self.sparse_index.index_documents(
                    collection_name,
                    documents,
                    document_ids,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "知识库向量写入成功，但 BM25 索引更新失败，将继续使用向量检索: %s",
                    exc,
                    exc_info=True,
                )
        return document_ids

    async def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        if top_k <= 0:
            return []

        dense_top_k = max(
            top_k,
            _positive_int(
                self.config.get("dense_top_k"),
                max(20, top_k * 4),
            ),
        )
        sparse_top_k = max(
            top_k,
            _positive_int(
                self.config.get("sparse_top_k"),
                max(20, top_k * 4),
            ),
        )
        fusion_top_k = max(
            top_k,
            _positive_int(
                self.config.get("fusion_top_k"),
                max(top_k * 2, top_k),
            ),
        )

        dense_results = await self.vector_db.search(
            collection_name,
            query,
            top_k=dense_top_k,
        )
        try:
            sparse_results = await self.sparse_index.search(
                collection_name,
                query,
                sparse_top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "知识库 '%s' 的 BM25 检索失败，将回退到向量检索: %s",
                collection_name,
                exc,
                exc_info=True,
            )
            return _deduplicate_results(dense_results, top_k)
        if not sparse_results:
            return _deduplicate_results(dense_results, top_k)

        dense_by_id = {
            _document_key(document): (document, float(score))
            for document, score in dense_results
        }
        sparse_by_id = {str(hit.document_id): hit for hit in sparse_results}
        dense_ranks = {
            _document_key(document): rank
            for rank, (document, _) in enumerate(dense_results, 1)
        }
        sparse_ranks = {
            str(hit.document_id): rank for rank, hit in enumerate(sparse_results, 1)
        }

        ranked_ids = sorted(
            set(dense_by_id) | set(sparse_by_id),
            key=lambda document_id: (
                -_rrf_score(
                    dense_ranks.get(document_id),
                    sparse_ranks.get(document_id),
                ),
                dense_ranks.get(document_id, math.inf),
                sparse_ranks.get(document_id, math.inf),
                document_id,
            ),
        )[:fusion_top_k]

        results: list[tuple[Document, float]] = []
        for document_id in ranked_ids:
            dense_item = dense_by_id.get(document_id)
            sparse_item = sparse_by_id.get(document_id)
            if dense_item is not None:
                document, dense_score = dense_item
                result_score = dense_score
                score_type = "dense_compatibility"
            elif sparse_item is not None:
                document = Document(
                    id=sparse_item.document_id,
                    text_content=sparse_item.text_content,
                    metadata=dict(sparse_item.metadata),
                )
                dense_score = None
                result_score = sparse_item.score
                score_type = "sparse_bm25"
            else:
                continue

            hybrid_score = _rrf_score(
                dense_ranks.get(document_id),
                sparse_ranks.get(document_id),
            )
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "_hybrid_score": hybrid_score,
                    "_dense_score": dense_score,
                    "_sparse_score": (
                        float(sparse_item.score) if sparse_item is not None else None
                    ),
                    "_score_type": score_type,
                }
            )
            document.metadata = metadata
            results.append((document, float(result_score)))

        return _deduplicate_results(results, top_k)

    async def delete_collection(self, collection_name: str) -> bool:
        deleted = await self.vector_db.delete_collection(collection_name)
        try:
            await self.sparse_index.delete_collection(collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("清理知识库 BM25 索引失败: %s", exc, exc_info=True)
        return deleted


async def maybe_search(
    vector_db: VectorDBBase,
    search_service: HybridSearchService | None,
    collection_name: str,
    query: str,
    top_k: int,
) -> list[tuple[Document, float]]:
    if search_service is not None:
        return await search_service.search(collection_name, query, top_k)
    return await vector_db.search(collection_name, query, top_k)


def _positive_int(raw: Any, fallback: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def _rrf_score(dense_rank: int | None, sparse_rank: int | None) -> float:
    score = 0.0
    if dense_rank is not None:
        score += 1.0 / (60 + dense_rank)
    if sparse_rank is not None:
        score += 1.0 / (60 + sparse_rank)
    return score


def _deduplicate_results(
    results: Sequence[tuple[Document, float]],
    top_k: int,
) -> list[tuple[Document, float]]:
    seen: set[str] = set()
    deduplicated: list[tuple[Document, float]] = []
    for document, score in results:
        key = document.text_content.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduplicated.append((document, float(score)))
        if len(deduplicated) >= top_k:
            break
    return deduplicated
