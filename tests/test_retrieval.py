from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_knowledge_base.core.retrieval import (
    HybridSearchService,
    SparseIndex,
)
from astrbot_plugin_knowledge_base.vector_store.base import Document


@pytest.mark.asyncio
async def test_sparse_index_retrieves_chinese_keywords(tmp_path):
    index = SparseIndex(tmp_path / "sparse.sqlite3")
    await index.initialize()
    await index.index_documents(
        "general",
        [
            Document(text_content="AstrBot 的忙碌日程配置", metadata={"source": "a"}),
            Document(text_content="向量数据库的连接配置", metadata={"source": "b"}),
        ],
        ["doc-a", "doc-b"],
    )

    results = await index.search("general", "忙碌日程", top_k=2)

    assert [hit.document_id for hit in results] == ["doc-a"]
    assert results[0].metadata["source"] == "a"
    assert results[0].score > 0


@pytest.mark.asyncio
async def test_hybrid_fusion_uses_dense_score_for_legacy_threshold(tmp_path):
    vector_db = AsyncMock()
    dense_documents = [
        Document(id="doc-a", text_content="semantic alpha result"),
        Document(id="doc-b", text_content="exact beta keyword result"),
    ]
    vector_db.search.return_value = [
        (dense_documents[0], 0.91),
        (dense_documents[1], 0.82),
    ]
    service = HybridSearchService(
        vector_db,
        tmp_path / "sparse.sqlite3",
        {"dense_top_k": 2, "sparse_top_k": 2, "fusion_top_k": 2},
    )
    await service.initialize()
    await service.sparse_index.index_documents(
        "general",
        dense_documents,
        ["doc-a", "doc-b"],
    )

    results = await service.search("general", "beta keyword", top_k=2)

    assert [document.id for document, _ in results] == ["doc-b", "doc-a"]
    assert results[0][1] == pytest.approx(0.82)
    assert results[0][0].metadata["_score_type"] == "dense_compatibility"
    assert results[0][0].metadata["_dense_score"] == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_missing_sparse_index_falls_back_to_dense(tmp_path):
    vector_db = AsyncMock()
    dense_results = [
        (Document(id="doc-a", text_content="dense result"), 0.88),
        (Document(id="doc-b", text_content="another result"), 0.77),
    ]
    vector_db.search.return_value = dense_results
    service = HybridSearchService(vector_db, tmp_path / "sparse.sqlite3")
    await service.initialize()

    results = await service.search("legacy", "query", top_k=1)

    assert results == dense_results[:1]
    vector_db.search.assert_awaited_once_with("legacy", "query", top_k=20)


@pytest.mark.asyncio
async def test_sparse_query_failure_falls_back_to_dense(tmp_path):
    vector_db = AsyncMock()
    dense_results = [
        (Document(id="doc-a", text_content="dense result"), 0.88),
        (Document(id="doc-a-duplicate", text_content="dense result"), 0.80),
    ]
    vector_db.search.return_value = dense_results
    service = HybridSearchService(vector_db, tmp_path / "sparse.sqlite3")
    await service.initialize()
    service.sparse_index.search = AsyncMock(side_effect=RuntimeError("corrupt index"))

    results = await service.search("legacy", "query", top_k=2)

    assert results == dense_results[:1]


@pytest.mark.asyncio
async def test_sparse_only_result_survives_dense_similarity_threshold(tmp_path):
    vector_db = AsyncMock()
    vector_db.search.return_value = [
        (Document(id="dense", text_content="semantic result"), 0.91)
    ]
    service = HybridSearchService(
        vector_db,
        tmp_path / "sparse.sqlite3",
        {"dense_top_k": 1, "sparse_top_k": 2, "fusion_top_k": 2},
    )
    await service.initialize()
    await service.sparse_index.index_documents(
        "general",
        [
            Document(id="dense", text_content="semantic result"),
            Document(id="sparse", text_content="exact schedule keyword"),
        ],
        ["dense", "sparse"],
    )

    results = await service.search("general", "schedule keyword", top_k=2)

    sparse_result = next(
        (document, score) for document, score in results if document.id == "sparse"
    )
    assert sparse_result[1] > 0
    assert sparse_result[0].metadata["_score_type"] == "sparse_bm25"
    assert sparse_result[0].metadata["_dense_score"] is None


@pytest.mark.asyncio
async def test_partial_vector_write_does_not_build_misaligned_sparse_index(tmp_path):
    vector_db = AsyncMock()
    vector_db.add_documents.return_value = ["doc-a"]
    service = HybridSearchService(vector_db, tmp_path / "sparse.sqlite3")
    await service.initialize()
    service.sparse_index.index_documents = AsyncMock()
    documents = [
        Document(text_content="first"),
        Document(text_content="second"),
    ]

    result = await service.add_documents("general", documents)

    assert result == ["doc-a"]
    service.sparse_index.index_documents.assert_not_awaited()
