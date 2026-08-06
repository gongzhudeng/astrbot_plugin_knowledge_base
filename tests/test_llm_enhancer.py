from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot_plugin_knowledge_base.core.llm_enhancer import (
    _parse_min_similarity_score,
    enhance_request_with_kb,
)


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [
        (0.5, 0.5),
        ("0.75", 0.75),
        (-1, 0.0),
        (2, 1.0),
        ("invalid", 0.5),
        (float("nan"), 0.5),
    ],
)
def test_parse_min_similarity_score(raw_score, expected):
    assert _parse_min_similarity_score(raw_score) == expected


def _make_dependencies(
    scores,
    threshold=None,
    extras=None,
    retrieval_service=None,
    max_contexts=None,
    max_insert_length=None,
):
    event = Mock()
    event_extras = dict(extras or {})
    event.get_extra = Mock(
        side_effect=lambda key, default=None: event_extras.get(key, default)
    )
    event.set_extra = Mock(
        side_effect=lambda key, value: event_extras.__setitem__(key, value)
    )

    req = SimpleNamespace(
        prompt="current query",
        system_prompt="original system prompt",
        contexts=[{"role": "user", "content": "history"}],
        extra_user_content_parts=[],
    )

    vector_db = Mock()
    vector_db.collection_exists = AsyncMock(return_value=True)
    vector_db.search = AsyncMock(
        return_value=[
            (
                SimpleNamespace(
                    text_content=f"document-{index}",
                    metadata={"source": f"source-{index}"},
                ),
                score,
            )
            for index, score in enumerate(scores)
        ]
    )

    user_prefs = Mock()
    user_prefs.get_user_default_collection = Mock(return_value="general")

    config = {
        "search_top_k": 3,
        "kb_llm_insertion_method": "prepend_prompt",
    }
    if threshold is not None:
        config["kb_llm_min_similarity_score"] = threshold
    if max_contexts is not None:
        config["kb_llm_max_contexts"] = max_contexts
    if max_insert_length is not None:
        config["kb_llm_max_insert_length"] = max_insert_length

    return (
        event,
        req,
        vector_db,
        user_prefs,
        config,
        event_extras,
        retrieval_service,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_query"),
    [
        ("current query", "current query"),
        ("Spark retrieval query", "Spark retrieval query"),
    ],
)
async def test_request_query_uses_prompt_and_keeps_injection_temporary(
    prompt, expected_query
):
    retrieval_service = Mock()
    retrieval_service.search = AsyncMock(
        return_value=[
            (
                SimpleNamespace(
                    text_content="knowledge result",
                    metadata={"source": "test"},
                ),
                0.9,
            )
        ]
    )
    event, req, vector_db, user_prefs, config, _, _ = _make_dependencies(
        [], retrieval_service=retrieval_service
    )
    req.prompt = prompt
    original_contexts = list(req.contexts)
    original_system_prompt = req.system_prompt

    await enhance_request_with_kb(
        event,
        req,
        vector_db,
        user_prefs,
        config,
        retrieval_service,
    )

    retrieval_service.search.assert_awaited_once_with(
        "general", expected_query, top_k=3
    )
    assert vector_db.search.await_count == 0
    assert len(req.extra_user_content_parts) == 1
    assert req.extra_user_content_parts[0]._no_save is True
    assert req.prompt == prompt
    assert req.contexts == original_contexts
    assert req.system_prompt == original_system_prompt


@pytest.mark.asyncio
async def test_background_retrieval_uses_override_and_accumulates_results():
    retrieval_service = Mock()
    retrieval_service.search = AsyncMock(
        return_value=[
            (
                SimpleNamespace(
                    text_content="busy schedule knowledge",
                    metadata={"source": "schedule"},
                ),
                0.9,
            )
        ]
    )
    extras = {
        "background_retrieval": True,
        "retrieval_query": "busy schedule query",
        "background_retrieval_results": ["previous result"],
    }
    event, req, vector_db, user_prefs, config, event_extras, _ = _make_dependencies(
        [], extras=extras, retrieval_service=retrieval_service
    )

    await enhance_request_with_kb(
        event,
        req,
        vector_db,
        user_prefs,
        config,
        retrieval_service,
    )

    retrieval_service.search.assert_awaited_once_with(
        "general", "busy schedule query", top_k=3
    )
    assert event_extras["background_retrieval_results"] == [
        "previous result",
        req.extra_user_content_parts[0].text,
    ]
    assert req.extra_user_content_parts[0]._no_save is True
    assert req.system_prompt == "original system prompt"
    assert req.contexts == [{"role": "user", "content": "history"}]


@pytest.mark.asyncio
async def test_injection_deduplicates_and_respects_limits():
    event, req, vector_db, user_prefs, config, _, _ = _make_dependencies(
        [], max_contexts=1, max_insert_length=80
    )
    vector_db.search.return_value = [
        (
            SimpleNamespace(
                text_content="same knowledge " * 10,
                metadata={"source": "first"},
            ),
            0.9,
        ),
        (
            SimpleNamespace(
                text_content="same knowledge " * 10,
                metadata={"source": "duplicate"},
            ),
            0.8,
        ),
    ]

    await enhance_request_with_kb(event, req, vector_db, user_prefs, config)

    assert len(req.extra_user_content_parts) == 1
    assert len(req.extra_user_content_parts[0].text) == 80
    assert req.extra_user_content_parts[0]._no_save is True


@pytest.mark.asyncio
async def test_sparse_only_result_bypasses_dense_similarity_threshold():
    event, req, vector_db, user_prefs, config, _, _ = _make_dependencies(
        [], threshold=0.9
    )
    vector_db.search.return_value = [
        (
            SimpleNamespace(
                text_content="exact sparse knowledge",
                metadata={"source": "sparse", "_score_type": "sparse_bm25"},
            ),
            0.2,
        )
    ]

    await enhance_request_with_kb(event, req, vector_db, user_prefs, config)

    assert len(req.extra_user_content_parts) == 1
    assert "exact sparse knowledge" in req.extra_user_content_parts[0].text
    assert req.extra_user_content_parts[0]._no_save is True


@pytest.mark.asyncio
async def test_background_results_are_not_written_when_all_scores_fail_threshold():
    extras = {
        "background_retrieval": True,
        "retrieval_query": "busy query",
        "background_retrieval_results": [],
    }
    event, req, vector_db, user_prefs, config, event_extras, _ = _make_dependencies(
        [0.2], threshold=0.5, extras=extras
    )

    await enhance_request_with_kb(event, req, vector_db, user_prefs, config)

    assert event_extras["background_retrieval_results"] == []
    assert req.extra_user_content_parts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scores", "threshold", "included", "excluded"),
    [
        ([0.5, 0.49], None, ["document-0"], ["document-1"]),
        ([0.8, 0.79], 0.8, ["document-0"], ["document-1"]),
        ([-0.2, 0.1], 0, ["document-0", "document-1"], []),
        ([0.9, 0.1], "invalid", ["document-0"], ["document-1"]),
    ],
)
async def test_enhance_request_filters_by_configured_threshold(
    scores, threshold, included, excluded
):
    event, req, vector_db, user_prefs, config, _, _ = _make_dependencies(
        scores, threshold
    )

    await enhance_request_with_kb(event, req, vector_db, user_prefs, config)

    assert len(req.extra_user_content_parts) == 1
    inserted_text = req.extra_user_content_parts[0].text
    for content in included:
        assert content in inserted_text
    for content in excluded:
        assert content not in inserted_text


@pytest.mark.asyncio
async def test_enhance_request_skips_injection_when_all_scores_are_filtered():
    event, req, vector_db, user_prefs, config, _, _ = _make_dependencies(
        [0.2, 0.49], 0.5
    )

    await enhance_request_with_kb(event, req, vector_db, user_prefs, config)

    assert req.extra_user_content_parts == []
