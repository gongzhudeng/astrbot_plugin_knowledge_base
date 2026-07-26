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


def _make_dependencies(scores, threshold=None):
    event = Mock()
    event.get_extra = Mock(return_value=None)
    event.set_extra = Mock()

    req = SimpleNamespace(
        prompt="current query",
        system_prompt="",
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

    return event, req, vector_db, user_prefs, config


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
    event, req, vector_db, user_prefs, config = _make_dependencies(scores, threshold)

    await enhance_request_with_kb(event, req, vector_db, user_prefs, config)

    assert len(req.extra_user_content_parts) == 1
    inserted_text = req.extra_user_content_parts[0].text
    for content in included:
        assert content in inserted_text
    for content in excluded:
        assert content not in inserted_text


@pytest.mark.asyncio
async def test_enhance_request_skips_injection_when_all_scores_are_filtered():
    event, req, vector_db, user_prefs, config = _make_dependencies([0.2, 0.49], 0.5)

    await enhance_request_with_kb(event, req, vector_db, user_prefs, config)

    assert req.extra_user_content_parts == []
