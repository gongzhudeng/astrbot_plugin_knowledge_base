# astrbot_plugin_knowledge_base/llm_enhancer.py
import math
from typing import TYPE_CHECKING

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

if TYPE_CHECKING:
    from ..vector_store.base import VectorDBBase
    from .retrieval import HybridSearchService
    from .user_prefs_handler import UserPrefsHandler


_DEFAULT_MIN_SIMILARITY_SCORE = 0.5


def _parse_min_similarity_score(raw_score: object) -> float:
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        logger.warning(
            "知识库最低相关度配置无效，使用默认值 "
            f"{_DEFAULT_MIN_SIMILARITY_SCORE}: {raw_score!r}"
        )
        return _DEFAULT_MIN_SIMILARITY_SCORE

    if not math.isfinite(score):
        logger.warning(
            "知识库最低相关度配置不是有限数值，使用默认值 "
            f"{_DEFAULT_MIN_SIMILARITY_SCORE}: {raw_score!r}"
        )
        return _DEFAULT_MIN_SIMILARITY_SCORE

    clamped_score = max(0.0, min(1.0, score))
    if clamped_score != score:
        logger.warning(
            f"知识库最低相关度配置 {score} 超出 0-1，已调整为 {clamped_score}。"
        )
    return clamped_score


def _parse_positive_int(raw_value: object, fallback: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def _score_passes_threshold(
    doc: object,
    score: float,
    min_similarity_score: float,
) -> bool:
    metadata = getattr(doc, "metadata", {}) or {}
    if metadata.get("_score_type") == "sparse_bm25":
        return score > 0
    return min_similarity_score == 0 or score >= min_similarity_score


async def enhance_request_with_kb(
    event: AstrMessageEvent,
    req: ProviderRequest,
    vector_db: "VectorDBBase",
    user_prefs_handler: "UserPrefsHandler",
    plugin_config: AstrBotConfig,
    retrieval_service: "HybridSearchService | None" = None,
):
    default_collection_name = user_prefs_handler.get_user_default_collection(event)

    if not default_collection_name:
        logger.debug("未找到当前会话的默认知识库，跳过知识库查询。")
        return

    if not await vector_db.collection_exists(default_collection_name):
        logger.warning(f"知识库 '{default_collection_name}' 不存在，跳过知识库查询。")
        return

    kb_search_top_k = plugin_config.get("search_top_k", 3)
    kb_insertion_method = plugin_config.get("kb_llm_insertion_method", "system_prompt")
    kb_context_template = plugin_config.get(
        "kb_llm_context_template",
        "这是相关的知识库信息，请参考这些信息来回答用户的问题：\n{retrieved_contexts}",
    )
    min_similarity_score = _parse_min_similarity_score(
        plugin_config.get("kb_llm_min_similarity_score", _DEFAULT_MIN_SIMILARITY_SCORE)
    )

    background_query = event.get_extra("retrieval_query")
    is_background_retrieval = bool(event.get_extra("background_retrieval"))
    user_query = (
        background_query.strip()
        if is_background_retrieval and isinstance(background_query, str)
        else req.prompt
    )
    if not user_query or not user_query.strip():
        logger.debug("用户查询为空，跳过知识库搜索。")
        return

    try:
        logger.info(
            f"为LLM请求在知识库 '{default_collection_name}' 中搜索: '{user_query[:50]}...' (top_k={kb_search_top_k})"
        )
        search_results = (
            await retrieval_service.search(
                default_collection_name,
                user_query,
                top_k=kb_search_top_k,
            )
            if retrieval_service is not None
            else await vector_db.search(
                default_collection_name, user_query, top_k=kb_search_top_k
            )
        )
    except Exception as e:
        logger.error(
            f"LLM 请求时从知识库 '{default_collection_name}' 搜索失败: {e}",
            exc_info=True,
        )
        return

    if not search_results:
        logger.info(
            f"在知识库 '{default_collection_name}' 中未找到与查询 '{user_query[:50]}...' 相关的内容。"
        )
        return

    retrieved_contexts_list = []
    candidate_scores = []
    seen_contents = set()
    max_contexts = _parse_positive_int(
        plugin_config.get("kb_llm_max_contexts", 10),
        10,
    )
    for doc, raw_score in search_results:
        normalized_content = doc.text_content.strip()
        if not normalized_content or normalized_content in seen_contents:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            logger.warning(
                f"文档 '{doc.text_content[:30]}...' 返回了无效相关度 "
                f"{raw_score!r}，已忽略。"
            )
            continue
        if not math.isfinite(score):
            logger.warning(
                f"文档 '{doc.text_content[:30]}...' 返回了非有限相关度 "
                f"{raw_score!r}，已忽略。"
            )
            continue

        candidate_scores.append(score)
        if _score_passes_threshold(doc, score, min_similarity_score):
            source_info = doc.metadata.get("source", "未知来源")
            context_item = (
                f"- 内容: {doc.text_content} (来源: {source_info}, 相关度: {score:.2f})"
            )
            retrieved_contexts_list.append(context_item)
            seen_contents.add(normalized_content)
            if len(retrieved_contexts_list) >= max_contexts:
                break
        else:
            logger.debug(
                f"文档 '{doc.text_content[:30]}...' 相关度 {score:.2f} "
                f"低于阈值 {min_similarity_score:.2f}，已忽略。"
            )

    logger.info(
        f"知识库相关度筛选: 后端={type(vector_db).__name__}, "
        f"候选={len(search_results)}, 通过={len(retrieved_contexts_list)}, "
        f"阈值={min_similarity_score:.2f}, "
        f"分数={[round(score, 4) for score in candidate_scores]}"
    )

    if not retrieved_contexts_list:
        logger.info(
            f"所有检索到的知识库内容相关度均低于阈值 {min_similarity_score}，不进行增强。"
        )
        return

    formatted_contexts = "\n".join(retrieved_contexts_list)
    knowledge_to_insert = kb_context_template.format(
        retrieved_contexts=formatted_contexts
    )

    max_kb_insert_length = _parse_positive_int(
        plugin_config.get("kb_llm_max_insert_length", 200000),
        200000,
    )
    if len(knowledge_to_insert) > max_kb_insert_length:
        logger.warning(
            f"知识库插入内容过长 ({len(knowledge_to_insert)} chars)，将被截断至 {max_kb_insert_length} chars。"
        )
        truncation_marker = "\n... [内容已截断]"
        if max_kb_insert_length <= len(truncation_marker):
            knowledge_to_insert = knowledge_to_insert[:max_kb_insert_length]
        else:
            knowledge_to_insert = (
                knowledge_to_insert[: max_kb_insert_length - len(truncation_marker)]
                + truncation_marker
            )

    if is_background_retrieval:
        results = event.get_extra("background_retrieval_results")
        if not isinstance(results, list):
            results = []
        results.append(knowledge_to_insert)
        event.set_extra("background_retrieval_results", results)

    if kb_insertion_method == "system_prompt":
        # Insert after busy_schedule cache block if it exists, otherwise append
        busy_schedule_end = "<!-- /BUSY_SCHEDULE_CACHE -->"
        if busy_schedule_end in req.system_prompt:
            idx = req.system_prompt.index(busy_schedule_end) + len(busy_schedule_end)
            req.system_prompt = (
                req.system_prompt[:idx]
                + f"\n\n{knowledge_to_insert}"
                + req.system_prompt[idx:]
            )
            logger.info(
                f"知识库内容已插入到 system_prompt 的忙碌日程块之后。长度: {len(knowledge_to_insert)}"
            )
        elif req.system_prompt:
            req.system_prompt = f"{req.system_prompt}\n\n{knowledge_to_insert}"
            logger.info(
                f"知识库内容已添加到 system_prompt（无忙碌日程块，追加到末尾）。长度: {len(knowledge_to_insert)}"
            )
        else:
            req.system_prompt = knowledge_to_insert
    elif kb_insertion_method == "prepend_prompt":
        # Inject as a temporary content part: visible to LLM this turn,
        # but never written into req.contexts or persistent history.
        req.extra_user_content_parts.append(
            TextPart(text=knowledge_to_insert).mark_as_temp()
        )
        logger.info(
            f"知识库内容已注入为临时用户内容块（不写入历史）。长度: {len(knowledge_to_insert)}"
        )
    else:
        logger.warning(
            f"未知的知识库内容插入方式: {kb_insertion_method}，将默认注入为临时用户内容块。"
        )
        req.extra_user_content_parts.append(
            TextPart(text=knowledge_to_insert).mark_as_temp()
        )

    if req.system_prompt:
        logger.debug(
            f"修改后的 ProviderRequest.system_prompt: {req.system_prompt[:200]}..."
        )
