# astrbot_plugin_knowledge_base/llm_enhancer.py
from typing import TYPE_CHECKING

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

if TYPE_CHECKING:
    from ..vector_store.base import VectorDBBase
    from .user_prefs_handler import UserPrefsHandler


async def enhance_request_with_kb(
    event: AstrMessageEvent,
    req: ProviderRequest,
    vector_db: "VectorDBBase",
    user_prefs_handler: "UserPrefsHandler",
    plugin_config: AstrBotConfig,
):
    default_collection_name = user_prefs_handler.get_user_default_collection(event)

    if not default_collection_name:
        logger.debug("未找到当前会话的默认知识库，跳过知识库查询。")
        return

    if not await vector_db.collection_exists(default_collection_name):
        logger.warning(
            f"知识库 '{default_collection_name}' 不存在，跳过知识库查询。"
        )
        return

    kb_search_top_k = plugin_config.get("search_top_k", 3)
    kb_insertion_method = plugin_config.get("kb_llm_insertion_method", "system_prompt")
    kb_context_template = plugin_config.get(
        "kb_llm_context_template",
        "这是相关的知识库信息，请参考这些信息来回答用户的问题：\n{retrieved_contexts}",
    )
    min_similarity_score = plugin_config.get("kb_llm_min_similarity_score", 0.5)

    user_query = req.prompt
    if not user_query or not user_query.strip():
        logger.debug("用户查询为空，跳过知识库搜索。")
        return

    try:
        logger.info(
            f"为LLM请求在知识库 '{default_collection_name}' 中搜索: '{user_query[:50]}...' (top_k={kb_search_top_k})"
        )
        search_results = await vector_db.search(
            default_collection_name, user_query, top_k=kb_search_top_k
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
    for doc, score in search_results:
        if score >= min_similarity_score:
            source_info = doc.metadata.get("source", "未知来源")
            context_item = (
                f"- 内容: {doc.text_content} (来源: {source_info}, 相关度: {score:.2f})"
            )
            retrieved_contexts_list.append(context_item)
        else:
            logger.debug(
                f"文档 '{doc.text_content[:30]}...' 相关度 {score:.2f} 低于阈值 {min_similarity_score}，已忽略。"
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

    max_kb_insert_length = plugin_config.get("kb_llm_max_insert_length", 200000)
    if len(knowledge_to_insert) > max_kb_insert_length:
        logger.warning(
            f"知识库插入内容过长 ({len(knowledge_to_insert)} chars)，将被截断至 {max_kb_insert_length} chars。"
        )
        knowledge_to_insert = (
            knowledge_to_insert[:max_kb_insert_length] + "\n... [内容已截断]"
        )

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