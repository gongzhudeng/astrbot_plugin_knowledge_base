import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructuredChunk:
    text: str
    section_path: str = ""
    chunk_index: int = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "section_path": self.section_path,
            "chunk_index": self.chunk_index,
            "content_hash": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
        }


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        length_function: Callable[[str], int],
        is_separator_regex: bool = False,
        separators: list[str] = None,
    ):
        """
        初始化递归字符文本分割器

        Args:
            chunk_size: 每个文本块的最大大小
            chunk_overlap: 每个文本块之间的重叠部分大小
            length_function: 计算文本长度的函数
            is_separator_regex: 分隔符是否为正则表达式
            separators: 用于分割文本的分隔符列表，按优先级排序
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.is_separator_regex = is_separator_regex

        # 默认分隔符列表，按优先级从高到低
        self.separators = separators or [
            "\n\n",  # 段落
            "\n",  # 换行
            "。",  # 中文句子
            "，",  # 中文逗号
            ". ",  # 句子
            ", ",  # 逗号分隔
            " ",  # 单词
            "",  # 字符
        ]

    def split_text(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> list[str]:
        """
        递归地将文本分割成块

        Args:
            text: 要分割的文本

        Returns:
            分割后的文本块列表
        """
        if not text:
            return []

        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        overlap = self.chunk_overlap if overlap is None else overlap
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

        text_length = self.length_function(text)
        if text_length <= chunk_size:
            return [text]

        for separator in self.separators:
            if separator == "":
                return self._split_by_character(text, chunk_size, overlap)

            if separator in text:
                splits = text.split(separator)
                # 重新添加分隔符（除了最后一个片段）
                splits = [s + separator for s in splits[:-1]] + [splits[-1]]
                splits = [s for s in splits if s]
                if len(splits) == 1:
                    continue

                # 递归合并分割后的文本块
                final_chunks = []
                current_chunk = []
                current_chunk_length = 0

                for split in splits:
                    split_length = self.length_function(split)

                    # 如果单个分割部分已经超过了chunk_size，需要递归分割
                    if split_length > chunk_size:
                        # 先处理当前积累的块
                        if current_chunk:
                            combined_text = "".join(current_chunk)
                            final_chunks.extend(
                                self.split_text(combined_text, chunk_size, overlap)
                            )
                            current_chunk = []
                            current_chunk_length = 0

                        # 递归分割过大的部分
                        final_chunks.extend(self.split_text(split, chunk_size, overlap))
                    # 如果添加这部分会使当前块超过chunk_size
                    elif current_chunk_length + split_length > chunk_size:
                        # 合并当前块并添加到结果中
                        combined_text = "".join(current_chunk)
                        final_chunks.append(combined_text)

                        # 处理重叠部分
                        overlap_start = max(0, len(combined_text) - overlap)
                        if overlap_start > 0:
                            overlap_text = combined_text[overlap_start:]
                            current_chunk = [overlap_text, split]
                            current_chunk_length = (
                                self.length_function(overlap_text) + split_length
                            )
                        else:
                            current_chunk = [split]
                            current_chunk_length = split_length
                    else:
                        # 添加到当前块
                        current_chunk.append(split)
                        current_chunk_length += split_length

                # 处理剩余的块
                if current_chunk:
                    final_chunks.append("".join(current_chunk))

                return final_chunks

        return [text]

    def _split_by_character(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> list[str]:
        """
        按字符级别分割文本

        Args:
            text: 要分割的文本

        Returns:
            分割后的文本块列表
        """
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        overlap = self.chunk_overlap if overlap is None else overlap
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")
        result = []
        for i in range(0, len(text), chunk_size - overlap):
            end = min(i + chunk_size, len(text))
            result.append(text[i:end])
            if end == len(text):
                break

        return result


class TextSplitterUtil:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        初始化文本分割器。
        Args:
            chunk_size: 每个块的目标大小 (字符数或 token 数，取决于分割器实现)
            chunk_overlap: 块之间的重叠大小
        """
        # 使用 Langchain 的 RecursiveCharacterTextSplitter，它按字符分割并尝试保持段落完整性
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # 按字符数计算长度
            is_separator_regex=False,
        )
        # logger.info(f"文本分割器初始化：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def split_text(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> list[str]:
        if not text or not text.strip():
            return []
        return self.splitter.split_text(text, chunk_size, overlap)

    def split_for_ingestion(
        self,
        text: str,
        source_name: str = "",
        chunk_size: int = None,
        overlap: int = None,
    ) -> list[StructuredChunk]:
        if not text or not text.strip():
            return []

        is_markdown = source_name.lower().endswith((".md", ".markdown"))
        if not is_markdown and not re.search(r"(?m)^#{1,6}\s+\S", text):
            chunks = self.split_text(text, chunk_size, overlap)
            return [
                StructuredChunk(text=chunk, chunk_index=index)
                for index, chunk in enumerate(chunks)
            ]

        structured_chunks: list[StructuredChunk] = []
        configured_size = self.splitter.chunk_size if chunk_size is None else chunk_size
        configured_overlap = self.splitter.chunk_overlap if overlap is None else overlap
        if configured_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if configured_overlap < 0 or configured_overlap >= configured_size:
            raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

        for section_path, section_text in _split_markdown_sections(text):
            full_prefix = f"章节: {section_path}\n\n" if section_path else ""
            prefix = full_prefix[: configured_size // 2]
            available_size = max(1, configured_size - len(prefix))
            section_overlap = min(configured_overlap, max(0, available_size - 1))
            chunks = self.split_text(section_text, available_size, section_overlap)
            for chunk in chunks:
                structured_chunks.append(
                    StructuredChunk(
                        text=f"{prefix}{chunk}".strip(),
                        section_path=section_path,
                        chunk_index=len(structured_chunks),
                    )
                )
        return structured_chunks


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    headings: list[str] = []
    sections: list[tuple[str, str]] = []
    current_lines: list[str] = []
    current_path = ""
    in_fence = False

    def flush() -> None:
        nonlocal current_lines
        section_text = "".join(current_lines).strip()
        if section_text:
            sections.append((current_path, section_text))
        current_lines = []

    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        heading = None if in_fence else heading_pattern.match(line.rstrip("\r\n"))
        if heading is None:
            current_lines.append(line)
            continue

        flush()
        level = len(heading.group(1))
        title = heading.group(2).strip()
        headings[level - 1 :] = [title]
        current_path = " / ".join(headings)

    flush()
    return sections or [("", text.strip())]


# 基于 tiktoken 的分割器
# class TokenTextSplitter:
#     def __init__(self, chunk_size: int, chunk_overlap: int, model_name: str = "text-embedding-ada-002"):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         try:
#             self.encoding = get_encoding(model_name)
#         except: # Fallback for models not directly supported by tiktoken's default list
#             self.encoding = get_encoding("cl100k_base")


#     def split_text(self, text: str) -> list[str]:
#         if not text or not text.strip():
#             return []
#         tokens = self.encoding.encode(text)
#         chunks = []
#         for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
#             chunk_tokens = tokens[i:i + self.chunk_size]
#             chunks.append(self.encoding.decode(chunk_tokens))
#         return chunks
