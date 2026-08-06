import pytest
from astrbot_plugin_knowledge_base.utils.text_splitter import TextSplitterUtil


def test_markdown_splitter_preserves_section_path_and_code_fence():
    splitter = TextSplitterUtil(chunk_size=80, chunk_overlap=0)
    markdown = """# Overview

This is the overview.

## Configuration

Use the setting below.

```python
# Not a markdown heading
value = 1
```
"""

    chunks = splitter.split_for_ingestion(markdown, "guide.md")

    assert chunks[0].section_path == "Overview"
    assert all(chunk.section_path == "Overview / Configuration" for chunk in chunks[1:])
    configuration_text = "\n".join(chunk.text for chunk in chunks[1:])
    assert "章节: Overview / Configuration" in configuration_text
    assert "# Not a markdown heading" in configuration_text
    assert chunks[-1].metadata()["chunk_index"] == len(chunks) - 1
    assert len(chunks[-1].metadata()["content_hash"]) == 64


def test_split_text_honors_explicit_zero_overlap():
    splitter = TextSplitterUtil(chunk_size=5, chunk_overlap=2)

    chunks = splitter.split_text("abcdefghij", chunk_size=5, overlap=0)

    assert chunks == ["abcde", "fghij"]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (5, -1), (5, 5)],
)
def test_structured_splitter_rejects_invalid_parameters(chunk_size, overlap):
    splitter = TextSplitterUtil(chunk_size=10, chunk_overlap=2)

    with pytest.raises(ValueError):
        splitter.split_for_ingestion(
            "# Heading\n\ncontent",
            "guide.md",
            chunk_size=chunk_size,
            overlap=overlap,
        )
