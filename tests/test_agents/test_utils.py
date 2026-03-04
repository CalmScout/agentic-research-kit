from src.agents.utils import (
    format_response_for_display,
    group_docs_by_source,
    parse_title_from_content,
)


def test_parse_title_from_content():
    content = "Title: My Document\nContent: This is the body."
    assert parse_title_from_content(content) == "My Document"

    assert parse_title_from_content("Just some text") is None

def test_group_docs_by_source():
    docs = [
        {"text": "chunk 1", "metadata": {"file_path": "file1.txt", "title": "Doc 1"}, "score": 0.9},
        {"text": "chunk 2", "metadata": {"file_path": "file1.txt", "title": "Doc 1"}, "score": 0.8},
        {"text": "chunk 3", "metadata": {"file_path": "file2.txt"}, "score": 0.7}
    ]

    grouped = group_docs_by_source(docs)
    assert len(grouped) == 2
    assert grouped[0]["source"] == "file1.txt"
    assert len(grouped[0]["chunks"]) == 2
    assert grouped[1]["source"] == "file2.txt"

def test_format_response_for_display():
    response = "The answer is 42."
    sources = [
        {"source": "src1", "title": "Title 1", "chunks": [{"content": "content 1"}]}
    ]

    formatted = format_response_for_display(response, sources)
    assert "The answer is 42." in formatted
    assert "**Sources**:" in formatted
    assert "Title 1" in formatted
    assert "src1" in formatted
