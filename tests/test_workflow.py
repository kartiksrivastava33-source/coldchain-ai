from src.orchestrator import content_to_text, count_tool_results
from langchain_core.messages import AIMessage, ToolMessage


def test_content_to_text_handles_string():
    assert content_to_text("hello") == "hello"


def test_content_to_text_handles_gemini_blocks():
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert content_to_text(content) == "first\nsecond"


def test_count_tool_results():
    messages = [
        AIMessage(content="thinking"),
        ToolMessage(content="result", tool_call_id="call_1"),
        ToolMessage(content="result2", tool_call_id="call_2"),
    ]
    assert count_tool_results(messages) == 2
