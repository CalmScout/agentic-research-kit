import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from src.agents.model_selector import Qwen2LangChainWrapper

@pytest.fixture
def mock_qwen_llm():
    mock = MagicMock()
    mock.max_new_tokens = 512
    # Mock tokenizer which is accessed as qwen2_llm.tokenizer in the wrapper
    mock.tokenizer = MagicMock()
    mock.tokenizer.apply_chat_template.return_value = "formatted prompt"
    return mock

def test_qwen_wrapper_init(mock_qwen_llm):
    wrapper = Qwen2LangChainWrapper(qwen2_llm=mock_qwen_llm)
    assert wrapper.qwen2_llm == mock_qwen_llm
    assert wrapper._llm_type == "qwen2_langchain_wrapper"

def test_qwen_wrapper_generate_basic(mock_qwen_llm):
    wrapper = Qwen2LangChainWrapper(qwen2_llm=mock_qwen_llm)
    mock_qwen_llm.generate.return_value = "Hello world"
    
    messages = [HumanMessage(content="Hi")]
    result = wrapper._generate(messages)
    
    assert isinstance(result.generations[0].message, AIMessage)
    assert result.generations[0].message.content == "Hello world"
    mock_qwen_llm.generate.assert_called_once()

def test_qwen_wrapper_bind_tools(mock_qwen_llm):
    wrapper = Qwen2LangChainWrapper(qwen2_llm=mock_qwen_llm)
    
    @tool
    def my_tool(query: str):
        """My tool description"""
        return "result"
        
    wrapper.bind_tools([my_tool])
    assert len(wrapper.tools_defs) == 1
    assert wrapper.tools_defs[0]["function"]["name"] == "my_tool"
    assert "parameters" in wrapper.tools_defs[0]["function"]

def test_qwen_wrapper_generate_with_tools(mock_qwen_llm):
    wrapper = Qwen2LangChainWrapper(qwen2_llm=mock_qwen_llm)
    tools = [{"type": "function", "function": {"name": "web_search", "description": "search web"}}]
    wrapper.bind_tools(tools)
    
    # Mock response with a tool call
    mock_qwen_llm.generate.return_value = '<|im_start|>call:web_search{"query": "superconductors"}<|im_end|>I will search now.'
    
    messages = [HumanMessage(content="Search for superconductors")]
    result = wrapper._generate(messages)
    
    ai_msg = result.generations[0].message
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["name"] == "web_search"
    assert ai_msg.content == "I will search now."
    
    # Verify apply_chat_template was called with tools
    mock_qwen_llm.tokenizer.apply_chat_template.assert_called_once()
    args, kwargs = mock_qwen_llm.tokenizer.apply_chat_template.call_args
    assert kwargs["tools"] == tools

@pytest.mark.asyncio
async def test_qwen_wrapper_agenerate(mock_qwen_llm):
    wrapper = Qwen2LangChainWrapper(qwen2_llm=mock_qwen_llm)
    mock_qwen_llm.generate.return_value = "Async response"
    
    messages = [HumanMessage(content="Hi")]
    result = await wrapper._agenerate(messages)
    
    assert result.generations[0].message.content == "Async response"
