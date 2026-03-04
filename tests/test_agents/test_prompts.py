from src.agents.prompts import get_template, PROMPT_TEMPLATES

def test_get_template():
    research = get_template("research")
    assert research is not None
    assert "Research Question" in research.user_prompt_template
    
    analysis = get_template("analysis")
    assert analysis is not None
    assert "Analysis Request" in analysis.user_prompt_template

def test_template_formatting_with_memory():
    template = get_template("research")
    
    # Test formatting with memory context
    formatted = template.format_user_prompt(
        query="What is AI?",
        memory_context="**Past Research**: AI is broad.",
        evidence_summary="Summary here.",
        sources_text="Source 1."
    )
    
    assert "What is AI?" in formatted
    assert "**Past Research**" in formatted
    assert "Summary here." in formatted

def test_template_formatting_without_memory():
    template = get_template("qa")
    
    # Test formatting with empty memory context
    formatted = template.format_user_prompt(
        query="Who are you?",
        memory_context="",
        evidence_summary="Information.",
        sources_text="Source A."
    )
    
    assert "Who are you?" in formatted
    assert "Information." in formatted
    # Should handle empty string without issues
