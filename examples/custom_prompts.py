"""
Example: Custom Prompt Templates

This example demonstrates how to:
1. Use built-in prompt templates (research, analysis, qa, claim_verification)
2. Create custom prompt templates
3. Load templates from files
4. Register custom templates for reuse

Use Case: Tailoring the agent's behavior for specific research domains or tasks.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.prompts import (
    PromptTemplate,
    list_templates,
    load_template_from_file,
    register_template,
)
from agents.workflow import query_with_agents

# ============================================================================
# Example 1: Using Built-in Templates
# ============================================================================

async def example_builtin_templates():
    """Demonstrate usage of built-in prompt templates."""

    print("=" * 80)
    print("Example 1: Built-in Prompt Templates")
    print("=" * 80)

    # List all available templates
    templates = list_templates()

    print("\n📋 Available Templates:")
    for name, description in templates.items():
        print(f"   - {name}: {description}")

    # Example queries using different templates
    query = "What are the main ethical concerns in AI healthcare?"

    print(f"\n🔍 Query: {query}")

    # Research template
    print("\n1️⃣  Using 'research' template:")
    result = await query_with_agents(
        query=query,
        prompt_template="research"
    )
    print(f"   {result['response'][:300]}...")

    # Analysis template
    print("\n2️⃣  Using 'analysis' template:")
    result = await query_with_agents(
        query=query,
        prompt_template="analysis"
    )
    print(f"   {result['response'][:300]}...")

    # Q&A template
    print("\n3️⃣  Using 'qa' template:")
    result = await query_with_agents(
        query=query,
        prompt_template="qa"
    )
    print(f"   {result['response'][:300]}...")


# ============================================================================
# Example 2: Creating Custom Templates
# ============================================================================

async def example_custom_templates():
    """Demonstrate creation and usage of custom prompt templates."""

    print("\n" + "=" * 80)
    print("Example 2: Custom Prompt Templates")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Custom Template 1: Literature Review
    # -----------------------------------------------------------------------
    print("\n📝 Creating custom 'literature_review' template...")

    literature_review_template = PromptTemplate(
        system_prompt="""You are an academic research assistant specializing in literature reviews and bibliometric analysis.

Your expertise includes:
- Identifying key themes and trends across multiple papers
- Synthesizing findings from diverse sources
- Highlighting methodological approaches
- Noting gaps in the existing literature""",

        user_prompt_template="""**Research Topic**: {query}

**Literature Evidence**:
{evidence_summary}

**Source Papers**:
{sources_text}

Provide a comprehensive literature review addressing the research topic.""",

        response_instructions="""**Literature Review Structure**:
1. **Introduction**: Brief overview of the topic's importance
2. **Key Themes**: Main themes identified across the literature
3. **Methodological Approaches**: Common research methods used
4. **Findings Synthesis**: Consolidated findings from multiple sources
5. **Gaps and Future Directions**: Unexplored areas and suggested research

- Cite sources using [Source X] notation
- Maintain academic tone and objectivity
- Keep response under 500 words

Begin your literature review directly."""
    )

    # Register the custom template
    register_template("literature_review", literature_review_template)
    print("   ✓ Registered 'literature_review' template")

    # Use the custom template
    query = "What are the main challenges in deploying AI in clinical settings?"
    print(f"\n🔍 Query: {query}")
    print("   Using custom 'literature_review' template...")

    result = await query_with_agents(
        query=query,
        prompt_template="literature_review"
    )
    print("\n📊 Response:")
    print(f"   {result['response'][:400]}...")

    # -----------------------------------------------------------------------
    # Custom Template 2: Trend Analysis
    # -----------------------------------------------------------------------
    print("\n📝 Creating custom 'trend_analysis' template...")

    trend_analysis_template = PromptTemplate(
        system_prompt="""You are an expert in trend analysis and pattern recognition across temporal data.

Your role is to identify:
- Evolution of concepts over time
- Shifting research focuses
- Emerging technologies or methodologies
- Declining or obsolete approaches""",

        user_prompt_template="""**Analysis Request**: {query}

**Available Data**:
{evidence_summary}

**Sources**:
{sources_text}

Analyze trends and patterns in the available data.""",

        response_instructions="""**Trend Analysis Structure**:
1. **Timeline Overview**: Key developments by time period
2. **Emerging Trends**: Growing areas of interest
3. **Declining Trends**: Areas losing focus
4. **Inflection Points**: Major shifts in direction
5. **Future Predictions**: Likely future developments based on trends

- Use chronological organization where possible
- Highlight turning points and paradigm shifts
- Keep response under 400 words

Begin your trend analysis directly."""
    )

    register_template("trend_analysis", trend_analysis_template)
    print("   ✓ Registered 'trend_analysis' template")

    # Use the trend analysis template
    query = "How has deep learning been applied to medical image analysis over the past decade?"
    print(f"\n🔍 Query: {query}")
    print("   Using 'trend_analysis' template...")

    result = await query_with_agents(
        query=query,
        prompt_template="trend_analysis"
    )
    print("\n📊 Response:")
    print(f"   {result['response'][:400]}...")


# ============================================================================
# Example 3: Loading Templates from Files
# ============================================================================

async def example_file_templates():
    """Demonstrate loading templates from external files."""

    print("\n" + "=" * 80)
    print("Example 3: Loading Templates from Files")
    print("=" * 80)

    # Create example template files
    templates_dir = Path("./examples/templates")
    templates_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # JSON Template File
    # -----------------------------------------------------------------------
    json_template_path = templates_dir / "systematic_review.json"

    import json
    json_template_data = {
        "system_prompt": "You are a systematic review expert following PRISMA guidelines.",
        "user_prompt_template": "**Review Question**: {query}\n\n**Evidence**:\n{evidence_summary}",
        "response_instructions": "Provide a systematic review following PRISMA guidelines."
    }

    with open(json_template_path, 'w') as f:
        json.dump(json_template_data, f, indent=2)

    print(f"\n📝 Created JSON template: {json_template_path}")

    # Load the JSON template
    template = load_template_from_file(str(json_template_path))
    register_template("systematic_review", template)
    print("   ✓ Loaded and registered 'systematic_review' template from JSON")

    # -----------------------------------------------------------------------
    # Text Template File
    # -----------------------------------------------------------------------
    text_template_path = templates_dir / "meta_analysis.txt"

    text_template_content = """You are a meta-analysis specialist.
Analyze the provided evidence statistically and thematically.
**Query**: {query}
**Evidence**: {evidence_summary}
**Sources**: {sources_text}
Provide meta-analytic findings with effect sizes if available."""

    with open(text_template_path, 'w') as f:
        f.write(text_template_content)

    print(f"\n📝 Created text template: {text_template_path}")

    # Load the text template
    template = load_template_from_file(str(text_template_path))
    register_template("meta_analysis", template)
    print("   ✓ Loaded and registered 'meta_analysis' template from text")

    print("\n✅ Template files created in ./examples/templates/")
    print("   You can now edit these files to customize prompts without changing code.")


# ============================================================================
# Example 4: Domain-Specific Templates
# ============================================================================

async def example_domain_templates():
    """Demonstrate domain-specific prompt templates."""

    print("\n" + "=" * 80)
    print("Example 4: Domain-Specific Templates")
    print("=" * 80)

    # Biomedical Research Template
    biomedical_template = PromptTemplate(
        system_prompt="""You are a biomedical research advisor with expertise in:
- Clinical research methodology
- Regulatory considerations (FDA, EMA)
- Statistical analysis in healthcare
- Ethical considerations in medical research""",

        user_prompt_template="""**Biomedical Research Query**: {query}

**Available Evidence**:
{evidence_summary}

**Sources**:
{sources_text}

Provide an evidence-based biomedical research perspective.""",

        response_instructions="""**Response Structure**:
1. **Evidence Assessment**: Quality and strength of available evidence
2. **Clinical Relevance**: Practical implications for healthcare
3. **Methodological Considerations**: Study design and analysis notes
4. **Limitations**: Caveats and constraints
5. **Recommendations**: Evidence-based guidance

- Emphasize clinical significance over statistical significance
- Note regulatory considerations when relevant
- Acknowledge uncertainty transparently
- Keep response under 400 words

Begin your response directly."""
    )

    register_template("biomedical", biomedical_template)
    print("✓ Registered 'biomedical' domain-specific template")

    # Use it
    query = "What is the evidence for AI-assisted radiology in detecting lung cancer?"
    print(f"\n🔍 Query: {query}")

    result = await query_with_agents(
        query=query,
        prompt_template="biomedical"
    )
    print("\n📊 Biomedical Research Response:")
    print(f"   {result['response'][:400]}...")


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run all examples."""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   Agentic Research Kit: Custom Prompt Templates                              ║
║                                                                              ║
║   This example demonstrates how to create, customize, and use                ║
║   prompt templates for different research domains and tasks.                ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run all examples
    await example_builtin_templates()
    await example_custom_templates()
    await example_file_templates()
    await example_domain_templates()

    print("\n" + "=" * 80)
    print("All Examples Complete!")
    print("=" * 80)
    print("\n💡 Tips:")
    print("   1. Start with built-in templates (research, analysis, qa)")
    print("   2. Create custom templates for domain-specific needs")
    print("   3. Store frequently-used templates in files for reuse")
    print("   4. Register templates at application startup for availability")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
