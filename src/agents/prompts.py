"""Configurable prompt templates for multi-agent RAG system.

Provides template-based prompt system that allows customization for different
use cases (research, fact-checking, analysis, etc.) without code changes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Generic prompt template for agent interactions.

    Attributes:
        system_prompt: System message that sets the AI's behavior
        user_prompt_template: Template for user queries with {placeholders}
        response_instructions: Instructions for how to format responses
    """

    system_prompt: str
    user_prompt_template: str
    response_instructions: str

    def format_user_prompt(self, **kwargs) -> str:
        """Format user prompt template with provided values.

        Args:
            **kwargs: Values to substitute into template

        Returns:
            Formatted prompt string
        """
        try:
            return self.user_prompt_template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing placeholder in template: {e}")
            return self.user_prompt_template

    def get_full_prompt(self, **kwargs) -> str:
        """Get complete prompt (system + user + instructions).

        Args:
            **kwargs: Values for user prompt template

        Returns:
            Complete prompt string
        """
        user_prompt = self.format_user_prompt(**kwargs)

        parts = [self.system_prompt, user_prompt]
        if self.response_instructions:
            parts.append(self.response_instructions)

        return "\n\n".join(parts)


# -------------------------------------------------------------------------
# Default Templates
# -------------------------------------------------------------------------


# Research template - for academic literature analysis and general research
RESEARCH_TEMPLATE = PromptTemplate(
    system_prompt="""You are an expert Research Synthesizer specializing in high-precision technical analysis.

Your goal is to answer the user's query based strictly on the provided documents. You must:
- Synthesize information from multiple sources without making assumptions.
- BE PRECISE: Do not attribute ownership or development of a technology (e.g. "Google's Willow chip") unless the source explicitly and directly states that ownership.
- USE EXACT TERMS: Do not use technical qualifiers like "fault-tolerant" or "stable" unless those exact words are used in the source to describe the finding.
- ACKNOWLEDGE GAPS: If a source has a date (e.g. a blog post from Dec 2024), do not state the finding itself happened on that date unless the text explicitly says so.
- Provide balanced perspectives on complex topics.""",
    user_prompt_template="""**Research Question**: {query}

{memory_context}

**Available Evidence**:
{evidence_summary}

**Sources**:
{sources_text}

Based on the evidence above, please provide a comprehensive response to the research question.""",
    response_instructions="""**Response Guidelines**:
1. Provide a direct, evidence-based response.
2. Cite sources using [Source X] notation for EVERY claim. Every sentence must have a citation.
3. IMPORTANT: Use ONLY the source numbers provided in the "Sources" section above.
4. ACCURACY: If the sources lack specific dates for an advancement, state "date not specified" rather than assuming the publication date is the advancement date.
5. Keep your response concise but informative (under 400 words).
6. Do NOT include metadata or introductory chat.

Begin your response directly with the answer.""",
)


# Analysis template - for analytical and evaluative tasks
ANALYSIS_TEMPLATE = PromptTemplate(
    system_prompt="""You are an analytical assistant specializing in detailed analysis and evaluation of complex topics.

Your role is to provide thorough, structured analysis that:
- Breaks down complex issues into components
- Identifies patterns and relationships
- Evaluates evidence and arguments
- Provides actionable insights""",
    user_prompt_template="""**Analysis Request**: {query}

{memory_context}

**Available Information**:
{evidence_summary}

**Sources**:
{sources_text}

Please provide a detailed analysis addressing this request.""",
    response_instructions="""**Response Guidelines**:
1. Structure your analysis with clear sections
2. Use evidence to support your analysis
3. Identify patterns, trends, and relationships
4. Provide data-driven insights where possible
5. Acknowledge limitations in the available information
6. Keep your response focused (under 500 words)
7. Do NOT include metadata in your response

Begin your analysis directly.""",
)


# Q&A template - for straightforward question answering
QA_TEMPLATE = PromptTemplate(
    system_prompt="""You are a helpful Q&A assistant. Provide clear, accurate answers based on the available information.""",
    user_prompt_template="""**Question**: {query}

{memory_context}

**Relevant Information**:
{evidence_summary}

**Sources**:
{sources_text}""",
    response_instructions="""Please answer the question based on the available information.
1. Provide a direct answer
2. Cite sources when appropriate
3. Keep your response concise (under 200 words)
4. Indicate if information is insufficient to answer completely

Begin your answer directly.""",
)


# -------------------------------------------------------------------------
# Template Registry
# -------------------------------------------------------------------------

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "research": RESEARCH_TEMPLATE,
    "analysis": ANALYSIS_TEMPLATE,
    "qa": QA_TEMPLATE,
    "default": RESEARCH_TEMPLATE,  # Default to research template
}


def get_template(template_name: str) -> PromptTemplate:
    """Get prompt template by name.

    Args:
        template_name: Name of the template (research, analysis, etc.)

    Returns:
        PromptTemplate instance

    Raises:
        ValueError: If template name is not found
    """
    if template_name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(f"Unknown template: {template_name}. " f"Available templates: {available}")

    return PROMPT_TEMPLATES[template_name]


def load_template_from_file(file_path: str) -> PromptTemplate:
    """Load custom prompt template from file.

    File format (JSON or simple text):
    For JSON:
    {
        "system_prompt": "...",
        "user_prompt_template": "...",
        "response_instructions": "..."
    }

    For text files, each line is a section:
    Line 1: system_prompt
    Line 2: user_prompt_template
    Line 3: response_instructions

    Args:
        file_path: Path to template file

    Returns:
        PromptTemplate instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {file_path}")

    if path.suffix == ".json":
        import json

        with open(path) as f:
            data = json.load(f)

        return PromptTemplate(
            system_prompt=data.get("system_prompt", ""),
            user_prompt_template=data.get("user_prompt_template", "{query}"),
            response_instructions=data.get("response_instructions", ""),
        )
    else:
        # Treat as text file with sections on separate lines
        with open(path) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 2:
            raise ValueError(
                f"Invalid template file: {file_path}. "
                "Must have at least system_prompt and user_prompt_template"
            )

        return PromptTemplate(
            system_prompt=lines[0],
            user_prompt_template=lines[1],
            response_instructions=lines[2] if len(lines) > 2 else "",
        )


def register_template(name: str, template: PromptTemplate) -> None:
    """Register a custom prompt template.

    Args:
        name: Name for the template
        template: PromptTemplate instance to register
    """
    PROMPT_TEMPLATES[name] = template
    logger.info(f"Registered custom prompt template: {name}")


def list_templates() -> dict[str, str]:
    """List all available prompt templates.

    Returns:
        Dictionary mapping template names to descriptions
    """
    return {
        "research": "Academic literature analysis and general research",
        "analysis": "Detailed analytical and evaluative tasks",
        "qa": "Straightforward question answering",
        "default": "Same as 'research' template",
    }
