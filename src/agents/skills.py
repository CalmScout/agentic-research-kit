"""
Skills loader for dynamic markdown-driven research protocols.
Adapts patterns from nanobot for dynamic prompt injection.
"""

from pathlib import Path

from src.utils.logger import logger


class SkillsLoader:
    """
    Loads domain-specific research protocols from SKILL.md files.
    These skills are injected into the agent's context to guide behavior.
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self._ensure_dir()

    def _ensure_dir(self):
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def list_available_skills(self) -> list[str]:
        """List all available skill files."""
        return [f.stem for f in self.skills_dir.glob("*.md")]

    def load_skill(self, skill_name: str) -> str | None:
        """Load the content of a specific SKILL.md file."""
        skill_path = self.skills_dir / f"{skill_name}.md"
        if not skill_path.exists():
            # Try lowercase matching
            skill_path = self.skills_dir / f"{skill_name.lower()}.md"
            if not skill_path.exists():
                logger.warning(f"Skill '{skill_name}' not found in {self.skills_dir}")
                return None

        try:
            with open(skill_path, encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Loaded skill: {skill_name}")
                return content
        except Exception as e:
            logger.error(f"Error loading skill {skill_name}: {e}")
            return None

    def get_skill_injection(self, skill_names: list[str]) -> str:
        """
        Loads multiple skills and returns a combined string for context injection.
        """
        injections = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                injections.append(f"### SKILL: {name}\n{content}")

        if not injections:
            return ""

        return (
            "\n\n"
            + "-" * 20
            + "\nDYNAMIC SKILLS INJECTED:\n"
            + "\n\n".join(injections)
            + "\n"
            + "-" * 20
            + "\n"
        )
