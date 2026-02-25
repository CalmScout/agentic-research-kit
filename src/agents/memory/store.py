"""Memory system for persistent research context across RAG sessions.

Adapts nanobot's two-layer memory pattern for RAG research use cases:
- RESEARCH_MEMORY.md - Long-term research findings and patterns
- QUERY_HISTORY.md - Grep-searchable log of all queries with metrics
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


class MemoryStore:
    """Two-layer memory system for research context persistence.

    Provides:
    1. Long-term memory for research patterns and findings
    2. Query history for grep-searchable audit log
    3. Session consolidation for summarizing research patterns

    Example:
        >>> memory = MemoryStore(Path("./workspace"))
        >>> memory.append_query_history("What causes climate change?", result)
        >>> context = memory.get_research_context()
    """

    def __init__(self, workspace: Path):
        """Initialize memory store with workspace directory.

        Args:
            workspace: Root directory for memory storage
        """
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "RESEARCH_MEMORY.md"
        self.history_file = self.memory_dir / "QUERY_HISTORY.md"

        import lancedb
        self.db = lancedb.connect(self.memory_dir / "lancedb")
        self.table_name = "research_memory"

        # Auto-reindex if table is missing but data exists
        if self.table_name not in self.db.list_tables() and self.memory_file.exists():
            logger.info("Initializing semantic memory table from existing markdown...")
            self._reindex_all_findings(self.read_long_term())

        logger.debug(f"Memory store initialized at: {self.memory_dir}")

    # -------------------------------------------------------------------------
    # Long-term Research Memory
    # -------------------------------------------------------------------------

    def read_long_term(self) -> str:
        """Read long-term research memory.

        Returns:
            str: Contents of RESEARCH_MEMORY.md, or empty string if not exists
        """
        if self.memory_file.exists():
            content = self.memory_file.read_text(encoding="utf-8")
            logger.debug(f"Read {len(content)} chars from research memory")
            return content
        return ""

    def write_long_term(self, content: str) -> None:
        """Write long-term research memory.

        Args:
            content: Markdown content to write to RESEARCH_MEMORY.md
        """
        self.memory_file.write_text(content, encoding="utf-8")
        logger.info(f"Wrote {len(content)} chars to research memory")
        self._reindex_all_findings(content)

    def _reindex_all_findings(self, content: str) -> None:
        """Parse RESEARCH_MEMORY.md and re-index into LanceDB."""
        try:
            if self.table_name in self.db.list_tables():
                self.db.drop_table(self.table_name)
                
            sections = content.split("## Finding")
            data = []
            
            # Lazy load embedder
            try:
                from src.agents.embeddings import embedder
            except ImportError:
                logger.warning("Embedder not available. Skipping LanceDB indexing.")
                return

            for section in sections:
                if not section.strip():
                    continue
                lines = section.strip().split("\n", 1)
                text = section.strip()
                timestamp = "Unknown"
                if lines[0].startswith("[") and "]" in lines[0]:
                    timestamp = lines[0].split("]")[0].strip("[")
                    if len(lines) > 1:
                        text = lines[1].strip()
                
                vector = embedder.embed_text(text)
                data.append({"vector": vector, "text": text, "timestamp": timestamp})
                
            if data:
                self.db.create_table(self.table_name, data)
                logger.info(f"Re-indexed {len(data)} findings in LanceDB")
        except Exception as e:
            logger.error(f"Failed to re-index LanceDB: {e}")

    def append_research_finding(self, finding: str) -> None:
        """Append a research finding to long-term memory.

        Args:
            finding: Research finding in markdown format
        """
        timestamp = datetime.now().isoformat()
        entry = f"\n## Finding [{timestamp}]\n\n{finding}\n"

        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(entry)

        logger.debug(f"Appended research finding to memory")
        
        # Index in LanceDB
        try:
            from src.agents.embeddings import embedder
            vector = embedder.embed_text(finding)
            data = [{"vector": vector, "text": finding, "timestamp": timestamp}]
            if self.table_name in self.db.list_tables():
                tbl = self.db.open_table(self.table_name)
                tbl.add(data)
            else:
                self.db.create_table(self.table_name, data)
            logger.debug("Indexed new finding in LanceDB")
        except Exception as e:
            logger.error(f"Failed to index finding in LanceDB: {e}")

    # -------------------------------------------------------------------------
    # Query History
    # -------------------------------------------------------------------------

    def append_query_history(
        self,
        query: str,
        result: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> None:
        """Log query with metrics to history file.

        Args:
            query: User's query
            result: Query result dictionary containing response, sources, etc.
            session_id: Optional session identifier
        """
        timestamp = datetime.now().isoformat()

        # Extract metrics
        sources_count = len(result.get("sources", []))
        retrieved_count = result.get("retrieved_count", 0)

        # Build entry
        entry = f"""## Query: {query}

- **Timestamp**: {timestamp}
- **Session ID**: {session_id or 'N/A'}
- **Sources**: {sources_count}
- **Retrieved**: {retrieved_count}
- **Query Type**: {result.get('query_type', 'unknown')}

### Response

{result.get('response', 'No response')[:500]}...

### Sources

"""
        # Add source summaries
        for i, source in enumerate(result.get("sources", [])[:3], 1):
            url = source.get("url", source.get("source", "Unknown"))
            text = source.get("text", source.get("content", ""))[:100]
            entry += f"{i}. {text}... [{url}]\n"

        entry += "\n" + "-" * 80 + "\n\n"

        # Append to history
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry)

        logger.debug(f"Logged query to history: {query[:50]}...")

    # -------------------------------------------------------------------------
    # Context Retrieval
    # -------------------------------------------------------------------------

    def get_research_context(self, query: str = None, max_chars: int = 2000, top_k: int = 5) -> str:
        """Get research context for query enhancement.

        Args:
            query: User's query for semantic retrieval (if None, falls back to full text truncation)
            max_chars: Maximum characters to return from long-term memory
            top_k: Number of most relevant findings to return (semantic mode)

        Returns:
            str: Research context in markdown format
        """
        if not query or self.table_name not in self.db.list_tables():
            # Fallback to simple truncation
            long_term = self.read_long_term()

            if not long_term:
                return ""

            if len(long_term) > max_chars:
                long_term = long_term[:max_chars] + "\n\n...(truncated)"

            return f"## Research Context\n\n{long_term}"

        # Semantic Retrieval
        try:
            from src.agents.embeddings import embedder
            vector = embedder.embed_text(query)
            tbl = self.db.open_table(self.table_name)
            results = tbl.search(vector).limit(top_k).to_list()
            
            if not results:
                return ""
                
            context = "## Research Context (Relevant Past Findings)\n\n"
            for r in results:
                context += f"### Finding [{r.get('timestamp', 'Unknown')}]\n{r.get('text', '')}\n\n"
            return context
        except Exception as e:
            logger.error(f"Failed to retrieve context from LanceDB: {e}")
            # Fallback
            long_term = self.read_long_term()
            if not long_term: return ""
            if len(long_term) > max_chars:
                long_term = long_term[:max_chars] + "\n\n...(truncated)"
            return f"## Research Context\n\n{long_term}"

    def get_recent_queries(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent queries from history file.

        Args:
            count: Number of recent queries to retrieve

        Returns:
            List of dicts with query, timestamp, etc.
        """
        if not self.history_file.exists():
            return []

        # Simple parsing (for production, use proper markdown parser)
        content = self.history_file.read_text(encoding="utf-8")
        sections = content.split("## Query:")

        recent_queries = []
        for section in sections[-count:]:
            if not section.strip():
                continue

            lines = section.strip().split("\n")
            query = lines[0] if lines else ""
            timestamp = "Unknown"

            # Extract metadata
            for line in lines[1:5]:
                if line.startswith("- **Timestamp**"):
                    timestamp = line.split(":")[1].strip()

            recent_queries.append({
                "query": query,
                "timestamp": timestamp
            })

        return recent_queries

    # -------------------------------------------------------------------------
    # Session Consolidation
    # -------------------------------------------------------------------------

    async def consolidate_session(
        self,
        session_queries: List[Dict[str, Any]],
        llm=None
    ) -> str:
        """Consolidate session queries into research findings.

        Uses LLM to identify research patterns and extract key findings.

        Args:
            session_queries: List of query results from a session
            llm: Optional LLM for intelligent consolidation (LangChain-compatible)

        Returns:
            str: Consolidated research findings
        """
        if not session_queries:
            return ""

        if llm is None:
            # Fallback to basic summary if no LLM provided
            logger.warning("No LLM provided for consolidation, using basic summary")
            findings = f"# Session Summary - {datetime.now().isoformat()}\n\n"
            findings += f"## Queries Analyzed: {len(session_queries)}\n\n"
            for q in session_queries[:5]:
                findings += f"- {q.get('query', '')[:80]}...\n"
            return findings

        # Prepare conversation for LLM
        lines = []
        for q in session_queries:
            query_text = q.get("query", "Unknown")
            response_text = q.get("response", "No response")
            lines.append(f"Query: {query_text}\nResponse: {response_text}\n")
        
        conversation = "\n".join(lines)
        current_memory = self.read_long_term()

        prompt = f"""You are a research memory consolidation agent. Process these research queries and their findings to update the long-term research memory.

Return a JSON object with exactly two keys:
1. "summary": A concise paragraph (3-5 sentences) summarizing the key research discoveries, confirmed facts, and any remaining uncertainties from this session.
2. "research_memory_update": The updated long-term research memory in Markdown format. Incorporate new findings, resolve contradictions if evidence is strong, and maintain a structured list of "Confirmed Facts" and "Active Research Questions".

## Current Long-term Research Memory
{current_memory or "(empty)"}

## New Research Session Data
{conversation}

**IMPORTANT**: Respond with ONLY valid JSON, no markdown fences.
"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            import json_repair

            response = await llm.ainvoke([
                SystemMessage(content="You are a research consolidation expert. Respond only with valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            text = response.content.strip()
            # Basic cleanup of markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                
            result = json_repair.loads(text)
            
            summary = result.get("summary", "Session consolidated.")
            updated_memory = result.get("research_memory_update", current_memory)
            
            if updated_memory and updated_memory != current_memory:
                self.write_long_term(updated_memory)
                logger.info("Research memory updated with consolidated findings")
                
            return summary
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return f"Error during consolidation: {str(e)}"
