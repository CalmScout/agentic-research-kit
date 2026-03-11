"""
Example: Analyzing Academic Literature on Bias in Biomedical AI

This example demonstrates how to use Agentic Research Kit (ARK) to:
1. Ingest academic papers (PDFs) on bias in biomedical AI
2. Query the system to analyze how bias has evolved over time
3. Use the research prompt template for academic analysis

Primary Use Case: Literature review and trend analysis in academic research.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.workflow import query_with_agents
from data_ingestion.universal_pipeline import UniversalIngestionPipeline


async def main():
    """Main workflow for analyzing bias in biomedical AI literature."""

    # ========================================================================
    # Step 1: Ingest Academic Papers (PDFs)
    # ========================================================================
    print("=" * 80)
    print("STEP 1: Ingesting Academic Papers")
    print("=" * 80)

    # Initialize the universal ingestion pipeline
    pipeline = UniversalIngestionPipeline(
        working_dir="./rag_storage/biased_ai",
        use_gpu=True  # Use GPU if available for faster embedding generation
    )

    # Ingest all PDF papers from the directory
    # You would place your PDF papers in ./papers/biased_ai/ directory
    papers_dir = "./papers/biased_ai"

    if Path(papers_dir).exists():
        print(f"\n📁 Ingesting PDFs from: {papers_dir}")

        stats = await pipeline.ingest_directory(
            dir_path=papers_dir,
            pattern="**/*.pdf",  # Recursively find all PDFs
            recursive=True
        )

        print("\n✅ Ingestion Complete:")
        print(f"   - Files processed: {stats['successful_files']}")
        print(f"   - Total chunks: {stats['total_chunks']}")
        print(f"   - Items ingested: {stats['ingest_stats']['total_items']}")
    else:
        print(f"\n⚠️  Papers directory not found: {papers_dir}")
        print("   Create this directory and add PDF papers for analysis.")
        print("   Continuing with any existing data in RAG storage...")

    # ========================================================================
    # Step 2: Query for Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Analyzing Bias Evolution")
    print("=" * 80)

    # Define research queries about bias in biomedical AI
    research_queries = [
        "How has bias in biomedical AI evolved from 2010 to 2026?",
        "What types of bias are most commonly discussed in biomedical AI literature?",
        "How do researchers suggest mitigating bias in medical AI systems?",
        "What are the key datasets used to study bias in healthcare AI?",
    ]

    for i, query in enumerate(research_queries, 1):
        print(f"\n🔍 Query {i}: {query}")

        try:
            # Query the system using the research prompt template
            result = await query_with_agents(
                query=query,
                prompt_template="research",  # Use research-focused prompt
                debug=False
            )

            print("\n📊 Response:")
            print(f"   {result['response'][:500]}...")  # Show first 500 chars
            print(f"\n   Confidence: {result['confidence']:.1%}")
            print(f"   Sources: {len(result.get('sources', []))} documents")

        except Exception as e:
            print(f"\n❌ Query failed: {e}")

    # ========================================================================
    # Step 3: Detailed Analysis Example
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Detailed Analysis")
    print("=" * 80)

    # More specific research query
    detailed_query = """
    Analyze the progression of bias research in biomedical AI:
    1. What were the primary concerns in early research (2010-2015)?
    2. How did the focus shift in later years (2016-2020)?
    3. What are the emerging themes in recent research (2021-2026)?
    4. Identify key milestone papers that shaped the discourse.
    """

    print(f"\n🔍 Detailed Query:{detailed_query}")

    try:
        result = await query_with_agents(
            query=detailed_query,
            prompt_template="analysis",  # Use analysis-focused prompt
            debug=False
        )

        print("\n📊 Analysis Results:")
        print(f"   {result['response']}")
        print(f"\n   Confidence: {result['confidence']:.1%}")

        # Show top sources
        if result.get('sources'):
            print("\n   Top Sources:")
            for j, source in enumerate(result['sources'][:5], 1):
                title = source.get('title', source.get('content', 'N/A'))[:60]
                print(f"      {j}. {title}...")

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   Agentic Research Kit: Bias in Biomedical AI Literature Analysis           ║
║                                                                              ║
║   This example demonstrates literature review and trend analysis           ║
║   using academic papers on bias in biomedical AI systems.                  ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(main())
