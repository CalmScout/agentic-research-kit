import asyncio

from src.agents.workflow import query_with_agents
from src.utils.logger import logger


async def test_react_loop():
    # We'll use a query that is likely to need refinement if the first pass is too narrow
    query = "What are the latest breakthroughs in ambient superconductor research as of 2025?"

    logger.info(f"Testing ReAct loop with query: {query}")

    # Run the workflow
    result = await query_with_agents(query, debug=True)

    print("\n" + "="*50)
    print("FINAL RESPONSE:")
    print(result.get("response"))
    print("="*50)

    print(f"\nVerification Status: {result.get('verification_status')}")
    print(f"Verification Feedback: {result.get('verification_feedback')}")
    print(f"Sources found: {len(result.get('sources', []))}")

    # Check iteration count
    iteration_count = result.get("iteration_count", 0)
    print(f"Number of iterations: {iteration_count}")

    if iteration_count > 1:
        print("SUCCESS: ReAct loop was triggered!")
    else:
        # Check if messages contain multiple verification steps (fallback)
        messages = result.get("messages", [])
        verification_steps = [m for m in messages if getattr(m, "name", "") == "verification_agent"]
        if len(verification_steps) > 1:
            print("SUCCESS: ReAct loop was triggered (detected via messages)!")
        else:
            print("INFO: ReAct loop was not triggered (maybe the first pass was sufficient).")

if __name__ == "__main__":
    asyncio.run(test_react_loop())
