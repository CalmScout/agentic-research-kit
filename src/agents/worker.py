"""
Async worker that bridges the MessageBus and the LangGraph workflow.
Listens for InboundMessages and executes the research pipeline with real-time feedback.
"""

import asyncio
import traceback
from pathlib import Path

from src.agents.base_state import BaseAgentState
from src.agents.bus import InboundMessage, MessageBus, OutboundMessage
from src.agents.memory import MemoryStore
from src.agents.workflow import create_multi_agent_workflow
from src.utils.logger import logger


class AgentWorker:
    """
    Background worker that pulls tasks from the MessageBus and executes them
    using the LangGraph multi-agent workflow.
    """

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._main_task: asyncio.Task | None = None
        self._is_running = False
        self._workflow_app = create_multi_agent_workflow()
        self._workspace_path = Path("./workspace")
        self._memory = MemoryStore(self._workspace_path)

    async def start(self):
        """Start the background worker loop."""
        if self._is_running:
            return

        self._is_running = True
        self._main_task = asyncio.create_task(self._worker_loop())
        logger.info("AgentWorker started and listening for tasks")

    async def stop(self):
        """Stop the background worker loop."""
        self._is_running = False
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        logger.info("AgentWorker stopped")

    async def _worker_loop(self):
        """Main loop that pulls from inbound queue and executes workflow."""
        while self._is_running:
            try:
                # Pull the next message from the bus
                message = await self.bus.get_inbound()
                await self._process_message(message)
                self.bus.task_done_inbound()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker Loop Error: {e}")
                logger.debug(traceback.format_exc())
                await asyncio.sleep(1)

    async def _process_message(self, message: InboundMessage):
        """Execute the LangGraph workflow with streaming updates."""
        session_id = message.session_id
        query = message.content

        logger.info(f"Worker: Processing query for session {session_id}")

        # Initialize context from memory
        context = self._memory.get_research_context(query)

        # Initial state
        initial_state: BaseAgentState = {
            "query": query,
            "query_image": message.metadata.get("query_image"),
            "retrieval_mode": message.metadata.get("retrieval_mode", "hybrid"),
            "memory_context": context,
            "session_id": session_id,
            "query_type": "text",
            "entities": [],
            "query_embedding": [],
            "retrieved_docs": [],
            "retrieval_scores": [],
            "retrieval_method": "keyword",
            "reranked_docs": [],
            "evidence_summary": "",
            "top_results": [],
            "response": "",
            "sources": [],
            "verification_status": "pending",
            "verification_feedback": "",
            "iteration_count": 0,
            "messages": [],
            "metadata": {},
        }

        try:
            # Emit "starting" event
            await self.bus.put_outbound(
                OutboundMessage(
                    session_id=session_id,
                    channel=message.channel,
                    chat_id=message.chat_id,
                    user_id=message.user_id,
                    content="Starting research workflow...",
                    status="thinking",
                )
            )

            final_state = initial_state

            # Use LangGraph .astream to get node completion updates
            async for event in self._workflow_app.astream(initial_state):
                # event is a dict like {"node_name": {state_updates}}
                node_name = list(event.keys())[0]
                updates = event[node_name]
                final_state.update(updates)

                # Emit specific updates based on node
                if node_name == "enhanced_retriever":
                    entities = updates.get("entities", [])
                    entity_str = (
                        f" (detected entities: {', '.join(entities[:3])})" if entities else ""
                    )
                    await self.bus.put_outbound(
                        OutboundMessage(
                            session_id=session_id,
                            channel=message.channel,
                            chat_id=message.chat_id,
                            user_id=message.user_id,
                            content=f"Researcher: Finished gathering evidence{entity_str}.",
                            status="thinking",
                        )
                    )
                elif node_name == "enhanced_response_generator":
                    await self.bus.put_outbound(
                        OutboundMessage(
                            session_id=session_id,
                            channel=message.channel,
                            chat_id=message.chat_id,
                            user_id=message.user_id,
                            content="Synthesizer: Drafting response based on evidence...",
                            status="thinking",
                        )
                    )
                elif node_name == "verification_agent":
                    status = updates.get("verification_status")
                    if status == "verified":
                        await self.bus.put_outbound(
                            OutboundMessage(
                                session_id=session_id,
                                channel=message.channel,
                                chat_id=message.chat_id,
                                user_id=message.user_id,
                                content="Critic: Fact-check passed. Finalizing report.",
                                status="thinking",
                            )
                        )
                    else:
                        await self.bus.put_outbound(
                            OutboundMessage(
                                session_id=session_id,
                                channel=message.channel,
                                chat_id=message.chat_id,
                                user_id=message.user_id,
                                content=f"Critic: Refinement requested ({status}). Re-evaluating sources...",
                                status="thinking",
                            )
                        )

            # Final step: Emit final response
            response_text = final_state.get("response", "")
            sources = final_state.get("retrieved_docs", [])
            if sources:
                response_text += "\n\n**Sources:**\n"
                for i, src in enumerate(sources[:3], 1):
                    url = src.get("url", "N/A")
                    response_text += f"{i}. {url}\n"

            # Update memory history
            self._memory.append_query_history(query, final_state, session_id=session_id)

            await self.bus.put_outbound(
                OutboundMessage(
                    session_id=session_id,
                    channel=message.channel,
                    chat_id=message.chat_id,
                    user_id=message.user_id,
                    content=response_text,
                    status="complete",
                    metadata={
                        "sources": sources,
                        "verification_status": final_state.get("verification_status"),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            logger.debug(traceback.format_exc())
            await self.bus.put_outbound(
                OutboundMessage(
                    session_id=session_id,
                    channel=message.channel,
                    chat_id=message.chat_id,
                    user_id=message.user_id,
                    content=f"An unexpected error occurred during research: {str(e)}",
                    status="error",
                )
            )
