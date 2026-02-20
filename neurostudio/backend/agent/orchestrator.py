"""
Multi-Agent Orchestrator - coordinates multiple AI agents working together.

Flow:
1. User sends complex task
2. Planner agent analyzes task and creates subtask plan
3. Orchestrator schedules subtasks based on dependencies
4. Agents execute subtasks (parallel when dependencies allow)
5. Results flow back through shared context
6. Final synthesis combines all agent results

Yields events for real-time UI updates.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

from ..config import load_config
from ..inference.engine import engine
from ..inference.providers import provider_registry
from ..tools.base import registry as tool_registry
from .roles import AgentRole, get_role, PLANNER
from .shared_context import (
    SharedContext,
    Subtask,
    SubtaskStatus,
    MessageType,
)

logger = logging.getLogger("neurostudio.orchestrator")


class MultiAgentOrchestrator:
    """Orchestrates multiple AI agents for complex task decomposition and execution."""

    def __init__(self):
        self.active_workflows: dict[str, SharedContext] = {}

    async def run_workflow(self, task: str, session_id: str) -> AsyncIterator[dict]:
        """
        Run a complete multi-agent workflow.

        Yields events:
          - {"type": "workflow_start", "workflow_id": "...", "task": "..."}
          - {"type": "planning", "status": "..."}
          - {"type": "plan_ready", "subtasks": [...]}
          - {"type": "subtask_start", "subtask": {...}, "agent": "..."}
          - {"type": "subtask_progress", "subtask_id": N, "content": "..."}
          - {"type": "subtask_complete", "subtask_id": N, "result": "..."}
          - {"type": "subtask_error", "subtask_id": N, "error": "..."}
          - {"type": "synthesis", "status": "..."}
          - {"type": "workflow_complete", "result": "...", "progress": {...}}
          - {"type": "workflow_error", "message": "..."}
        """
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        context = SharedContext(workflow_id, task)
        self.active_workflows[workflow_id] = context

        yield {
            "type": "workflow_start",
            "workflow_id": workflow_id,
            "task": task,
        }

        # ── Phase 1: Planning ──
        yield {"type": "planning", "status": "Planner analizuje zadanie..."}

        try:
            plan = await self._run_planner(task, context)
        except Exception as e:
            logger.error("Planning failed: %s", e)
            context.status = "failed"
            yield {"type": "workflow_error", "message": f"Planowanie nie powiodlo sie: {e}"}
            return

        if not context.subtasks:
            context.status = "failed"
            yield {"type": "workflow_error", "message": "Planner nie wygenerowol planu podzadan."}
            return

        yield {
            "type": "plan_ready",
            "subtasks": [st.to_dict() for st in context.subtasks],
        }

        # ── Phase 2: Execute subtasks ──
        max_iterations = len(context.subtasks) * 2  # Safety limit
        iteration = 0

        while not context.all_done and iteration < max_iterations:
            iteration += 1
            ready = context.get_ready_subtasks()

            if not ready:
                # Check if we're stuck (remaining tasks have unmet dependencies)
                pending = [st for st in context.subtasks if st.status == SubtaskStatus.PENDING]
                if pending:
                    # Skip tasks with failed dependencies
                    for st in pending:
                        failed_deps = [
                            d for d in st.depends_on
                            if any(s.id == d and s.status == SubtaskStatus.FAILED
                                   for s in context.subtasks)
                        ]
                        if failed_deps:
                            context.mark_subtask_failed(
                                st.id,
                                f"Pominieto - zaleznosci nie powiodly sie: {failed_deps}",
                            )
                            st.status = SubtaskStatus.SKIPPED
                            yield {
                                "type": "subtask_error",
                                "subtask_id": st.id,
                                "error": f"Pominieto (zaleznosc nieudana)",
                            }
                    continue
                break

            # Run ready subtasks concurrently
            tasks = []
            for subtask in ready:
                context.mark_subtask_running(subtask.id)
                yield {
                    "type": "subtask_start",
                    "subtask": subtask.to_dict(),
                    "agent": subtask.role,
                }
                tasks.append(self._run_subtask(subtask, context))

            # Gather results (parallel execution)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for subtask, result in zip(ready, results):
                if isinstance(result, Exception):
                    error_msg = str(result)
                    context.mark_subtask_failed(subtask.id, error_msg)
                    yield {
                        "type": "subtask_error",
                        "subtask_id": subtask.id,
                        "error": error_msg,
                    }
                else:
                    context.mark_subtask_completed(subtask.id, result)
                    yield {
                        "type": "subtask_complete",
                        "subtask_id": subtask.id,
                        "result": result[:500] if result else "",
                    }

                yield {
                    "type": "progress_update",
                    "progress": context.progress,
                }

        # ── Phase 3: Synthesis ──
        yield {"type": "synthesis", "status": "Lacze wyniki agentow..."}

        try:
            final = await self._synthesize(context)
            context.final_result = final
            context.status = "completed"
        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            # Fallback: just concatenate results
            parts = []
            for st in context.subtasks:
                if st.status == SubtaskStatus.COMPLETED and st.result:
                    parts.append(f"**{st.title}** ({st.role}):\n{st.result}")
            final = "\n\n---\n\n".join(parts) if parts else "Brak wynikow."
            context.final_result = final
            context.status = "completed"

        yield {
            "type": "workflow_complete",
            "workflow_id": workflow_id,
            "result": final,
            "progress": context.progress,
        }

    # ── Internal methods ──

    async def _run_planner(self, task: str, context: SharedContext) -> str:
        """Run the Planner agent to decompose the task."""
        messages = [
            {"role": "system", "content": PLANNER.system_prompt},
            {"role": "user", "content": (
                f"Przeanalizuj to zadanie i stworz plan podzadan:\n\n{task}\n\n"
                f"Dostepne role: coder, reviewer, researcher\n"
                f"Zwroc TYLKO JSON z lista podzadan."
            )},
        ]

        response = await self._inference(messages, PLANNER)

        # Parse planner's JSON response
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        subtasks = self._parse_plan(content)
        for st_data in subtasks:
            context.add_subtask(
                title=st_data.get("title", "Podzadanie"),
                role=st_data.get("role", "coder"),
                description=st_data.get("description", ""),
                depends_on=st_data.get("depends_on", []),
            )

        return content

    def _parse_plan(self, content: str) -> list[dict]:
        """Parse planner's response to extract subtask list."""
        # Try to extract JSON from response
        try:
            # Find JSON block in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                subtasks = data.get("subtasks", [])
                if subtasks:
                    return subtasks
        except json.JSONDecodeError:
            pass

        # Try JSON array directly
        try:
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                subtasks = json.loads(content[start:end])
                if isinstance(subtasks, list):
                    return subtasks
        except json.JSONDecodeError:
            pass

        # Fallback: single task with original content
        logger.warning("Could not parse planner output, creating single subtask")
        return [{"title": "Wykonaj zadanie", "role": "coder", "description": content}]

    async def _run_subtask(self, subtask: Subtask, context: SharedContext) -> str:
        """Run a single subtask with the assigned agent role."""
        role = get_role(subtask.role)
        if not role:
            raise ValueError(f"Nieznana rola: {subtask.role}")

        # Build context-aware prompt
        context_summary = context.build_context_summary(subtask.role)

        messages = [
            {"role": "system", "content": role.system_prompt},
            {"role": "user", "content": (
                f"{context_summary}\n\n"
                f"---\n"
                f"Twoje zadanie: {subtask.title}\n"
                f"Opis: {subtask.description}\n\n"
                f"Wykonaj to zadanie i podaj wynik."
            )},
        ]

        # Build tool schema (filtered by role's allowed tools)
        tools_schema = None
        if role.allowed_tools:
            all_tools = tool_registry.to_openai_tools()
            tools_schema = [t for t in all_tools if t["function"]["name"] in role.allowed_tools]

        response = await self._inference(messages, role, tools_schema)

        # Handle tool calls (single iteration for subtasks)
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            messages.append(message)
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    tool_args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}

                tool = tool_registry.get(tool_name)
                if tool and tool_name in role.allowed_tools:
                    try:
                        result = await tool.execute(**tool_args)
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                else:
                    result = {"success": False, "error": f"Narzedzie niedozwolone: {tool_name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

                # Store tool results as artifacts
                artifact_key = f"subtask_{subtask.id}_{tool_name}"
                context.set_artifact(artifact_key, result)

            # Get final response after tool use
            response = await self._inference(messages, role)
            message = response.get("choices", [{}])[0].get("message", {})

        result_text = message.get("content", "")

        # Store result as message in shared context
        context.send_message(
            from_agent=subtask.role,
            to_agent="all",
            msg_type=MessageType.RESULT,
            content=result_text[:1000],
            metadata={"subtask_id": subtask.id},
        )

        return result_text

    async def _synthesize(self, context: SharedContext) -> str:
        """Synthesize all agent results into a final coherent response."""
        parts = []
        for st in context.subtasks:
            if st.status == SubtaskStatus.COMPLETED and st.result:
                parts.append(f"[{st.role}: {st.title}]\n{st.result}")

        if not parts:
            return "Brak wynikow do polaczenia."

        all_results = "\n\n---\n\n".join(parts)

        messages = [
            {"role": "system", "content": (
                "Jestes syntezatorem. Otrzymujesz wyniki pracy kilku agentow AI. "
                "Polacz je w jedna spojną, czytelna odpowiedz dla uzytkownika. "
                "Zachowaj najwazniejsze informacje. Nie powtarzaj sie."
            )},
            {"role": "user", "content": (
                f"Zadanie oryginalne: {context.original_task}\n\n"
                f"Wyniki agentow:\n\n{all_results}\n\n"
                f"Polacz to w jedna odpowiedz."
            )},
        ]

        response = await self._inference(messages)
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def _inference(self, messages: list[dict], role: AgentRole | None = None,
                         tools_schema: list[dict] | None = None) -> dict:
        """Run inference using the best available provider.

        If the role has a preferred provider/model, try that first.
        Falls back to active provider or local engine.
        """
        # Check if role has preferred provider
        if role and role.preferred_provider:
            provider = provider_registry.get(role.preferred_provider)
            if provider and provider.is_available:
                return await provider.chat_completion(
                    messages=messages,
                    tools=tools_schema,
                    temperature=0.7,
                    model=role.preferred_model,
                )

        # Use active provider (cloud or local)
        active = provider_registry.active_provider
        if active and active.provider_id != "local" and active.is_available:
            return await active.chat_completion(
                messages=messages,
                tools=tools_schema,
                temperature=0.7,
            )

        # Fallback to local engine
        if engine.is_running:
            return await engine.chat_completion(
                messages=messages,
                tools=tools_schema,
                temperature=0.7,
                stream=False,
            )

        raise RuntimeError("Brak dostepnego providera AI. Zaladuj model lub podlacz API.")

    # ── Workflow management ──

    def get_workflow(self, workflow_id: str) -> SharedContext | None:
        """Get a workflow by ID."""
        return self.active_workflows.get(workflow_id)

    def list_workflows(self) -> list[dict]:
        """List all active/completed workflows."""
        return [
            {
                "workflow_id": ctx.workflow_id,
                "task": ctx.original_task[:100],
                "status": ctx.status,
                "progress": ctx.progress,
                "created_at": ctx.created_at,
            }
            for ctx in self.active_workflows.values()
        ]

    def cleanup_old_workflows(self, max_age_seconds: int = 3600):
        """Remove workflows older than max_age_seconds."""
        now = time.time()
        to_remove = [
            wid for wid, ctx in self.active_workflows.items()
            if now - ctx.created_at > max_age_seconds and ctx.status != "running"
        ]
        for wid in to_remove:
            del self.active_workflows[wid]


# Singleton
orchestrator = MultiAgentOrchestrator()
