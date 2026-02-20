"""
Shared Context - inter-agent communication and shared memory.

Provides:
- SharedContext: shared state for a multi-agent workflow run
- Message passing between agents (typed messages)
- Artifact storage (code, documents, results from agents)
- Workflow-level metadata and status tracking
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    TASK = "task"           # Task assignment from orchestrator/planner
    RESULT = "result"       # Result from completed subtask
    QUESTION = "question"   # Agent asking another agent
    INFO = "info"           # Informational update
    ERROR = "error"         # Error report


class SubtaskStatus(str, Enum):
    """Status of a subtask in the workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentMessage:
    """A message passed between agents."""
    msg_id: int
    from_agent: str         # role_id of sender
    to_agent: str           # role_id of recipient (or "all")
    msg_type: MessageType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "msg_type": self.msg_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Subtask:
    """A single subtask in a multi-agent workflow."""
    id: int
    title: str
    role: str               # role_id of assigned agent
    description: str
    depends_on: list[int] = field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "role": self.role,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class SharedContext:
    """Shared memory and communication hub for a multi-agent workflow.

    Each workflow run gets its own SharedContext instance.
    Agents read/write artifacts and exchange messages through this object.
    """

    def __init__(self, workflow_id: str, original_task: str):
        self.workflow_id = workflow_id
        self.original_task = original_task
        self.created_at = time.time()

        # Subtask management
        self.subtasks: list[Subtask] = []

        # Inter-agent message log
        self._messages: list[AgentMessage] = []
        self._msg_counter: int = 0

        # Shared artifacts (code, documents, data produced by agents)
        self._artifacts: dict[str, Any] = {}

        # Final synthesis result
        self.final_result: Optional[str] = None
        self.status: str = "running"  # running | completed | failed

    # ── Subtask management ──

    def add_subtask(self, title: str, role: str, description: str,
                    depends_on: list[int] | None = None) -> Subtask:
        """Add a subtask to the workflow plan."""
        task_id = len(self.subtasks) + 1
        subtask = Subtask(
            id=task_id,
            title=title,
            role=role,
            description=description,
            depends_on=depends_on or [],
        )
        self.subtasks.append(subtask)
        return subtask

    def get_subtask(self, task_id: int) -> Subtask | None:
        """Get subtask by ID (1-based)."""
        for st in self.subtasks:
            if st.id == task_id:
                return st
        return None

    def get_ready_subtasks(self) -> list[Subtask]:
        """Get subtasks whose dependencies are all completed (ready to run)."""
        completed_ids = {st.id for st in self.subtasks if st.status == SubtaskStatus.COMPLETED}
        ready = []
        for st in self.subtasks:
            if st.status != SubtaskStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in st.depends_on):
                ready.append(st)
        return ready

    def mark_subtask_running(self, task_id: int):
        """Mark a subtask as running."""
        st = self.get_subtask(task_id)
        if st:
            st.status = SubtaskStatus.RUNNING
            st.started_at = time.time()

    def mark_subtask_completed(self, task_id: int, result: str):
        """Mark a subtask as completed with result."""
        st = self.get_subtask(task_id)
        if st:
            st.status = SubtaskStatus.COMPLETED
            st.result = result
            st.completed_at = time.time()

    def mark_subtask_failed(self, task_id: int, error: str):
        """Mark a subtask as failed."""
        st = self.get_subtask(task_id)
        if st:
            st.status = SubtaskStatus.FAILED
            st.error = error
            st.completed_at = time.time()

    @property
    def all_done(self) -> bool:
        """Check if all subtasks are completed or failed."""
        return all(
            st.status in (SubtaskStatus.COMPLETED, SubtaskStatus.FAILED, SubtaskStatus.SKIPPED)
            for st in self.subtasks
        )

    @property
    def progress(self) -> dict:
        """Get workflow progress summary."""
        total = len(self.subtasks)
        done = sum(1 for st in self.subtasks if st.status == SubtaskStatus.COMPLETED)
        failed = sum(1 for st in self.subtasks if st.status == SubtaskStatus.FAILED)
        running = sum(1 for st in self.subtasks if st.status == SubtaskStatus.RUNNING)
        return {
            "total": total,
            "completed": done,
            "failed": failed,
            "running": running,
            "pending": total - done - failed - running,
            "percent": round(done / total * 100) if total > 0 else 0,
        }

    # ── Message passing ──

    def send_message(self, from_agent: str, to_agent: str,
                     msg_type: MessageType, content: str,
                     metadata: dict | None = None) -> AgentMessage:
        """Send a message between agents."""
        self._msg_counter += 1
        msg = AgentMessage(
            msg_id=self._msg_counter,
            from_agent=from_agent,
            to_agent=to_agent,
            msg_type=msg_type,
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(msg)
        return msg

    def get_messages_for(self, agent_id: str) -> list[AgentMessage]:
        """Get all messages sent to a specific agent (or broadcast)."""
        return [
            m for m in self._messages
            if m.to_agent in (agent_id, "all")
        ]

    def get_all_messages(self) -> list[AgentMessage]:
        """Get all messages in the workflow."""
        return list(self._messages)

    # ── Artifact storage ──

    def set_artifact(self, key: str, value: Any):
        """Store a named artifact (code, data, result)."""
        self._artifacts[key] = value

    def get_artifact(self, key: str) -> Any | None:
        """Get a stored artifact."""
        return self._artifacts.get(key)

    def list_artifacts(self) -> list[str]:
        """List all artifact keys."""
        return list(self._artifacts.keys())

    # ── Context summary for agents ──

    def build_context_summary(self, for_agent: str) -> str:
        """Build a context summary string that an agent receives before its task.

        Includes: original task, completed subtask results, messages for this agent.
        """
        parts = [f"Zadanie glowne: {self.original_task}\n"]

        # Completed subtask results
        completed = [st for st in self.subtasks if st.status == SubtaskStatus.COMPLETED and st.result]
        if completed:
            parts.append("Wyniki poprzednich podzadan:")
            for st in completed:
                result_preview = st.result[:500] if st.result else ""
                parts.append(f"  [{st.role}] {st.title}: {result_preview}")
            parts.append("")

        # Messages for this agent
        messages = self.get_messages_for(for_agent)
        if messages:
            parts.append("Wiadomosci do Ciebie:")
            for msg in messages[-5:]:  # Last 5 messages
                parts.append(f"  [{msg.from_agent} -> {msg.to_agent}] {msg.content[:200]}")
            parts.append("")

        # Relevant artifacts
        artifacts = self.list_artifacts()
        if artifacts:
            parts.append(f"Dostepne artefakty: {', '.join(artifacts)}")

        return "\n".join(parts)

    # ── Serialization ──

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "original_task": self.original_task,
            "created_at": self.created_at,
            "status": self.status,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "messages": [m.to_dict() for m in self._messages],
            "artifacts": list(self._artifacts.keys()),
            "progress": self.progress,
            "final_result": self.final_result,
        }
