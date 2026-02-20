"""Tests for multi-agent system: roles, shared context, orchestrator."""
import asyncio
import json
import pytest

from neurostudio.backend.agent.roles import (
    AgentRole, get_role, list_roles, list_roles_dict,
    PLANNER, CODER, REVIEWER, RESEARCHER, BUILTIN_ROLES,
)
from neurostudio.backend.agent.shared_context import (
    SharedContext, Subtask, SubtaskStatus, AgentMessage, MessageType,
)
from neurostudio.backend.agent.orchestrator import MultiAgentOrchestrator


# ════════════════════════════════════════════
#  Agent Roles
# ════════════════════════════════════════════

class TestAgentRole:
    def test_role_has_required_fields(self):
        role = AgentRole(
            role_id="test",
            name="Test",
            description="Test role",
            system_prompt="You are a test.",
        )
        assert role.role_id == "test"
        assert role.name == "Test"
        assert role.system_prompt == "You are a test."
        assert role.allowed_tools == []

    def test_role_to_dict(self):
        role = CODER
        d = role.to_dict()
        assert d["role_id"] == "coder"
        assert d["name"] == "Coder"
        assert "read_file" in d["allowed_tools"]
        assert d["color"] == "#00b894"

    def test_role_with_tools(self):
        assert "read_file" in CODER.allowed_tools
        assert "execute_code" in CODER.allowed_tools
        assert "web_search" not in CODER.allowed_tools

    def test_role_with_preferred_provider(self):
        role = AgentRole(
            role_id="custom",
            name="Custom",
            description="Custom role",
            system_prompt="prompt",
            preferred_provider="openai",
            preferred_model="gpt-4o",
        )
        assert role.preferred_provider == "openai"
        assert role.preferred_model == "gpt-4o"


class TestBuiltinRoles:
    def test_planner_has_no_tools(self):
        assert PLANNER.allowed_tools == []
        assert "plan" in PLANNER.description.lower() or "planer" in PLANNER.description.lower()

    def test_coder_has_filesystem_tools(self):
        assert "read_file" in CODER.allowed_tools
        assert "execute_code" in CODER.allowed_tools

    def test_reviewer_is_read_only(self):
        assert "read_file" in REVIEWER.allowed_tools
        assert "execute_code" not in REVIEWER.allowed_tools
        assert "run_command" not in REVIEWER.allowed_tools

    def test_researcher_has_web_tools(self):
        assert "web_search" in RESEARCHER.allowed_tools
        assert "web_fetch" in RESEARCHER.allowed_tools
        assert "execute_code" not in RESEARCHER.allowed_tools

    def test_all_roles_have_unique_ids(self):
        ids = [r.role_id for r in list_roles()]
        assert len(ids) == len(set(ids))

    def test_all_roles_have_unique_colors(self):
        colors = [r.color for r in list_roles()]
        assert len(colors) == len(set(colors))


class TestRoleRegistry:
    def test_get_role_by_id(self):
        role = get_role("coder")
        assert role is not None
        assert role.role_id == "coder"

    def test_get_unknown_role(self):
        assert get_role("nonexistent") is None

    def test_list_roles(self):
        roles = list_roles()
        assert len(roles) == 4
        ids = {r.role_id for r in roles}
        assert ids == {"planner", "coder", "reviewer", "researcher"}

    def test_list_roles_dict(self):
        roles = list_roles_dict()
        assert isinstance(roles, list)
        assert all(isinstance(r, dict) for r in roles)
        assert all("role_id" in r for r in roles)

    def test_builtin_roles_dict(self):
        assert "planner" in BUILTIN_ROLES
        assert "coder" in BUILTIN_ROLES
        assert "reviewer" in BUILTIN_ROLES
        assert "researcher" in BUILTIN_ROLES


# ════════════════════════════════════════════
#  Shared Context
# ════════════════════════════════════════════

class TestSharedContext:
    def test_create_context(self):
        ctx = SharedContext("wf_test", "Build a website")
        assert ctx.workflow_id == "wf_test"
        assert ctx.original_task == "Build a website"
        assert ctx.status == "running"
        assert ctx.subtasks == []

    def test_add_subtask(self):
        ctx = SharedContext("wf1", "task")
        st = ctx.add_subtask("Write code", "coder", "Write the main module")
        assert st.id == 1
        assert st.title == "Write code"
        assert st.role == "coder"
        assert st.status == SubtaskStatus.PENDING

    def test_add_multiple_subtasks(self):
        ctx = SharedContext("wf1", "task")
        st1 = ctx.add_subtask("Step 1", "coder", "First step")
        st2 = ctx.add_subtask("Step 2", "reviewer", "Second step", depends_on=[1])
        assert st1.id == 1
        assert st2.id == 2
        assert st2.depends_on == [1]

    def test_get_subtask(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step 1", "coder", "desc")
        st = ctx.get_subtask(1)
        assert st is not None
        assert st.title == "Step 1"
        assert ctx.get_subtask(99) is None


class TestSubtaskDependencies:
    def test_ready_subtasks_no_deps(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step 1", "coder", "desc")
        ctx.add_subtask("Step 2", "reviewer", "desc")
        ready = ctx.get_ready_subtasks()
        assert len(ready) == 2

    def test_ready_subtasks_with_deps(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step 1", "coder", "desc")
        ctx.add_subtask("Step 2", "reviewer", "desc", depends_on=[1])
        ready = ctx.get_ready_subtasks()
        assert len(ready) == 1
        assert ready[0].id == 1

    def test_ready_after_completion(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step 1", "coder", "desc")
        ctx.add_subtask("Step 2", "reviewer", "desc", depends_on=[1])
        ctx.mark_subtask_running(1)
        ctx.mark_subtask_completed(1, "Done!")
        ready = ctx.get_ready_subtasks()
        assert len(ready) == 1
        assert ready[0].id == 2

    def test_no_ready_when_dep_running(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step 1", "coder", "desc")
        ctx.add_subtask("Step 2", "reviewer", "desc", depends_on=[1])
        ctx.mark_subtask_running(1)
        ready = ctx.get_ready_subtasks()
        assert len(ready) == 0


class TestSubtaskStatus:
    def test_mark_running(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step", "coder", "desc")
        ctx.mark_subtask_running(1)
        st = ctx.get_subtask(1)
        assert st.status == SubtaskStatus.RUNNING
        assert st.started_at is not None

    def test_mark_completed(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step", "coder", "desc")
        ctx.mark_subtask_running(1)
        ctx.mark_subtask_completed(1, "Result text")
        st = ctx.get_subtask(1)
        assert st.status == SubtaskStatus.COMPLETED
        assert st.result == "Result text"
        assert st.completed_at is not None

    def test_mark_failed(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step", "coder", "desc")
        ctx.mark_subtask_running(1)
        ctx.mark_subtask_failed(1, "Something went wrong")
        st = ctx.get_subtask(1)
        assert st.status == SubtaskStatus.FAILED
        assert st.error == "Something went wrong"

    def test_all_done(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step 1", "coder", "desc")
        ctx.add_subtask("Step 2", "reviewer", "desc")
        assert not ctx.all_done
        ctx.mark_subtask_completed(1, "ok")
        assert not ctx.all_done
        ctx.mark_subtask_completed(2, "ok")
        assert ctx.all_done

    def test_all_done_with_failed(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Step 1", "coder", "desc")
        ctx.add_subtask("Step 2", "reviewer", "desc")
        ctx.mark_subtask_completed(1, "ok")
        ctx.mark_subtask_failed(2, "error")
        assert ctx.all_done


class TestProgress:
    def test_progress_empty(self):
        ctx = SharedContext("wf1", "task")
        p = ctx.progress
        assert p["total"] == 0
        assert p["percent"] == 0

    def test_progress_tracking(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("S1", "coder", "d")
        ctx.add_subtask("S2", "reviewer", "d")
        ctx.add_subtask("S3", "researcher", "d")
        ctx.add_subtask("S4", "coder", "d")

        p = ctx.progress
        assert p["total"] == 4
        assert p["completed"] == 0
        assert p["percent"] == 0

        ctx.mark_subtask_running(1)
        p = ctx.progress
        assert p["running"] == 1

        ctx.mark_subtask_completed(1, "ok")
        ctx.mark_subtask_completed(2, "ok")
        p = ctx.progress
        assert p["completed"] == 2
        assert p["percent"] == 50


# ════════════════════════════════════════════
#  Message Passing
# ════════════════════════════════════════════

class TestMessagePassing:
    def test_send_message(self):
        ctx = SharedContext("wf1", "task")
        msg = ctx.send_message("coder", "reviewer", MessageType.RESULT, "Code is ready")
        assert msg.msg_id == 1
        assert msg.from_agent == "coder"
        assert msg.to_agent == "reviewer"
        assert msg.content == "Code is ready"

    def test_get_messages_for_agent(self):
        ctx = SharedContext("wf1", "task")
        ctx.send_message("planner", "coder", MessageType.TASK, "Write code")
        ctx.send_message("planner", "reviewer", MessageType.TASK, "Review code")
        ctx.send_message("coder", "all", MessageType.INFO, "Code done")

        coder_msgs = ctx.get_messages_for("coder")
        assert len(coder_msgs) == 2  # direct + broadcast

        reviewer_msgs = ctx.get_messages_for("reviewer")
        assert len(reviewer_msgs) == 2  # direct + broadcast

    def test_message_types(self):
        assert MessageType.TASK == "task"
        assert MessageType.RESULT == "result"
        assert MessageType.QUESTION == "question"
        assert MessageType.ERROR == "error"

    def test_all_messages(self):
        ctx = SharedContext("wf1", "task")
        ctx.send_message("a", "b", MessageType.INFO, "msg1")
        ctx.send_message("b", "a", MessageType.INFO, "msg2")
        assert len(ctx.get_all_messages()) == 2

    def test_message_to_dict(self):
        ctx = SharedContext("wf1", "task")
        msg = ctx.send_message("coder", "reviewer", MessageType.RESULT, "Done")
        d = msg.to_dict()
        assert d["from_agent"] == "coder"
        assert d["msg_type"] == "result"


# ════════════════════════════════════════════
#  Artifacts
# ════════════════════════════════════════════

class TestArtifacts:
    def test_set_and_get_artifact(self):
        ctx = SharedContext("wf1", "task")
        ctx.set_artifact("code", "def hello(): pass")
        assert ctx.get_artifact("code") == "def hello(): pass"

    def test_get_missing_artifact(self):
        ctx = SharedContext("wf1", "task")
        assert ctx.get_artifact("nonexistent") is None

    def test_list_artifacts(self):
        ctx = SharedContext("wf1", "task")
        ctx.set_artifact("code", "...")
        ctx.set_artifact("review", "...")
        keys = ctx.list_artifacts()
        assert "code" in keys
        assert "review" in keys

    def test_overwrite_artifact(self):
        ctx = SharedContext("wf1", "task")
        ctx.set_artifact("code", "v1")
        ctx.set_artifact("code", "v2")
        assert ctx.get_artifact("code") == "v2"


# ════════════════════════════════════════════
#  Context Summary
# ════════════════════════════════════════════

class TestContextSummary:
    def test_summary_includes_task(self):
        ctx = SharedContext("wf1", "Build a website")
        summary = ctx.build_context_summary("coder")
        assert "Build a website" in summary

    def test_summary_includes_completed_results(self):
        ctx = SharedContext("wf1", "task")
        ctx.add_subtask("Research", "researcher", "desc")
        ctx.mark_subtask_completed(1, "Found useful information")
        summary = ctx.build_context_summary("coder")
        assert "Found useful information" in summary

    def test_summary_includes_messages(self):
        ctx = SharedContext("wf1", "task")
        ctx.send_message("planner", "coder", MessageType.TASK, "Write the code")
        summary = ctx.build_context_summary("coder")
        assert "Write the code" in summary

    def test_summary_includes_artifacts(self):
        ctx = SharedContext("wf1", "task")
        ctx.set_artifact("data.csv", "some data")
        summary = ctx.build_context_summary("researcher")
        assert "data.csv" in summary


# ════════════════════════════════════════════
#  Serialization
# ════════════════════════════════════════════

class TestSerialization:
    def test_subtask_to_dict(self):
        st = Subtask(1, "Test task", "coder", "Description", depends_on=[])
        d = st.to_dict()
        assert d["id"] == 1
        assert d["title"] == "Test task"
        assert d["role"] == "coder"
        assert d["status"] == "pending"

    def test_context_to_dict(self):
        ctx = SharedContext("wf_test", "Build something")
        ctx.add_subtask("Step 1", "coder", "desc")
        ctx.send_message("a", "b", MessageType.INFO, "hello")
        ctx.set_artifact("code", "...")

        d = ctx.to_dict()
        assert d["workflow_id"] == "wf_test"
        assert d["original_task"] == "Build something"
        assert len(d["subtasks"]) == 1
        assert len(d["messages"]) == 1
        assert "code" in d["artifacts"]
        assert d["status"] == "running"
        assert "progress" in d


# ════════════════════════════════════════════
#  Orchestrator (unit tests, no inference)
# ════════════════════════════════════════════

class TestOrchestratorParsePlan:
    def setup_method(self):
        self.orch = MultiAgentOrchestrator()

    def test_parse_valid_json_plan(self):
        content = '{"subtasks": [{"title": "Write code", "role": "coder", "description": "desc", "depends_on": []}]}'
        result = self.orch._parse_plan(content)
        assert len(result) == 1
        assert result[0]["title"] == "Write code"
        assert result[0]["role"] == "coder"

    def test_parse_json_with_surrounding_text(self):
        content = 'Here is my plan:\n{"subtasks": [{"title": "Step 1", "role": "coder", "description": "d"}]}\nEnd.'
        result = self.orch._parse_plan(content)
        assert len(result) == 1
        assert result[0]["title"] == "Step 1"

    def test_parse_json_array(self):
        content = '[{"title": "A", "role": "coder", "description": "d"}, {"title": "B", "role": "reviewer", "description": "d"}]'
        result = self.orch._parse_plan(content)
        assert len(result) == 2

    def test_parse_invalid_json_fallback(self):
        content = "I cannot generate JSON but here is a plan: do everything"
        result = self.orch._parse_plan(content)
        assert len(result) == 1
        assert result[0]["role"] == "coder"

    def test_parse_multi_subtask_plan(self):
        plan = json.dumps({
            "subtasks": [
                {"id": 1, "title": "Research", "role": "researcher", "description": "Find info", "depends_on": []},
                {"id": 2, "title": "Code it", "role": "coder", "description": "Write code", "depends_on": [1]},
                {"id": 3, "title": "Review", "role": "reviewer", "description": "Check code", "depends_on": [2]},
            ]
        })
        result = self.orch._parse_plan(plan)
        assert len(result) == 3
        assert result[2]["depends_on"] == [2]


class TestOrchestratorWorkflowManagement:
    def test_list_workflows_empty(self):
        orch = MultiAgentOrchestrator()
        assert orch.list_workflows() == []

    def test_get_workflow_missing(self):
        orch = MultiAgentOrchestrator()
        assert orch.get_workflow("nonexistent") is None

    def test_cleanup_old_workflows(self):
        orch = MultiAgentOrchestrator()
        ctx = SharedContext("wf_old", "old task")
        ctx.created_at = 0  # Very old
        ctx.status = "completed"
        orch.active_workflows["wf_old"] = ctx

        ctx2 = SharedContext("wf_new", "new task")
        ctx2.status = "completed"
        orch.active_workflows["wf_new"] = ctx2

        orch.cleanup_old_workflows(max_age_seconds=10)
        assert "wf_old" not in orch.active_workflows
        assert "wf_new" in orch.active_workflows

    def test_cleanup_preserves_running(self):
        orch = MultiAgentOrchestrator()
        ctx = SharedContext("wf_running", "running task")
        ctx.created_at = 0  # Very old
        ctx.status = "running"  # But still running
        orch.active_workflows["wf_running"] = ctx

        orch.cleanup_old_workflows(max_age_seconds=10)
        assert "wf_running" in orch.active_workflows
