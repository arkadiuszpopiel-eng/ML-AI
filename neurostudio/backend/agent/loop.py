"""
Agent Loop - the core reasoning loop that connects the LLM with tools.

Flow:
1. User sends message
2. Message + tool definitions sent to LLM
3. LLM responds with text and/or tool calls
4. If tool calls: execute tools, send results back to LLM, goto 3
5. If final text: send to user
"""
import json
import logging
from typing import AsyncIterator, Any

from ..config import load_config
from ..inference.engine import engine
from ..inference.router import select_model
from ..tools.base import registry

# Import tools to trigger registration
from ..tools import filesystem, code_executor, web_search, web_fetch, shell, rag_search  # noqa: F401
from ..rag.engine import rag_engine

logger = logging.getLogger("neurostudio.agent")


class AgentLoop:
    """Orchestrates the conversation between user, LLM, and tools."""

    def __init__(self):
        self.conversations: dict[str, list[dict]] = {}

    def get_or_create_conversation(self, session_id: str) -> list[dict]:
        """Get or create a conversation history."""
        if session_id not in self.conversations:
            config = load_config()
            system_prompt = config.get("agent", {}).get("system_prompt", "You are a helpful AI assistant.")

            # Build tool descriptions into system prompt
            tools = registry.list_tools()
            if tools:
                tool_descriptions = "\n\nYou have access to the following tools:\n"
                for tool in tools:
                    tool_descriptions += f"\n- **{tool.name}**: {tool.description}"
                system_prompt += tool_descriptions

            self.conversations[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
        return self.conversations[session_id]

    def clear_conversation(self, session_id: str):
        """Clear a conversation history."""
        self.conversations.pop(session_id, None)

    async def process_message(self, session_id: str, user_message: str) -> AsyncIterator[dict]:
        """
        Process a user message through the agent loop.
        Yields events as dicts:
          - {"type": "text", "content": "..."}       -- text chunk from LLM
          - {"type": "tool_call", "tool": "...", "args": {...}} -- tool being called
          - {"type": "tool_result", "tool": "...", "result": {...}} -- tool result
          - {"type": "model_switch", "from": "...", "to": "..."} -- model was switched
          - {"type": "error", "message": "..."}       -- error
          - {"type": "done"}                           -- conversation turn complete
        """
        config = load_config()
        agent_config = config.get("agent", {})
        max_tool_calls = agent_config.get("max_tool_calls", 10)

        # Check if router suggests a different model
        suggested_model = select_model(user_message)
        if suggested_model and suggested_model != engine.current_model:
            old_model = engine.current_model
            yield {"type": "model_switch", "from": old_model, "to": suggested_model}
            success = await engine.start(suggested_model)
            if not success:
                yield {"type": "error", "message": f"Failed to switch to model: {suggested_model}"}
                return

        if not engine.is_running:
            yield {"type": "error", "message": "No model is loaded. Please load a model first."}
            return

        conversation = self.get_or_create_conversation(session_id)

        # Inject RAG context if relevant documents exist
        rag_context = rag_engine.get_context_for_query(user_message, max_chars=2000)
        if rag_context:
            augmented_msg = (
                f"{user_message}\n\n"
                f"[Relevant context from indexed documents]:\n{rag_context}"
            )
            conversation.append({"role": "user", "content": augmented_msg})
        else:
            conversation.append({"role": "user", "content": user_message})

        tools_schema = registry.to_openai_tools()
        tool_call_count = 0

        while tool_call_count < max_tool_calls:
            try:
                # Get LLM response (non-streaming for tool call detection)
                response = await engine.chat_completion(
                    messages=conversation,
                    tools=tools_schema if tools_schema else None,
                    stream=False,
                )
            except Exception as e:
                logger.error("Inference error: %s", e)
                yield {"type": "error", "message": f"Inference error: {str(e)}"}
                return

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Add assistant message to conversation
            conversation.append(message)

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                for tc in tool_calls:
                    tool_call_count += 1
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    try:
                        tool_args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        tool_args = {}

                    yield {"type": "tool_call", "tool": tool_name, "args": tool_args, "id": tc.get("id")}

                    # Execute the tool
                    tool = registry.get(tool_name)
                    if tool:
                        try:
                            result = await tool.execute(**tool_args)
                        except Exception as e:
                            result = {"success": False, "error": f"Tool execution error: {str(e)}"}
                    else:
                        result = {"success": False, "error": f"Unknown tool: {tool_name}"}

                    yield {"type": "tool_result", "tool": tool_name, "result": result}

                    # Add tool result to conversation
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
            else:
                # No tool calls - this is the final response
                content = message.get("content", "")
                if content:
                    yield {"type": "text", "content": content}
                break

        if tool_call_count >= max_tool_calls:
            yield {"type": "text", "content": f"\n\n[Reached maximum of {max_tool_calls} tool calls for this turn.]"}

        yield {"type": "done"}

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self.conversations.keys())


# Singleton instance
agent = AgentLoop()
