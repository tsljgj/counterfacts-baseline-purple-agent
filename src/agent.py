"""Baseline purple agent for AQA benchmark - uses AQAAgentManager with tools."""
import os
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from agent_wrapper import AQAAgentManager


class Agent:
    """Purple agent that uses AQAAgentManager with web search, web visit, and code execution."""

    def __init__(self):
        self.messenger = Messenger()

        # Initialize AQAAgentManager with configuration
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model = os.getenv("AGENT_MODEL", "qwen/qwen3-235b-a22b-2507")

        if not api_key:
            print("Warning: OPENROUTER_API_KEY not set. Agent will fail on requests.")

        # Initialize the agent manager with tools
        self.agent_manager = AQAAgentManager(
            base_url=base_url,
            api_key=api_key,
            model=model,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            enable_reasoning=True,
            enable_visualization=os.getenv("ENABLE_VISUALIZATION", "false").lower() == "true"
        )

        # System prompt for the agent
        self.system_prompt = (
            "You are a helpful AI assistant that can search the web, visit web pages, "
            "and execute Python code to answer questions accurately. "
            "Always provide complete, well-reasoned answers. "
            "If you execute code, make sure to print the results. "
            "Before finishing, ensure your final response contains the complete answer."
        )

        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Answer questions using AQAAgentManager with tools.

        Args:
            message: The incoming message containing a question
            updater: Report progress and results
        """
        question = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Processing question with tools...")
        )

        try:
            # Use the agent manager to process the question
            answer = await self.agent_manager.run_with_tools(
                prompt=question,
                system_prompt=self.system_prompt,
                max_iterations=self.max_iterations
            )

            if not answer:
                answer = "I was unable to generate an answer."

        except Exception as e:
            print(f"Error running agent: {e}")
            import traceback
            traceback.print_exc()
            answer = f"Error: Unable to generate answer ({str(e)})"

        # Return the answer
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=answer))],
            name="Answer",
        )
