"""Baseline purple agent for AQA benchmark - uses OpenAI to answer questions."""
import os
from openai import AsyncOpenAI
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class Agent:
    """Simple baseline agent that uses OpenAI to answer questions."""

    def __init__(self):
        self.messenger = Messenger()
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. Agent will fail on requests.")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Answer questions using OpenAI.

        Args:
            message: The incoming message containing a question
            updater: Report progress and results
        """
        question = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Processing question...")
        )

        try:
            # Call OpenAI to get an answer
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions concisely and accurately. Provide direct, short answers without extra explanation unless necessary."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=0.0,  # Deterministic answers
                max_tokens=150
            )

            answer = response.choices[0].message.content
            if not answer:
                answer = "I don't know."

        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            answer = f"Error: Unable to generate answer ({str(e)})"

        # Return the answer
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=answer))],
            name="Answer",
        )
