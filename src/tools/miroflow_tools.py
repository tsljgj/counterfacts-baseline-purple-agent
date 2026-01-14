import logging
import os
from typing import Any, Tuple

from openai import AsyncOpenAI

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MiroflowReasoningTool(BaseTool):
    """A tool for solving hard math problem, puzzle, riddle and IQ test questions that require a lot of chain of thought efforts."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        
        self.reasoning_model = config.get("reasoning_model", "o4-mini")
        self.client = AsyncOpenAI(base_url=config.get("reasoning_base_url", "https://api.openai.com/v1"), 
                                  api_key=config.get("reasoning_api_key", os.getenv("OPENAI_API_KEY")))
        self.system_prompt = """\
You are a reasoning expert that performs the task of analysing problems and questions by reasoning and providing your thinking process and complete answer.

Be cautious and transparent in your output:
- Wrap your thinking process in <think> </think> tags and your complete answer in <answer> </answer> tags.
- If the task cannot be solved, say so clearly.
- If more context is needed, return a clarification request and do not proceed with tool use.
"""
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        question = parameters.get("question", "")
        
        for _ in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model=self.reasoning_model,
                    messages=[{"role": "system", "content": self.system_prompt},
                              {"role": "user", "content": question}],
                )
                think, answer = self.extract_think_answer(response.choices[0].message.content)
                return answer, 0.0, {"think": think}
            except Exception as e:
                logger.error(f"Error in reasoning tool: {e}")
                continue
        return "Error in reasoning tool", -0.1, {"error": "Error in reasoning tool"}
    
    def extract_think_answer(self, response: str) -> Tuple[str, str]:
        think = response.split('<think>')[1].split('</think>')[0]
        answer = response.split('<answer>')[1].split('</answer>')[0]
        return think, answer
    
class MiroflowPlanningTool(BaseTool):
    """A tool for making a plan to solve a complex problem or question."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        
        self.planning_model = config.get("planning_model", "o4-mini")
        self.client = AsyncOpenAI(base_url=config.get("planning_base_url", "https://api.openai.com/v1"), 
                                  api_key=config.get("planning_api_key", os.getenv("OPENAI_API_KEY")))
#         self.system_prompt = """\
# You are an agent that performs the task of making a plan to solve a complex problem or question based on the user-supplied context.

# Be cautious and transparent in your output:
# - Wrap your thinking process in <think> </think> tags and your plan in <plan> </plan> tags.
# - If more context is needed, return a clarification request and do not proceed with tool use.
# - If the task cannot be solved, say so clearly.
# """

        self.system_prompt = """\
You are a planning expert that performs the task of making a step-by-step plan for solving a complex problem or question, based on the user-supplied context.

Guidelines for your response:
- Wrap your reasoning in <think> </think> tags and your actionable plan in <plan> </plan> tags.
- Your plan should consist of clear, sequential steps that describe how to approach and solve the problem, not a table of contents or report structure.
- Be concrete, fine-grained, but grounded in the tools and options available to the user. Do not imagine interfaces or interactions that are not described by the user. 
- If context is missing, return a clarification request instead of proceeding.
- If the task cannot be solved, state that clearly.
"""
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        context = parameters.get("context", "")
        
        for _ in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model=self.planning_model,
                    messages=[{"role": "system", "content": self.system_prompt},
                              {"role": "user", "content": context}],
                )
                think, plan = self.extract_think_plan(response.choices[0].message.content)
                return plan, 0.0, {"think": think}
            except Exception as e:
                logger.error(f"Error in planning tool: {e}")
                continue
        return "Error in planning tool", -0.1, {"error": "Error in planning tool"}
    
    def extract_think_plan(self, response: str) -> Tuple[str, str]:
        think = response.split('<think>')[1].split('</think>')[0]
        plan = response.split('<plan>')[1].split('</plan>')[0]
        return think, plan

class MiroflowReflectionTool(BaseTool):
    """A tool for reflecting on the previous tool calls and results step by step, analyze any problems or mistakes made, and suggest how to improve or correct them in the next steps."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        
        self.reflection_model = config.get("reflection_model", "o4-mini")
        self.client = AsyncOpenAI(base_url=config.get("reflection_base_url", "https://api.openai.com/v1"), 
                                  api_key=config.get("reflection_api_key", os.getenv("OPENAI_API_KEY")))
        self.system_prompt = """\
You are an agent that performs the task of reflecting on user-supplied context of previous tool calls and results step by step, analyze any problems or mistakes made, and suggest how to improve or correct them in the next steps.

Be cautious and transparent in your output:
- Wrap your thinking process in <think> </think> tags and your reflection in <reflection> </reflection> tags.
- If more context is needed, return a clarification request and do not proceed with tool use.
"""
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        context = parameters.get("context", "")
        
        for _ in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model=self.reflection_model,
                    messages=[{"role": "system", "content": self.system_prompt},
                              {"role": "user", "content": context}],
                )
                think, reflection = self.extract_think_reflection(response.choices[0].message.content)
                return reflection, 0.0, {"think": think}
            except Exception as e:
                logger.error(f"Error in reflection tool: {e}")
                continue
        return "Error in reflection tool", -0.1, {"error": "Error in reflection tool"}
    
    def extract_think_reflection(self, response: str) -> Tuple[str, str]:
        think = response.split('<think>')[1].split('</think>')[0]
        reflection = response.split('<reflection>')[1].split('</reflection>')[0]
        return think, reflection

