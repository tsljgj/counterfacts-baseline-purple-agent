import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--model", type=str, help="Foundation model to use (e.g., qwen/qwen-2.5-72b-instruct). Overrides AGENT_MODEL env var.")
    args = parser.parse_args()

    # AQA Baseline Purple Agent Card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/

    skill = AgentSkill(
        id="qa_answering_with_tools",
        name="Advanced Question Answering",
        description="Answer questions using AI with web search, web page visiting, and Python code execution capabilities",
        tags=["qa", "web-search", "code-execution", "research", "baseline", "purple-agent"],
        examples=[
            "What is the current stock price of Apple?",
            "Search for recent papers on quantum computing and summarize the key findings",
            "Calculate the compound interest on $10,000 at 5% for 10 years",
            "What are the latest developments in AI safety research?",
            "Visit https://example.com and extract the main points",
            "Write and execute code to find prime numbers between 1 and 100"
        ]
    )

    agent_card = AgentCard(
        name="AQA Baseline Purple Agent",
        description="A baseline purple agent for AgentBeats competition that answers questions using AI with tools: web search (SerpAPI), web page visiting, and Python REPL execution. Designed for evaluation by green agents.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(model=args.model),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
