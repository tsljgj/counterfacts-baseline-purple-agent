# Counterfacts Baseline Purple Agent

A baseline purple agent for the [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats) that uses the [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol. This agent serves as a baseline for testing green (evaluation) agents in the competition.

## Features

This purple agent can:
- **Web Search**: Search the web using SerpAPI/Tavily to find current information
- **Web Page Visiting**: Visit and extract content from web pages
- **Python Code Execution**: Run Python code for calculations, data analysis, and problem-solving

## Project Structure

```
src/
├─ server.py         # Server setup and agent card configuration
├─ executor.py       # A2A request handling
├─ agent.py          # Purple agent implementation
├─ agent_wrapper.py  # Agent manager with tool support
├─ messenger.py      # A2A messaging utilities
└─ tools/            # Tool implementations (web search, visit, code exec)
tests/
└─ test_agent.py     # Agent tests
Dockerfile           # Docker configuration
pyproject.toml       # Python dependencies
.env.example         # Environment variable template
```

## Setup

### 1. Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Required API keys:
- `OPENROUTER_API_KEY`: For the main LLM (get from [OpenRouter](https://openrouter.ai/))
- `SERPAPI_API_KEY`: For web search (get from [SerpAPI](https://serpapi.com/))
- `TAVILY_API_KEY`: For Tavily search (get from [Tavily](https://tavily.com/))
- `OPENAI_API_KEY`: For web page summarization (get from [OpenAI](https://platform.openai.com/))

Optional configuration:
- `AGENT_MODEL`: LLM model to use (default: `qwen/qwen3-235b-a22b-2507`)
- `MAX_ITERATIONS`: Maximum tool calling iterations (default: `10`)
- `DEBUG`: Enable debug mode (default: `false`)
- `ENABLE_VISUALIZATION`: Enable execution visualization (default: `false`)

### 2. Install Dependencies

```bash
# Install dependencies
uv sync

# Install with test dependencies
uv sync --extra test
```

## Running Locally

```bash
# Run the server
uv run src/server.py

# Run on custom host/port
uv run src/server.py --host 0.0.0.0 --port 9009
```

## Running with Docker

```bash
# Build the image
docker build -t counterfacts-baseline-purple-agent .

# Run the container with environment variables
docker run -p 9009:9009 \
  -e OPENROUTER_API_KEY=your-key \
  -e SERPAPI_API_KEY=your-key \
  -e TAVILY_API_KEY=your-key \
  -e OPENAI_API_KEY=your-key \
  counterfacts-baseline-purple-agent

# Or use an env file
docker run -p 9009:9009 --env-file .env counterfacts-baseline-purple-agent
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## AgentBeats Competition Integration

This agent is designed as a baseline purple agent for the AgentBeats competition:

1. **Purple Agent Role**: This agent is evaluated by green agents (benchmark agents) in the competition
2. **A2A Protocol**: Uses the A2A protocol for interoperability with any green agent
3. **Capabilities**: Demonstrates web search, web visiting, and code execution abilities
4. **Baseline Performance**: Serves as a reference point for comparing more advanced agents

### Submitting to AgentBeats

1. Build and publish your Docker image to a container registry
2. Register your agent at [AgentBeats Dashboard](https://agentbeats.dev/)
3. Select "purple" as the agent type
4. Provide your container image reference and repository URL
5. Your agent will be evaluated by green agents from Phase 1

For more information, see the [AgentBeats Tutorial](https://docs.agentbeats.dev/tutorial/).

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

**Important**: Add your API keys as GitHub secrets (Settings → Secrets and variables → Actions → Repository secrets):
- `OPENROUTER_API_KEY`
- `SERPAPI_API_KEY`
- `TAVILY_API_KEY`
- `OPENAI_API_KEY`

These will be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).

## Architecture

The agent uses the following components:

- **AQAAgentManager** ([agent_wrapper.py](src/agent_wrapper.py)): Manages the agent lifecycle, tool execution, and conversation history
- **Agent** ([agent.py](src/agent.py)): A2A protocol interface that wraps the AQAAgentManager
- **Tools** ([src/tools/](src/tools/)):
  - Web search via SerpAPI and Tavily
  - Web page visiting with content extraction
  - Python REPL for code execution
- **Executor** ([executor.py](src/executor.py)): Handles A2A requests and manages agent instances
- **Server** ([server.py](src/server.py)): HTTP server with agent card configuration
