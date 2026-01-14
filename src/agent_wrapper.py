"""
Agent wrapper with comprehensive visualization support
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI
from tools.schemas import OpenAIFunctionToolSchema
from tools.websailor_tools import WebSailorMiroflowMultiWebSearchTool, WebSailorMultiVisitTool
from tools.deer_flow_tool import DeerFlowCodeTool


class AgentLogger:
    """Centralized logger for agent execution tracking"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logs = []
        self.current_iteration = 0
        
    def log_llm_call(self, messages: List[Dict], iteration: int):
        """Log an LLM call with full message history"""
        if not self.enabled:
            return
            
        log_entry = {
            "type": "llm_call",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "messages": self._serialize_messages(messages)
        }
        self.logs.append(log_entry)
        
    def log_llm_response(self, response: Any, iteration: int):
        """Log LLM response"""
        if not self.enabled:
            return
            
        message = response.choices[0].message
        log_entry = {
            "type": "llm_response",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "content": message.content,
            "has_tool_calls": hasattr(message, 'tool_calls') and message.tool_calls is not None,
            "tool_calls": self._serialize_tool_calls(message.tool_calls) if hasattr(message, 'tool_calls') and message.tool_calls else []
        }
        self.logs.append(log_entry)
        
    def log_tool_execution(self, tool_name: str, parameters: Dict, iteration: int):
        """Log tool execution start"""
        if not self.enabled:
            return
            
        log_entry = {
            "type": "tool_execution_start",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "parameters": parameters
        }
        self.logs.append(log_entry)
        
    def log_tool_result(self, tool_name: str, result: str, reward: float, metrics: Dict, iteration: int):
        """Log tool execution result"""
        if not self.enabled:
            return
            
        log_entry = {
            "type": "tool_execution_result",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "result": result[:1000] + "..." if len(str(result)) > 1000 else result,
            "result_length": len(str(result)),
            "reward": reward,
            "metrics": metrics
        }
        self.logs.append(log_entry)
        
    def log_tool_error(self, tool_name: str, error: str, iteration: int):
        """Log tool execution error"""
        if not self.enabled:
            return
            
        log_entry = {
            "type": "tool_execution_error",
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "error": error
        }
        self.logs.append(log_entry)
    
    def _serialize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Serialize messages for logging"""
        serialized = []
        for msg in messages:
            serialized_msg = {
                "role": msg.get("role"),
                "content": msg.get("content")
            }
            if msg.get("tool_calls"):
                serialized_msg["tool_calls"] = self._serialize_tool_calls(msg.get("tool_calls"))
            if msg.get("tool_call_id"):
                serialized_msg["tool_call_id"] = msg.get("tool_call_id")
            serialized.append(serialized_msg)
        return serialized
    
    def _serialize_tool_calls(self, tool_calls) -> List[Dict]:
        """Serialize tool calls"""
        if not tool_calls:
            return []
        
        serialized = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                serialized.append(tc)
            else:
                serialized.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
        return serialized
    
    def print_visualization(self, output_file=None):
        """Print a visual representation of the execution
        
        Args:
            output_file: Optional file path to write output (in addition to console)
        """
        if not self.enabled or not self.logs:
            return
        
        # Build output as string first
        output_lines = []
        output_lines.append("\n" + "="*80)
        output_lines.append("AGENT EXECUTION VISUALIZATION")
        output_lines.append("="*80)
        
        current_iteration = -1
        
        for log in self.logs:
            if log["iteration"] != current_iteration:
                current_iteration = log["iteration"]
                output_lines.append(f"\n{'─'*80}")
                output_lines.append(f"ITERATION {current_iteration}")
                output_lines.append(f"{'─'*80}")
            
            if log["type"] == "llm_call":
                output_lines.append(f"\n🤖 LLM CALL")
                output_lines.append(f"   Messages sent to LLM: {len(log['messages'])}")
                for i, msg in enumerate(log['messages']):
                    output_lines.append(f"\n   Message {i+1} [{msg['role'].upper()}]:")
                    content = msg.get('content', '')
                    if content:
                        output_lines.append(f"      ┌─── RAW CONTENT ───")
                        # Show content with proper line breaks
                        for line in str(content).split('\n'):
                            output_lines.append(f"      │ {line}")
                        output_lines.append(f"      └──────────────────")
                    if msg.get('tool_calls'):
                        output_lines.append(f"      Tool Calls: {len(msg['tool_calls'])}")
                        for tc in msg['tool_calls']:
                            func = tc.get('function', {})
                            output_lines.append(f"         - {func.get('name')}: {func.get('arguments', '')[:150]}")
                    if msg.get('tool_call_id'):
                        output_lines.append(f"      Tool Call ID: {msg['tool_call_id']}")
            
            elif log["type"] == "llm_response":
                output_lines.append(f"\n💭 LLM RESPONSE")
                if log['content']:
                    output_lines.append(f"   ┌─── RAW LLM OUTPUT ───")
                    # Show full content for LLM response
                    for line in str(log['content']).split('\n'):
                        output_lines.append(f"   │ {line}")
                    output_lines.append(f"   └─────────────────────")
                output_lines.append(f"   Has Tool Calls: {log['has_tool_calls']}")
                if log['tool_calls']:
                    output_lines.append(f"   Tool Calls: {len(log['tool_calls'])}")
                    for tc in log['tool_calls']:
                        func = tc.get('function', {})
                        output_lines.append(f"      - {func.get('name')}")
                        args = func.get('arguments', '')
                        output_lines.append(f"        ┌─── RAW ARGUMENTS ───")
                        # Try to pretty-print JSON arguments
                        try:
                            args_dict = json.loads(args)
                            args_pretty = json.dumps(args_dict, indent=10)
                            for line in args_pretty.split('\n'):
                                output_lines.append(f"        │ {line}")
                        except:
                            # Not JSON, show as-is
                            for line in args.split('\n'):
                                output_lines.append(f"        │ {line}")
                        output_lines.append(f"        └────────────────────")
            
            elif log["type"] == "tool_execution_start":
                output_lines.append(f"\n🔧 TOOL EXECUTION: {log['tool_name']}")
                output_lines.append(f"   ┌─── RAW PARAMETERS ───")
                for key, value in log['parameters'].items():
                    value_str = str(value)
                    # Show full code for python tool
                    if log['tool_name'] == 'python_repl_tool' and key == 'code':
                        output_lines.append(f"   │ {key}:")
                        for line in value_str.split('\n'):
                            output_lines.append(f"   │   {line}")
                    else:
                        # Try to format as JSON if possible
                        try:
                            if isinstance(value, (dict, list)):
                                value_json = json.dumps(value, indent=8)
                                output_lines.append(f"   │ {key}:")
                                for line in value_json.split('\n'):
                                    output_lines.append(f"   │   {line}")
                            else:
                                if len(value_str) > 500:
                                    value_str = value_str[:500] + "..."
                                output_lines.append(f"   │ {key}: {value_str}")
                        except:
                            if len(value_str) > 500:
                                value_str = value_str[:500] + "..."
                            output_lines.append(f"   │ {key}: {value_str}")
                output_lines.append(f"   └─────────────────────")
            
            elif log["type"] == "tool_execution_result":
                output_lines.append(f"\n✅ TOOL RESULT: {log['tool_name']}")
                output_lines.append(f"   Result Length: {log['result_length']} chars")
                output_lines.append(f"   Reward: {log['reward']}")
                
                # Show metrics if available
                if log['metrics']:
                    output_lines.append(f"   ┌─── RAW METRICS (from tool) ───")
                    try:
                        metrics_json = json.dumps(log['metrics'], indent=8)
                        for line in metrics_json.split('\n'):
                            output_lines.append(f"   │ {line}")
                    except:
                        output_lines.append(f"   │ {log['metrics']}")
                    output_lines.append(f"   └───────────────────────────────")
                
                # Show the raw result from tool
                output_lines.append(f"   ┌─── RAW TOOL OUTPUT ───")
                result_str = str(log['result'])
                
                # Try to parse as JSON and pretty-print
                try:
                    # Try to parse the entire result as JSON
                    result_json = json.loads(result_str)
                    result_pretty = json.dumps(result_json, indent=8)
                    for line in result_pretty.split('\n'):
                        output_lines.append(f"   │ {line}")
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, show as plain text
                    for line in result_str.split('\n'):
                        output_lines.append(f"   │ {line}")
                
                output_lines.append(f"   └──────────────────────")
                
                # For python_repl_tool, try to extract and show structured information
                if log['tool_name'] == 'python_repl_tool':
                    output_lines.append(f"   ┌─── PARSED OUTPUT (if available) ───")
                    # Check if metrics has structured data
                    if log['metrics']:
                        if 'result' in log['metrics']:
                            output_lines.append(f"   │ Python Return Value: {log['metrics'].get('result', 'None')}")
                        if 'stdout' in log['metrics']:
                            stdout_val = log['metrics'].get('stdout', '')
                            if stdout_val:
                                output_lines.append(f"   │ Stdout Output:")
                                for line in str(stdout_val).split('\n'):
                                    output_lines.append(f"   │   {line}")
                            else:
                                output_lines.append(f"   │ Stdout Output: (empty)")
                        if 'stderr' in log['metrics']:
                            stderr_val = log['metrics'].get('stderr', '')
                            if stderr_val:
                                output_lines.append(f"   │ Stderr Output:")
                                for line in str(stderr_val).split('\n'):
                                    output_lines.append(f"   │   {line}")
                    output_lines.append(f"   └────────────────────────────────────")
            
            elif log["type"] == "tool_execution_error":
                output_lines.append(f"\n❌ TOOL ERROR: {log['tool_name']}")
                output_lines.append(f"   Error: {log['error']}")
        
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"EXECUTION COMPLETE - {len([l for l in self.logs if l['type'] == 'llm_call'])} LLM calls")
        output_lines.append(f"{'='*80}\n")
        
        # Print to console
        output_text = '\n'.join(output_lines)
        print(output_text)
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
    
    def get_sequence_summary(self) -> str:
        """Get a concise sequence summary"""
        if not self.logs:
            return "No execution logs"
        
        summary_lines = ["EXECUTION SEQUENCE SUMMARY:", ""]
        
        for i, log in enumerate(self.logs):
            if log["type"] == "llm_call":
                summary_lines.append(f"{i+1}. 🤖 LLM Call (Iteration {log['iteration']})")
            elif log["type"] == "llm_response":
                tool_info = f" with {len(log['tool_calls'])} tool calls" if log['tool_calls'] else ""
                summary_lines.append(f"{i+1}. 💭 LLM Response{tool_info}")
            elif log["type"] == "tool_execution_start":
                summary_lines.append(f"{i+1}. 🔧 Tool: {log['tool_name']}")
            elif log["type"] == "tool_execution_result":
                summary_lines.append(f"{i+1}. ✅ Result from {log['tool_name']}")
            elif log["type"] == "tool_execution_error":
                summary_lines.append(f"{i+1}. ❌ Error in {log['tool_name']}")
        
        return "\n".join(summary_lines)
    
    def save_to_json(self, filepath: str):
        """Save logs to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)


class OpenSourceAgent:
    """Agent class for handling LLM calls with tool support and visualization"""
    
    def __init__(self, client: AsyncOpenAI, model: str, tool_instances: dict, 
                 system_prompt: str, debug: bool = False, enable_reasoning: bool = True,
                 logger: Optional[AgentLogger] = None):
        self.client = client
        self.model = model
        self.tool_instances = tool_instances
        self.tool_schemas = [tool.get_openai_tool_schema() for tool in tool_instances.values()]
        self.system_prompt = system_prompt
        self.debug = debug
        self.enable_reasoning = enable_reasoning
        self.messages = [{"role": "system", "content": system_prompt}]
        self.logger = logger or AgentLogger(enabled=False)
        
    def add_message(self, content: dict):
        """Add a message to the conversation history"""
        self.messages.append(content)
        
    async def get_response(self, iteration: int = 0):
        """Get response from the model with tool support, reasoning control, and automatic retry"""
        import asyncio
        import json
        import random
        
        # Log the LLM call
        self.logger.log_llm_call(self.messages, iteration)
        
        # Build extra_body for DeepSeek reasoning control
        extra_body = {}
        if "deepseek" in self.model.lower() and self.enable_reasoning:
            extra_body["reasoning"] = {"enabled": True}
            if self.debug:
                print(f"[DEBUG] Enabled reasoning mode for DeepSeek")
        
        # Retry logic
        max_attempts = 3
        base_delay = 2.0
        
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=[tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None,
                    temperature=0.7,
                    max_tokens=32768,
                    extra_body=extra_body if extra_body else None
                )
                
                # Success! Log if this was a retry
                if attempt > 0:
                    print(f"✓ API call succeeded on attempt {attempt + 1}/{max_attempts}")
                
                # Log the LLM response
                self.logger.log_llm_response(response, iteration)
                
                return response
            
            except (json.JSONDecodeError, ConnectionError, TimeoutError) as e:
                error_type = type(e).__name__
                
                if attempt < max_attempts - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random() * 0.5)
                    
                    # Log the error
                    print(f"⚠ API call failed (attempt {attempt + 1}/{max_attempts}): {error_type}")
                    if isinstance(e, json.JSONDecodeError):
                        print(f"   JSON error at line {e.lineno}, column {e.colno}")
                    print(f"   Retrying in {delay:.1f} seconds...")
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
                else:
                    # Last attempt failed - log and raise
                    print(f"✗ API call failed after {max_attempts} attempts: {error_type}")
                    raise
            
            except Exception as e:
                # Catch-all for any other unexpected errors
                error_type = type(e).__name__
                
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random() * 0.5)
                    print(f"⚠ Unexpected error (attempt {attempt + 1}/{max_attempts}): {error_type}")
                    print(f"   {str(e)[:100]}")
                    print(f"   Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print(f"✗ Unexpected error after {max_attempts} attempts")
                    raise


class AQAAgentManager:
    """Manager for Academic Question Agent with tool support and visualization"""
    
    def __init__(self, base_url: str = None, api_key: str = None, 
                 model: str = "qwen/qwen3-235b-a22b-2507", 
                 debug: bool = False,
                 enable_reasoning: bool = True,
                 site_url: str = None,
                 site_name: str = None,
                 enable_visualization: bool = True):
        """
        Initialize the agent manager with OpenRouter/VLLM endpoint
        
        Args:
            base_url: API base URL (default: OpenRouter)
            api_key: API key
            model: Model identifier
            debug: Enable debug logging
            enable_reasoning: Enable reasoning mode for DeepSeek models
            site_url: Your site URL for OpenRouter rankings (optional)
            site_name: Your site name for OpenRouter rankings (optional)
            enable_visualization: Enable execution visualization (default: True)
        """
        
        # Use environment variables if not provided
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model
        self.debug = debug
        self.enable_reasoning = enable_reasoning
        self.enable_visualization = enable_visualization
        
        # OpenRouter optional headers for attribution
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name
        
        if self.debug and self.extra_headers:
            print(f"[DEBUG] Using OpenRouter headers: {self.extra_headers}")
        
        # Initialize client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=600,
            default_headers=self.extra_headers if self.extra_headers else None
        )
        
        # Initialize tools
        self.tool_instances = self._initialize_tools()
        
    def _initialize_tools(self) -> dict:
        """Initialize tool instances"""
        
        # Tool configurations
        TOOL_SCHEMAS = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Performs web searches to find information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "array",
                                "description": "Array of search queries",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "visit_tool",
                    "description": "Visit webpage(s) and return content summary",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "URLs to visit"
                            },
                            "goal": {
                                "type": "string",
                                "description": "Information goal"
                            }
                        },
                        "required": ["url", "goal"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "python_repl_tool",
                    "description": "Execute Python code for calculations or analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]
        
        TOOL_CLASSES = [
            WebSailorMiroflowMultiWebSearchTool,
            WebSailorMultiVisitTool,
            DeerFlowCodeTool
        ]
        
        TOOL_CONFIGS = [
            {},  # web search config
            {    # visit tool config
                "summarize_model": "gpt-4o-mini",
                "summarize_model_base_url": "https://api.openai.com/v1",
                "summarize_model_api_key": os.getenv("OPENAI_API_KEY", ""),
                "max_webpage_tokens": 28000,
                "max_context_length": 32768
            },
            {}   # code tool config
        ]
        
        tool_instances = {}
        for tool_cls, tool_cfg, tool_schema in zip(TOOL_CLASSES, TOOL_CONFIGS, TOOL_SCHEMAS):
            schema_obj = OpenAIFunctionToolSchema.model_validate(tool_schema)
            tool = tool_cls(config=tool_cfg, tool_schema=schema_obj)
            tool_instances[tool.name] = tool
        
        if self.debug:
            print(f"[DEBUG] Initialized {len(tool_instances)} tools: {list(tool_instances.keys())}")
        
        return tool_instances
    
    async def run_with_tools(self, prompt: str, system_prompt: str = None, 
                            max_iterations: int = 10, output_file: str = None) -> str:
        """
        Run the agent with tool support and visualization.
        
        Args:
            prompt: The user prompt/question
            system_prompt: System prompt for the agent
            max_iterations: Maximum tool calling iterations
            output_file: Optional file path to save visualization output
            
        Returns:
            The final response string
        """
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant that can search the web, visit pages, and run Python code."
        
        # Create logger for this execution
        logger = AgentLogger(enabled=self.enable_visualization)
        
        # Create agent instance
        agent = OpenSourceAgent(
            client=self.client,
            model=self.model,
            tool_instances=self.tool_instances,
            system_prompt=system_prompt,
            debug=self.debug,
            enable_reasoning=self.enable_reasoning,
            logger=logger
        )
        
        # Add user message
        agent.add_message({"role": "user", "content": prompt})
        
        # Initialize tool instances
        instance_id = f"agent_{id(agent)}_{asyncio.get_event_loop().time()}"
        for tool in self.tool_instances.values():
            await tool.create(instance_id=instance_id)
        
        try:
            # Tool calling loop
            for iteration in range(max_iterations):
                # Get response from model
                response = await agent.get_response(iteration=iteration)
                
                # Extract message
                message = response.choices[0].message

                # Add assistant message
                agent.add_message({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else None
                })
                
                # If no tool calls, we're done
                if not hasattr(message, 'tool_calls') or not message.tool_calls:
                    break
                
                # Execute tool calls
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.log_tool_error(fn_name, f"Invalid JSON arguments: {str(e)}", iteration)
                        agent.add_message({
                            "role": "tool",
                            "content": f"Error: Invalid JSON arguments: {str(e)}",
                            "tool_call_id": tool_call.id
                        })
                        continue
                    
                    if fn_name in self.tool_instances:
                        # Log tool execution start
                        logger.log_tool_execution(fn_name, fn_args, iteration)
                        
                        # Execute tool
                        result, reward, metrics = await self.tool_instances[fn_name].execute(
                            instance_id=instance_id,
                            parameters=fn_args
                        )
                        
                        # Log tool result
                        logger.log_tool_result(fn_name, result, reward, metrics, iteration)
                        
                        # Add tool result
                        agent.add_message({
                            "role": "tool",
                            "content": str(result) if result else "Tool executed but returned no result",
                            "tool_call_id": tool_call.id
                        })
                    else:
                        logger.log_tool_error(fn_name, f"Unknown tool '{fn_name}'", iteration)
                        agent.add_message({
                            "role": "tool",
                            "content": f"Error: Unknown tool '{fn_name}'",
                            "tool_call_id": tool_call.id
                        })
            
            # Print visualization
            if self.enable_visualization:
                logger.print_visualization(output_file=output_file)
            
            # Get final answer - return the last assistant message content
            final_messages = [m for m in agent.messages if m.get("role") == "assistant" and m.get("content")]
            if final_messages:
                final_response = final_messages[-1].get("content", "")
                if self.debug:
                    print(f"[DEBUG] Final response length: {len(final_response)}")
                return final_response

            if self.debug:
                print(f"[DEBUG] No final response found")
            return ""
            
        finally:
            # Clean up tool instances
            for tool in self.tool_instances.values():
                try:
                    await tool.release(instance_id=instance_id)
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Warning: Failed to release tool: {e}")
    
    async def run_with_tools_detailed(self, prompt: str, system_prompt: str = None, 
                                     max_iterations: int = 10, output_file: str = None) -> Tuple[str, List[Dict], AgentLogger]:
        """
        Run the agent with tool support and return response, conversation history, and logger.
        
        Args:
            prompt: The user prompt/question
            system_prompt: System prompt for the agent
            max_iterations: Maximum tool calling iterations
            output_file: Optional file path to save visualization output
            
        Returns:
            Tuple of (final_response_string, conversation_list, logger)
        """
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant that can search the web, visit pages, and run Python code. If you run code, please always print out the result. Before you choose to stop, you must always make sure the final, complete answer is included in your response."
        
        # Create logger for this execution
        logger = AgentLogger(enabled=self.enable_visualization)
        
        # Create agent instance
        agent = OpenSourceAgent(
            client=self.client,
            model=self.model,
            tool_instances=self.tool_instances,
            system_prompt=system_prompt,
            debug=self.debug,
            enable_reasoning=self.enable_reasoning,
            logger=logger
        )
        
        # Add user message
        agent.add_message({"role": "user", "content": prompt})
        
        # Initialize tool instances
        instance_id = f"agent_{id(agent)}_{asyncio.get_event_loop().time()}"
        for tool in self.tool_instances.values():
            await tool.create(instance_id=instance_id)
        
        try:
            # Tool calling loop
            for iteration in range(max_iterations):
                # Get response from model
                response = await agent.get_response(iteration=iteration)
                
                # Extract message
                message = response.choices[0].message

                # Add assistant message
                agent.add_message({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') and message.tool_calls else None
                })
                
                # If no tool calls, we're done
                if not hasattr(message, 'tool_calls') or not message.tool_calls:
                    break
                
                # Execute tool calls
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        logger.log_tool_error(fn_name, f"Invalid JSON arguments: {str(e)}", iteration)
                        agent.add_message({
                            "role": "tool",
                            "content": f"Error: Invalid JSON arguments: {str(e)}",
                            "tool_call_id": tool_call.id
                        })
                        continue
                    
                    if fn_name in self.tool_instances:
                        # Log tool execution start
                        logger.log_tool_execution(fn_name, fn_args, iteration)
                        
                        # Execute tool
                        result, reward, metrics = await self.tool_instances[fn_name].execute(
                            instance_id=instance_id,
                            parameters=fn_args
                        )
                        
                        # Log tool result
                        logger.log_tool_result(fn_name, result, reward, metrics, iteration)
                        
                        # Add tool result
                        agent.add_message({
                            "role": "tool",
                            "content": str(result) if result else "Tool executed but returned no result",
                            "tool_call_id": tool_call.id
                        })
                    else:
                        logger.log_tool_error(fn_name, f"Unknown tool '{fn_name}'", iteration)
                        agent.add_message({
                            "role": "tool",
                            "content": f"Error: Unknown tool '{fn_name}'",
                            "tool_call_id": tool_call.id
                        })
            
            # Get final answer
            final_messages = [m for m in agent.messages if m.get("role") == "assistant" and m.get("content")]
            final_response = final_messages[-1].get("content", "") if final_messages else ""

            return final_response, agent.messages, logger
            
        finally:
            # Clean up tool instances
            for tool in self.tool_instances.values():
                try:
                    await tool.release(instance_id=instance_id)
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Warning: Failed to release tool: {e}")
    
    async def run_without_tools(self, prompt: str, system_prompt: str = None) -> str:
        """
        Run the agent without tools (simple completion).
        
        Args:
            prompt: The user prompt
            system_prompt: System prompt
            
        Returns:
            The response string
        """
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Build extra_body for DeepSeek reasoning control
        extra_body = {}
        if "deepseek" in self.model.lower() and self.enable_reasoning:
            extra_body["reasoning"] = {"enabled": True}
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=32768,
            extra_body=extra_body if extra_body else None
        )
        
        return response.choices[0].message.content


# Convenience functions for backward compatibility
async def call_llm_with_tools(prompt: str, system_prompt: str = None, 
                              base_url: str = None, api_key: str = None,
                              model: str = "qwen/qwen3-235b-a22b-2507",
                              enable_visualization: bool = True) -> str:
    """
    Wrapper function to replace original LLM calls with tool support.
    """
    manager = AQAAgentManager(base_url=base_url, api_key=api_key, model=model,
                             enable_visualization=enable_visualization)
    return await manager.run_with_tools(prompt, system_prompt)


async def call_llm_simple(prompt: str, system_prompt: str = None,
                          base_url: str = None, api_key: str = None,
                          model: str = "qwen/qwen3-235b-a22b-2507") -> str:
    """
    Wrapper function for simple LLM calls without tools.
    """
    manager = AQAAgentManager(base_url=base_url, api_key=api_key, model=model,
                             enable_visualization=False)
    return await manager.run_without_tools(prompt, system_prompt)