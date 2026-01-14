import requests
from itertools import cycle
import threading
import os


from typing import Any, Optional, Tuple
import json

from .base_tool import BaseTool
from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS, RunCodeResponse, RunStatus

# Default sandbox servers - can be overridden via environment variable or function parameter
DEFAULT_SANDBOX_SERVERS = [
    # "fs-mbz-gpu-044", # Add more servers here
]

# Thread-safe cycle iterator for round-robin load balancing
server_cycle = None
cycle_lock = threading.Lock()


def _parse_sandbox_servers(servers_input):
    """Parse sandbox servers from various input formats"""
    if not servers_input:
        return DEFAULT_SANDBOX_SERVERS
    
    if isinstance(servers_input, str):
        # Single server or comma-separated servers
        if ',' in servers_input:
            return [server.strip() for server in servers_input.split(',')]
        else:
            return [servers_input.strip()]
    elif isinstance(servers_input, list):
        return servers_input
    else:
        raise ValueError(f"Invalid sandbox servers format: {type(servers_input)}. Expected str or list.")


def _get_next_server(server_cycle):
    """Get the next server in round-robin fashion thread-safely."""
    with cycle_lock:
        return next(server_cycle)


def code_exec_sandboxfusion(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, sandbox_servers=None):
    """
    Execute Python code using SandboxFusion remote service.
    
    Args:
        code: Python code to execute
        stdin: Optional input to pass to the code
        timeout: Timeout in seconds (default from utils)
        sandbox_servers: Optional server names for sandbox servers. Can be:
                        - Single server string: "fs-mbz-gpu-044"
                        - Comma-separated servers: "fs-mbz-gpu-044,fs-mbz-gpu-045"
                        - List of servers: ["fs-mbz-gpu-044", "fs-mbz-gpu-045"]
                        - None: Uses SANDBOX_FUSION_SERVERS environment variable or default
        
    Returns:
        tuple: (success: bool, output: str)
    """
    try:
        # Determine sandbox servers to use
        if sandbox_servers is None:
            sandbox_servers = os.getenv('SANDBOX_FUSION_SERVERS', '')
        
        servers = _parse_sandbox_servers(sandbox_servers)
        server_cycle = cycle(servers)
        
        if not servers:
            return False, _ERROR_MSG_PREFIX + "No sandbox servers configured. Set SANDBOX_FUSION_SERVERS environment variable or pass sandbox_servers parameter."
        
        request_data = {
            "language": "python",
            "code": code,
            "stdin": stdin,
            "run_timeout": timeout
        }
        
        # Try each server (for load balancing/failover)
        for _ in range(len(servers)):
            try:
                server = _get_next_server(server_cycle)
                url = f"http://{server}:8080/run_code"
                response = requests.post(url, json=request_data, timeout=timeout + 2)
                
                if response.status_code != 200:
                    continue  # Try next server
                
                result = RunCodeResponse(**response.json())
                if result.status == RunStatus.Success:
                    return True, result.run_result.stdout
                else:
                    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{result.run_result.stdout}\n\nSTDERR:\n{result.run_result.stderr}"
                    
            except requests.exceptions.RequestException:
                continue  # Try next server
        
        # If we get here, all servers failed
        return False, _ERROR_MSG_PREFIX + f"All sandbox servers failed to process the request. Servers tried: {servers}"
            
    except Exception as e:
        return False, _ERROR_MSG_PREFIX + f"Execution error: {str(e)}"


def code_exec_sandboxfusion_with_pytest(code, pytest_code, timeout=_DEFAULT_TIMEOUT_SECONDS, sandbox_servers=None):
    """
    Execute Python code with pytest using SandboxFusion remote service.
    
    Args:
        code: Python solution code
        pytest_code: Pytest test code
        timeout: Timeout in seconds
        sandbox_servers: Optional server names for sandbox servers (same format as code_exec_sandboxfusion)
        
    Returns:
        tuple: (success: bool, output: str)
    """
    # Combine the solution code and test code
    combined_code = f"""
{code}

{pytest_code}
"""
    return code_exec_sandboxfusion(combined_code, timeout=timeout, sandbox_servers=sandbox_servers)


class CodeExecutionTool(BaseTool):
    """Tool for executing Python code in a sandboxed environment."""
    
    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.sandbox_servers = config.get('sandbox_servers', None)
        self.default_timeout = config.get('timeout', _DEFAULT_TIMEOUT_SECONDS)
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        return await super().create(instance_id, **kwargs)
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute Python code in the sandbox.
        
        Args:
            instance_id: The instance id of the tool.
            parameters: Dictionary containing:
                - code: Python code to execute (required)
                - stdin: Optional input to pass to the code
                - timeout: Optional timeout override
                
        Returns:
            tuple: (tool_response, tool_reward_score, tool_metrics)
        """
        try:
            code = parameters.get('code')
            if not code:
                return "Error: No code provided to execute.", 0.0, {"error": "missing_code"}
            
            stdin = parameters.get('stdin', None)
            timeout = parameters.get('timeout', self.default_timeout)
            
            success, output = code_exec_sandboxfusion(
                code=code,
                stdin=stdin,
                timeout=timeout,
                sandbox_servers=self.sandbox_servers
            )
            
            if success:
                response = f"Output:\n```\n{output}\n```"
                reward = 1.0
                metrics = {"success": True, "output_length": len(output)}
            else:
                response = f"Execution failed:\n\n{output}"
                reward = 0.0
                metrics = {"success": False, "error": True}
            
            return response, reward, metrics
            
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            return error_msg, 0.0, {"error": "tool_exception", "exception": str(e)}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        pass  # No cleanup needed for stateless code execution


class CodeExecutionWithPytestTool(BaseTool):
    """Tool for executing Python code with pytest in a sandboxed environment."""
    
    def __init__(self, config: dict, tool_schema):
        super().__init__(config, tool_schema)
        self.sandbox_servers = config.get('sandbox_servers', None)
        self.default_timeout = config.get('timeout', _DEFAULT_TIMEOUT_SECONDS)
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        return await super().create(instance_id, **kwargs)
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute Python code with pytest in the sandbox.
        
        Args:
            instance_id: The instance id of the tool.
            parameters: Dictionary containing:
                - code: Python solution code (required)
                - pytest_code: Pytest test code (required)
                - timeout: Optional timeout override
                
        Returns:
            tuple: (tool_response, tool_reward_score, tool_metrics)
        """
        try:
            code = parameters.get('code')
            pytest_code = parameters.get('pytest_code')
            
            if not code:
                return "Error: No solution code provided.", 0.0, {"error": "missing_code"}
            if not pytest_code:
                return "Error: No pytest code provided.", 0.0, {"error": "missing_pytest_code"}
            
            timeout = parameters.get('timeout', self.default_timeout)
            
            success, output = code_exec_sandboxfusion_with_pytest(
                code=code,
                pytest_code=pytest_code,
                timeout=timeout,
                sandbox_servers=self.sandbox_servers
            )
            
            if success:
                response = f"Code and tests executed successfully:\n\n```\n{output}\n```"
                reward = 1.0
                metrics = {"success": True, "output_length": len(output)}
            else:
                response = f"Code/test execution failed:\n\n{output}"
                reward = 0.0
                metrics = {"success": False, "error": True}
            
            return response, reward, metrics
            
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            return error_msg, 0.0, {"error": "tool_exception", "exception": str(e)}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        pass  # No cleanup needed for stateless code execution 