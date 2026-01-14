import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4
import asyncio
import json
from openai import AsyncOpenAI
from langchain_experimental.utilities import PythonREPL
import tiktoken

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .tavily_search.tavily_search_results_with_images import (
    TavilySearchResultsWithImages,
)
from .crawler import Crawler

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

PythonREPLTool_init = PythonREPL.__init__

def PythonREPLTool_init_wrapper(self, *args, **kwargs):
    PythonREPLTool_init(self, *args, **kwargs)
    self.globals = self.locals

PythonREPL.__init__ = PythonREPLTool_init_wrapper


class DeerFlowSearchTool(BaseTool):
    """A tool for performing web search queries using SerpAPI directly."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.",
                "parameters": {
                    "properties": {
                        "query": {
                            "description": "search query to look up",
                            "type": "string"
                        }
                    },
                    "required": [
                        "query"
                    ],
                    "type": "object"
                }
            }
        },
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.max_search_results = config.get("max_search_results", 3)
        self.include_raw_content = config.get("include_raw_content", True)
        self.include_images = config.get("include_images", True)
        self.include_image_descriptions = config.get("include_image_descriptions", True)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "search_tool": TavilySearchResultsWithImages(
                name="web_search",
                max_results=self.max_search_results,
                include_raw_content=self.include_raw_content,
                include_images=self.include_images,
                include_image_descriptions=self.include_image_descriptions,
            )
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        query = parameters.get("query", "")
        
        if not isinstance(query, str):
            query = str(query)

        try:
            # Perform the web search
            search_tool = self._instance_dict[instance_id]["search_tool"]
            result = await search_tool.ainvoke({"args": {"query": query}, "type": "tool_call", "id": "foo", "name": "tavily"})
            
            return result.content, 0.0, {
                "result_length": len(result.content),
                "artifact": result.artifact
            }
        except Exception as e:
            error_msg = f"Error performing search for '{query}': {str(e)}"
            logger.error(error_msg)
            return error_msg, -0.1, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful search
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            
            
class DeerFlowCrawlTool(BaseTool):
    """A tool for performing web search queries using SerpAPI directly."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "crawl_tool",
                "description": "Use this to crawl a url and get a readable content in markdown format.",
                "parameters": {
                    "properties": {
                        "url": {
                            "description": "The url to crawl.",
                            "type": "string"
                        }
                    },
                    "required": [
                        "url"
                    ],
                    "type": "object"
                }
            }
        },
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "crawler": Crawler()
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        url = parameters.get("url", "")
        
        if not isinstance(url, str):
            url = str(url)

        try:
            # Perform the web search
            crawler = self._instance_dict[instance_id]["crawler"]
            article = crawler.crawl(url)
            return json.dumps({"url": url, "crawled_content": article.to_markdown()[:10000]}), 0.0, {
                "url": url, 
                "result_length": len(article.to_markdown())
            }
        except Exception as e:
            error_msg = f"Error crawling url '{url}': {str(e)}"
            logger.error(error_msg)
            return error_msg, -0.1, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful search
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            
class DeerFlowVisitTool(DeerFlowCrawlTool):
    """A tool for performing web search queries using SerpAPI directly."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.summarize_model = config.get("summarize_model", "gpt-4.1-mini")
        self.client = AsyncOpenAI(base_url=config.get("summarize_model_base_url", "https://api.openai.com/v1"), 
                                  api_key=config.get("summarize_model_api_key", os.getenv("OPENAI_API_KEY")),
                                  timeout=config.get("summarize_model_timeout", 300))
        self.max_tokens = config.get("max_webpage_tokens", 28000)
        self.max_context_length = config.get("max_context_length", 32768)
        
        self.enc = tiktoken.encoding_for_model("gpt-4o")
        
        self.websailor_format = config.get("websailor_format", False)
        self.summarize_with_long_context = config.get("summarize_with_long_context", False)
        self.summarize_no_evidence = config.get("summarize_no_evidence", False)
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        url = parameters.get("url", "")
        visit_goal = parameters.get("visit_goal", "")
        
        if not isinstance(url, str):
            url = str(url)

        if not isinstance(visit_goal, str):
            visit_goal = str(visit_goal)
        
        crawl_num_attempts = 5
        for _ in range(crawl_num_attempts):
            try:
                # Perform the web search
                crawler = self._instance_dict[instance_id]["crawler"]
                article = await crawler.crawl(url)
                article_content = article.to_markdown()
                
                # Summarize the article content
                summary = await self.summarize_article(article_content, visit_goal)
                # with open("debug-1.txt", 'w') as f:
                #     f.write(summary)
                
                if self.websailor_format:
                    format_num_attempts = 5
                    for _ in range(format_num_attempts):
                        try:
                            useful_information = self.websailor_format_summary(url, visit_goal, summary)
                            return useful_information, 0.0, {
                                "url": url, 
                                "full_webpage": article_content,
                            }
                        except Exception as e:
                            # with open("debug.txt", 'w') as f:
                            #     f.write(summary)
                            #     f.write(str(e))
                            logger.error(f"Error formatting summary: {str(e)}")
                            summary = await self.summarize_article(article_content, visit_goal)
                            continue
                    error_msg = f"Error formatting summary after {format_num_attempts} attempts"
                    logger.error(error_msg)
                    return error_msg, -0.1, {"error": error_msg}
                else:
                    return json.dumps({"url": url, "webpage_summary": summary}), 0.0, {
                        "url": url, 
                        "full_webpage": article_content,
                    }
            except Exception as e:
                logger.error(f"Error crawling url '{url}': {str(e)}")
                continue
        error_msg = f"Error crawling url '{url}' after {crawl_num_attempts} attempts"
        logger.error(error_msg)
        return error_msg, -0.1, {"error": error_msg}
    
    def websailor_format_summary(self, url: str, visit_goal: str, summary: str) -> str:
        if summary.startswith('```json'):
            summary = summary[7:]
        if summary.endswith('```'):
            summary = summary[:-3]

        raw = json.loads(summary, strict=False)
        useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=visit_goal)
        # useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
        useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"
        return useful_information

    async def summarize_article(self, article_content: str, visit_goal: str) -> str:

        # Check if the article content is too long
        if len(self.enc.encode(article_content)) > self.max_tokens:
            # Truncate the article content
            article_content = self.enc.decode(self.enc.encode(article_content)[:self.max_tokens])
            
#         prompt = f"""Please summarize the following webpage content. The user's goal is: **{visit_goal}**. \
# Present the summary in markdown format, using bullet points or short paragraphs. Be clear and concise.

# Webpage content:

# {article_content}"""

        if self.summarize_no_evidence:
            prompt = f"""Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{article_content}

## **User Goal**
{visit_goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "summary" feilds**. You must escape every backslash (e.g. in LaTeX code) with double backslashes.
"""
        else:
            prompt = f"""Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{article_content}

## **User Goal**
{visit_goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**. You must escape every backslash (e.g. in LaTeX code) with double backslashes.
"""
        # system_prompt = "You are an expert in summarizing web pages"
        # prompt_length = len(self.enc.encode(prompt)) + 2 + len(self.enc.encode(system_prompt))
        prompt_length = len(self.enc.encode(prompt)) + 1

        if self.summarize_with_long_context:
            max_ctx_len = 4096
        else:
            max_ctx_len = 2048
        
        if self.summarize_model.startswith('Qwen/Qwen3'):
            response = await self.client.chat.completions.create(
                model=self.summarize_model,
                messages=[{"role": "user",  
                           "content": prompt}],
                max_completion_tokens=min(self.max_context_length - prompt_length, max_ctx_len),
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20, 
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            # print(f'Qwen/Qwen3 Response: {response.choices[0]}\n\n')
            # print('-' * 100)
            # return response.choices[0].message.reasoning_content
            return response.choices[0].message.content
            
        else:
            response = await self.client.chat.completions.create(
                model=self.summarize_model,
                messages=[{"role": "user",  
                        "content": prompt}],
                max_completion_tokens=min(self.max_context_length - prompt_length, max_ctx_len),
            )
            return response.choices[0].message.content
    
    
import asyncio, os, signal, sys, time, psutil
from concurrent.futures import ProcessPoolExecutor
def kill_proc_tree(pid: int):
    try:
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            c.kill()
        p.kill()
    except psutil.NoSuchProcess:
        pass

async def run_with_hard_timeout(python_repl: PythonREPL, code: str, timeout_s: int):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = loop.run_in_executor(ex, python_repl.run, code)
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            # Kill the worker process & its children
            for proc in ex._processes.values():
                kill_proc_tree(proc.pid)
            raise

class DeerFlowCodeTool(BaseTool):
    """A tool for performing code execution."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "python_repl_tool",
                "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value,\n    you should print it out with `print(...)`. This is visible to the user.",
                "parameters": {
                    "properties": {
                        "code": {
                            "description": "The python code to execute to do further analysis or calculation.",
                            "type": "string"
                        }
                    },
                    "required": [
                        "code"
                    ],
                    "type": "object"
                }
            }
        }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "python_repl": PythonREPL()
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        code = parameters.get("code", "")
        
        if not isinstance(code, str):
            error_msg = f"Invalid input: code must be a string, got {type(code)}"
            logger.error(error_msg)
            result_str = f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"
            return result_str, -0.1, {"code": code, "error": error_msg}

        logger.info("Executing Python code")
        try:
            python_repl = self._instance_dict[instance_id]["python_repl"]
            # result = python_repl.run(code, timeout=60)
            result = await run_with_hard_timeout(python_repl, code, 60)
            # Check if the result is an error message by looking for typical error patterns
            if isinstance(result, str) and ("Error" in result or "Exception" in result):
                logger.error(result)
                result_str = f"Error executing code:\n```python\n{code}\n```\nError: {result}"
                return result_str, -0.1, {"code": code, "error": result}
            logger.info("Code execution successful")
        except BaseException as e:
            error_msg = repr(e)
            logger.error(error_msg)
            result_str = f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"
            return result_str, -0.1, {"code": code, "error": error_msg}

        result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        
        return result_str, 0.0, {
            "code": code, 
            "result": result
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful code execution
        return 0.0  
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            
# if __name__ == "__main__":
#     tool = DeerFlowSearchTool(config={}, tool_schema={})
#     instance_id = tool.create()
#     print(asyncio.run(tool.execute(instance_id, {"query": "What is the capital of France?"})))
#     tool.release(instance_id)

#     tool = DeerFlowCrawlTool(config={}, tool_schema={})
#     instance_id = tool.create()
#     print(asyncio.run(tool.execute(instance_id, {"url": "https://www.google.com"})))
#     tool.release(instance_id)
    
#     tool = DeerFlowCodeTool(config={}, tool_schema={})
#     instance_id = tool.create()
#     print(asyncio.run(tool.execute(instance_id, {"code": "print('Hello, world!')"})))
#     tool.release(instance_id)