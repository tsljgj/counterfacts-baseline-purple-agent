# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import json
import yaml
import time
from typing import Any, Optional, Tuple, Dict, List, Union
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .deep_research_utils import MinimalBrowser, LiteLLMModel, populate_template


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class WebSearchTool(BaseTool):
    """A tool for performing web search queries using SerpAPI directly."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform a web search query using Google and return the search results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The web search query to perform.",
                        },
                    },
                    "required": ["query"],
                },
            }
        }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        if self.serpapi_key is None:
            raise ValueError("Environment variable SERPAPI_API_KEY is required for SearchInformationTool")


    async def _search_google(self, query: str) -> str:
        """Perform search using SerpAPI and return formatted results."""
        try:
            # Import here to avoid dependency issues if not installed
            from serpapi import GoogleSearch
        except ImportError:
            raise ImportError("serpapi package is required. Install with: pip install google-search-results")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" not in results.keys():
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        
        if len(results["organic_results"]) == 0:
            return f"No results found for '{query}'. Try with a more general query."

        # Format results
        web_snippets: List[str] = list()
        idx = 0
        
        for page in results["organic_results"]:
            idx += 1
            date_published = ""
            if "date" in page:
                date_published = "\nDate published: " + page["date"]

            source = ""
            if "source" in page:
                source = "\nSource: " + page["source"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            result_entry = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}{snippet}"
            result_entry = result_entry.replace("Your browser can't play this video.", "")
            web_snippets.append(result_entry)

        # Create header and content
        header = f"Address: google: {query}\nTitle: {query} - google search\nViewport position: Showing page 1 of 1.\n"
        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )
        
        return header + "=======================\n" + content

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "search_history": [],
            "last_result": "",
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        query = parameters.get("query", "")
        
        if not isinstance(query, str):
            query = str(query)

        try:
            # Perform the web search
            result = await self._search_google(query)
            
            # Store the result
            self._instance_dict[instance_id]["search_history"].append(query)
            self._instance_dict[instance_id]["last_result"] = result
            
            return result.strip(), 0.0, {
                "query": query, 
                "result_length": len(result)
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


class QueryPageTool(BaseTool):
    """A tool for querying a web page with a specific question."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "query_page",
                "description": "Query a web page given its URL and a query string to ask a specific question to the page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL of the web page to query."},
                        "query": {"type": "string", "description": "The query for the web page."}
                    },
                    "required": ["url", "query"],
                },
            }
        }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        if self.firecrawl_api_key is None:
            raise ValueError("Environment variable FIRECRAWL_API_KEY is required for QueryPageTool")
        
        self.llm = LiteLLMModel(model=config.get('query_page_llm_model'),
                                api_base=config.get('query_page_llm_api_base'),
                                api_key=config.get('query_page_llm_api_key', os.getenv('OPENAI_API_KEY')))
        
        prompt_template_file = config.get('query_page_prompt_template_file')
        with open(prompt_template_file, 'r') as f:
            prompt_template_dict = yaml.safe_load(f)
            self.query_prompt = prompt_template_dict['tool_prompts']["query_observation"]
            self.summarize_query_prompt = prompt_template_dict['tool_prompts']["summarize_query_responses"]
        
    async def create(self, instance_id: Optional[str] = None, task: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Create a unique browser instance for this instance_id
        self._instance_dict[instance_id] = {
            "browser": MinimalBrowser(firecrawl_api_key=self.firecrawl_api_key),
            "task": task,
            "query_history": [],
            "last_result": "",
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        url = parameters.get("url", "")
        query = parameters.get("query", "")
        
        if not isinstance(url, str):
            url = str(url)
        if not isinstance(query, str):
            query = str(query)

        if instance_id not in self._instance_dict:
            raise ValueError(f"Instance {instance_id} not found. Call create() first.")

        try:
            # Get the browser instance for this specific instance_id
            browser = self._instance_dict[instance_id]["browser"]
            task = self._instance_dict[instance_id]["task"]
            
            # Visit the page using this instance's browser
            browser.visit_page(url)
            all_states = browser.get_all_states()
            obs_lst = []
            for header, content in all_states:
                obs_lst.append(header.strip() + "\n=======================\n" + content)
            # print(f"Obs_lst: {obs_lst[0]}")
            # print("\n=======================\n")
            
            # Query each observation
            batched_inputs = []
            for observation in obs_lst:
                batched_inputs.append([{
                    "role": 'user',
                    "content": [{
                        "type": "text",
                        "text": populate_template(
                            self.query_prompt, variables={"query": query, "task": task, "observation": observation}
                        ),
                    }],
                }])
                # print(self._populate_template(self.query_prompt, variables={"query": query, "task": task, "observation": observation}))
                # print("\n=======================\n")
            if False and self.llm.model == "hosted_vllm/Qwen/Qwen3-32B":
                responses = self.llm(batched_inputs, chat_template_kwargs={"enable_thinking": False})
            else:
                responses = self.llm(batched_inputs)
            # responses = 'This is a simulated response. Please proceed based on your best knowledge.'
            
            # Summarize responses
            summary_prompt = populate_template(
                self.summarize_query_prompt,
                variables={"query": query, "task": task, "responses": responses}
            )
            summary_input_messages = [
                {
                    "role": 'user',
                    "content": [{
                        "type": "text",
                        "text": summary_prompt,
                    }],
                }
            ]
            # print(summary_prompt)
            # print("\n=======================\n")
            if False and self.llm.model == "hosted_vllm/Qwen/Qwen3-32B":
                summary = self.llm(summary_input_messages, chat_template_kwargs={"enable_thinking": False})
            else:
                summary = self.llm(summary_input_messages)
            # summary = 'This is a simulated response. Please proceed based on your best knowledge.'
            
            self._instance_dict[instance_id]["query_history"].append((url, query))
            self._instance_dict[instance_id]["last_result"] = summary
            
            return summary, 0.0, {"url": url, "query": query, "observations_count": len(obs_lst)}
            
        except Exception as e:
            error_msg = f"Error querying page '{url}' with query '{query}': {str(e)}"
            logger.error(error_msg)
            return error_msg, -0.1, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            # Clean up the browser instance
            self._instance_dict[instance_id]["browser"].reset()
            del self._instance_dict[instance_id]["browser"]
            del self._instance_dict[instance_id]

