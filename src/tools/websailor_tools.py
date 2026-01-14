import logging
import os
import json
from typing import Any, Optional, Tuple
from uuid import uuid4
import asyncio
import aiohttp
from openai import AsyncOpenAI
import transformers
import tiktoken

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class WebSailorMiroflowWebSearchTool(BaseTool):
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
                        "q": {
                            "type": "string",
                            "description": "Search query string",
                        },
                        "gl": {
                            "type": "string",
                            "description": "Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')",
                        },
                        "hl": {
                            "type": "string",
                            "description": "Optional language code for search results in ISO 639-1 format (e.g., 'en')",
                        },
                        "location": {
                            "type": "string",
                            "description": "Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')",
                        },
                        "num": {
                            "type": "number",
                            "description": "Number of results to return (default: 10)",
                        },
                        "tbs": {
                            "type": "string",
                            "description": "Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)",
                        },
                    },
                    "required": ["q", "gl", "hl"],
                },
            }
        }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.serpapi_key = config.get("serpapi_key", os.getenv("SERPAPI_API_KEY"))
        if self.serpapi_key is None:
            raise ValueError("Environment variable SERPAPI_API_KEY is required for SearchInformationTool")
        self.default_num = config.get("default_num", 10)
        self.default_gl = config.get("default_gl", "us")
        self.default_hl = config.get("default_hl", "en")
        self.default_location = config.get("default_location", None)
        self.default_tbs = config.get("default_tbs", None)
        


    async def _search_google(self, 
                             q: str,
                             num: int = 10,
                             gl: str = 'us',
                             hl: str = 'en',
                             location: Optional[str] = None,
                             tbs: Optional[str] = None) -> str:
        """Perform search using SerpAPI and return formatted results."""
        try:
            # Import here to avoid dependency issues if not installed
            # from serpapi import GoogleSearch
            from serpapi import SerpApiClient
        except ImportError:
            raise ImportError("serpapi package is required. Install with: pip install google-search-results")

        params = {
            "engine": "google",
            "q": q,
            "api_key": self.serpapi_key,
            "num": num,
            "page": 1,
            "gl": gl,
            "hl": hl,
        }
        if location is not None:
            params["location"] = location
        if tbs is not None:
            params["tbs"] = tbs

        # search = GoogleSearch(params)
        search = SerpApiClient(params, engine='google', timeout=60)
        results = search.get_dict()
        
        try:
            if "organic_results" not in results:
                raise Exception(f"No results found for query: '{q}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic_results" in results:
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

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"

                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{q}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        except:
            content = f"No results found for '{q}'. Try with a more general query, or remove the year filter."
        
        return content
        

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "search_history": [],
            "last_result": "",
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        q = parameters.get("q", "")
        gl = parameters.get("gl", self.default_gl)
        hl = parameters.get("hl", self.default_hl)
        num = parameters.get("num", self.default_num)
        location = parameters.get("location", self.default_location)
        tbs = parameters.get("tbs", self.default_tbs)
        
        if not isinstance(q, str):
            q = str(q)

        try:
            # Perform the web search
            result = await self._search_google(q, num, gl, hl, location, tbs)
            
            # Store the result
            self._instance_dict[instance_id]["search_history"].append(q)
            self._instance_dict[instance_id]["last_result"] = result
            
            return result.strip(), 0.0, {
                "query": q, 
                "result_length": len(result)
            }
        except Exception as e:
            error_msg = f"Error performing search for '{q}': {str(e)}"
            logger.error(error_msg)
            return error_msg, -0.1, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful search
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class WebSailorMiroflowMultiWebSearchTool(BaseTool):
    """A tool for performing potentially multiple web search queries using SerpAPI directly."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Performs batched web searches: supply an array 'query' with each item containing a search query and other parameters; the tool retrieves search results for each query in one call.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "description": "Array of query objects. Include multiple complementary search queries in a single call.",
                            "items": {
                                "type": "string",
                            },
                            "maxItems": 3,
                        },
                        "gl": {
                            "type": "string",
                            "description": "Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')",
                        },
                        "hl": {
                            "type": "string",
                            "description": "Optional language code for search results in ISO 639-1 format (e.g., 'en')",
                        },
                        "location": {
                            "type": "string",
                            "description": "Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')",
                        },
                        "num": {
                            "type": "number",
                            "description": "Number of results to return (default: 10)",
                        },
                        "tbs": {
                            "type": "string",
                            "description": "Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year)",
                        },
                        
                    },
                    "required": ["query", "gl", "hl"],
                },
            }
        }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.serpapi_key = config.get("serpapi_key", os.getenv("SERPAPI_API_KEY"))
        if self.serpapi_key is None:
            raise ValueError("Environment variable SERPAPI_API_KEY is required for SearchInformationTool")
        self.default_num = config.get("default_num", 10)
        self.default_gl = config.get("default_gl", "us")
        self.default_hl = config.get("default_hl", "en")
        self.default_location = config.get("default_location", None)
        self.default_tbs = config.get("default_tbs", None)

        self.max_queries = config.get("max_queries", 3) # Add a hard cap on how many queries one batch should be
        
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}   
        return instance_id
    
    # async def google_search(self, query: dict):
    async def google_search(self, q, gl, hl, num, location, tbs):
        """Perform search using SerpAPI and return formatted results."""
        try:
            # Import here to avoid dependency issues if not installed
            # from serpapi import GoogleSearch
            from serpapi import SerpApiClient
        except ImportError:
            raise ImportError("serpapi package is required. Install with: pip install google-search-results")
        
        params = {
            "engine": "google",
            "q": q,
            "api_key": self.serpapi_key,
            "num": num,
            "page": 1,
            "gl": gl,
            "hl": hl,
        }
        if location is not None:
            params["location"] = location
        if tbs is not None:
            params["tbs"] = tbs

        # search = GoogleSearch(params)
        search = SerpApiClient(params, engine='google', timeout=60)
        results = search.get_dict()
        
        try:
            if "organic_results" not in results:
                raise Exception(f"No results found for query: '{q}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic_results" in results:
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

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"

                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{q}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        except:
            content = f"No results found for '{q}'. Try with a more general query, or remove the year filter."
        
        return content
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        
        try:
            query = parameters["query"]
        except:
            return ("[Search] Invalid request format: Input must be a JSON object containing 'query' field", 
                    0.0, 
                    {"error": "Invalid request format: Input must be a JSON object containing 'query' field"})
            
        gl = parameters.get("gl", self.default_gl)
        hl = parameters.get("hl", self.default_hl)
        num = parameters.get("num", self.default_num)
        location = parameters.get("location", self.default_location)
        tbs = parameters.get("tbs", self.default_tbs)
            
        if isinstance(query, str):
            response = await self.google_search(query, gl, hl, num, location, tbs)
        else:
            assert isinstance(query, list)
            if len(query) > self.max_queries: # Truncate queries when it's too many queries per batch
                query = query[:self.max_queries]
                truncated_note = f"\n\n[Note] Truncated to {self.max_queries} queries to limit cost."
            else:
                print(f"length of query batch: {len(query)}")
                truncated_note = ""
            sem = asyncio.Semaphore(3)

            async def _run(q: Any) -> str:
                async with sem:
                    res = await self.google_search(q, gl, hl, num, location, tbs)   # now async
                    return str(res)

            # gather preserves input order
            results = await asyncio.gather(*(_run(q) for q in query))
            
            response = "\n=======\n".join(results) + truncated_note # add truncated note if queries are truncated
        
        return response.strip(), 0.0, {
            "query": query, 
            "result_length": len(response)
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful search
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
            
            

async def post_json(url, headers):
    t = aiohttp.ClientTimeout(total=60, connect=10, sock_connect=10, sock_read=50)
    async with aiohttp.ClientSession(timeout=t) as s:
        async with s.post(url, headers=headers) as r:
            r.raise_for_status()
            # return await r.json()
            return await r.text(), r.status

class WebSailorMultiVisitTool(BaseTool):
    """A tool for visiting potentially multiple webpages and summarizing them with a LLM."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Expected tool_schema format:
        {
            "type": "function",
            "function": {
                "name": "visit",
                "description": "Visit webpage(s) and return the summary of the content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                        },
                        "goal": {
                            "type": "string",
                            "description": "The specific information goal for visiting webpage(s)."
                        }
                    },
                    "required": [
                        "url",
                        "goal"
                    ]
                }
            }
        },
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.summarize_model = config.get("summarize_model", "gpt-4.1-mini")
        self.client = AsyncOpenAI(base_url=config.get("summarize_model_base_url", "https://api.openai.com/v1"), 
                                  api_key=config.get("summarize_model_api_key", os.getenv("OPENAI_API_KEY")),
                                  timeout=config.get("summarize_model_timeout", 300))
        self.max_tokens = config.get("max_webpage_tokens", 28000)
        self.max_context_length = config.get("max_context_length", 32768)
        self.jina_api_key = config.get("jina_api_key", os.getenv("JINA_API_KEY"))

        if self.summarize_model.startswith('Qwen/Qwen3'):
            self.enc = transformers.AutoTokenizer.from_pretrained(self.summarize_model)
        else:
            self.enc = tiktoken.encoding_for_model("gpt-4o")
            
        self.extractor_prompt = config.get("extractor_prompt", """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{article_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**. You must escape every backslash (e.g. in LaTeX code) with double backslashes.
""")
            
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id
    
    async def call_server(self, msgs, max_tries=10):
        
        for attempt in range(max_tries):
            try:
                prompt_length = len(self.enc.encode(msgs[0]["content"])) + 1
                max_completion_tokens = 4096
                if self.summarize_model.startswith('Qwen/Qwen3'):
                    response = await self.client.chat.completions.create(
                        model=self.summarize_model,
                        messages=msgs,
                        max_completion_tokens=min(self.max_context_length - prompt_length, max_completion_tokens),
                        temperature=0.7,
                        top_p=0.8,
                        presence_penalty=1.5,
                        extra_body={
                            "top_k": 20, 
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    
                else:
                    response = await self.client.chat.completions.create(
                        model=self.summarize_model,
                        messages=msgs,
                        max_completion_tokens=min(self.max_context_length - prompt_length, max_completion_tokens),
                    )
                    
                content = response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except:
                if attempt == (max_tries - 1):
                    return ""
                continue
                
    async def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
        }
        max_retries = 3
        timeout = 10
        
        for attempt in range(max_retries):
            try:
                response, status = await post_json(f"https://r.jina.ai/{url}", headers=headers)
                # response = requests.get(
                #     f"https://r.jina.ai/{url}",
                #     headers=headers,
                #     timeout=timeout
                # )
                if status == 200:
                    webpage_content = response
                    return webpage_content
                else:
                    print(response)
                    raise ValueError("jina readpage error")
            except Exception as e:
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
        return "[visit] Failed to read page."
    
    # def truncate_content(self, content: str) -> int:
    #     # Check if the article content is too long
    #     if len(self.enc.encode(content)) > self.max_tokens:
    #         # Truncate the article content
    #         content = self.enc.decode(self.enc.encode(content)[:self.max_tokens])
    #     return content
                
    async def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            # Alternate between jina and aidata
            content = await self.jina_readpage(url)
            sevice = "jina"

            # Check if we got valid content
            # print(sevice)
            # print(content)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                # content = content[:WEBCONTENT_MAXLENGTH]
                content = self.enc.decode(self.enc.encode(content)[:self.max_tokens])
                messages = [{"role":"user", "content": self.extractor_prompt.format(article_content=content, goal=goal)}]
                parse_retry_times = 0
                raw = await self.call_server(messages)

                # 如果网页超长，返回结果是 {\n 这种形式
                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[visit] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    print(status_msg)
                    content = content[:truncate_length]
                    extraction_prompt = self.extractor_prompt.format(
                        article_content=content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = await self.call_server(messages)
                    summary_retries -= 1
                # 说明 raw 的长度大于10或者已经retry 超出了 
                parse_retry_times = 0
                while parse_retry_times < 3:
                    try:
                        # 尝试 parse json
                        raw = json.loads(raw)
                        break
                    except:
                        raw = await self.call_server(messages)
                        parse_retry_times += 1
                # parse 失败
                if parse_retry_times >= 3:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                # parse 成功
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                    useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

                    summary_retries -= 1

                if len(useful_information) < 10 and summary_retries < 0:
                    print("[visit] Could not generate valid summary after maximum retries")
                    useful_information = "[visit] Failed to read page"
                return useful_information
                
            # If we're on the last attempt, return the last result
            if attempt == max_attempts - 1:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                return useful_information
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            url = parameters["url"]
            goal = parameters["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields", 0.0, {"error": "Invalid request format"}

        if isinstance(url, str):
            response = await self.readpage(url, goal)
        else:
            response = []
            assert isinstance(url, list)
            sem = asyncio.Semaphore(3)
            async def _run(u: Any) -> str:
                async with sem:
                    res = await self.readpage(u, goal)
                    return str(res)
                
            results = await asyncio.gather(*(_run(u) for u in url))
            response = "\n=======\n".join(results)
        
        # print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip(), 0.0, {
            "url": url, 
            "goal": goal, 
            "result_length": len(response)
        }
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Base reward for successful search
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]