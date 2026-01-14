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

import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, parse_qs, urlparse


class MinimalBrowser:
    """A browser for visiting web pages and managing content viewports."""
    
    def __init__(
        self, 
        firecrawl_api_key: str,
        viewport_size: Optional[int] = 1024 * 4,
    ):
        try:
            from firecrawl import FirecrawlApp
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import SRTFormatter
        except ImportError:
            raise ImportError(
                "Required packages not found. Install with: "
                "pip install firecrawl-py youtube-transcript-api"
            )
            
        self.firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
        self.viewport_size = viewport_size
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = list()
        self.history: List[Tuple[str, float]] = list()

        self.page_title: str = ""
        self._page_content: str = ""
        self._find_on_page_query: Union[str, None] = None
        self._find_on_page_last_result: Union[int, None] = None 

        self.ytt_api = YouTubeTranscriptApi()
        self.ytt_formatter = SRTFormatter()

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        if not self.history:
            return "about:blank"
        return self.history[-1][0]
    
    def reset(self) -> None:
        """Reset the browser state."""
        self.history = []
        self.viewport_current_page = 0
        self.viewport_pages = []
        self.page_title = ""
        self._page_content = ""
        self._find_on_page_query = None
        self._find_on_page_last_result = None

    def set_address(self, uri_or_path: str) -> None:
        """Set the current address and fetch the page content."""
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        else:
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0
        self._find_on_page_query = None
        self._find_on_page_last_result = None

    def _fetch_page(self, url: str) -> None:
        """Fetch page content from URL."""
        try:
            if url.startswith("https://www.youtube.com/watch?"):
                transcript_text = ""
                params = parse_qs(urlparse(url).query)
                if "v" in params:
                    assert isinstance(params["v"][0], str)
                    video_id = str(params["v"][0])
                    transcript = self.ytt_api.fetch(video_id)
                    transcript_text = self.ytt_formatter.format_transcript(transcript)
                self.page_title = f"# YouTube Video Transcript"
                self._set_page_content(transcript_text)
            else:
                scrape_result = self.firecrawl_app.scrape_url(url, formats=['markdown']).dict()
                self.page_title = f"{scrape_result.get('title', '')}\t{scrape_result.get('description', '')}"
                self._set_page_content(scrape_result['markdown'])
        except Exception as e:
            self.page_title = f"Error encountered while fetching {url}"
            self._set_page_content(f"Error: {e}")

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        if not self.viewport_pages:
            return ""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]
    
    def get_all_viewports(self) -> List[str]:
        """Return the content of all viewports."""
        return [self.page_content[bounds[0] : bounds[1]] for bounds in self.viewport_pages]
    
    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str, split: bool = True) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        if split:
            self._split_pages()
        else:
            self.viewport_pages = [(0, len(content))]
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def _split_pages(self) -> None:
        """Split content into viewport-sized pages."""
        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx
            
    def visit_page(self, path_or_uri: str) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri)
        return self.viewport

    def state(self) -> Tuple[str, str]:
        """Return the current browser state as (header, viewport_content)."""
        header = f"Address: {self.address}\n"
        if self.page_title is not None:
            header += f"Title: {self.page_title}\n"

        current_page = self.viewport_current_page
        total_pages = len(self.viewport_pages)

        address = self.address
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == address:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                break

        header += f"Viewport position: Showing page {current_page + 1} of {total_pages}.\n"
        return (header, self.viewport)

    def get_all_states(self) -> List[Tuple[str, str]]:
        """Return the state for all viewports."""
        all_viewports = self.get_all_viewports()
        all_states = []
        for i, v in enumerate(all_viewports):
            header = f"Address: {self.address}\n"
            if self.page_title is not None:
                header += f"Title: {self.page_title}\n"
            total_pages = len(self.viewport_pages)
            address = self.address
            for j in range(len(self.history) - 2, -1, -1):  # Start from the second last
                if self.history[j][0] == address:
                    header += f"You previously visited this page {round(time.time() - self.history[j][1])} seconds ago.\n"
                    break
            header += f"Viewport position: Showing page {i + 1} of {total_pages}.\n"
            all_states.append((header, v))
        return all_states 

def populate_template(template: str, variables: Dict[str, Any]) -> str:
    from jinja2 import StrictUndefined, Template
    
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

class LiteLLMModel:
    """
    Wrapper for LiteLLM model calls
    """
    def __init__(
        self, 
        model: str = "gpt-4o",
        api_base=None,
        api_key="sk-123",
        enable_thinking=False,
        **kwargs
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.kwargs = kwargs

    def __call__(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        **kwargs,
    ) -> str:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        try:
            import litellm
            logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        except ImportError:
            raise ImportError("litellm package is required. Install with: pip install litellm")
        
        # litellm._turn_on_debug()
        completion_kwargs = {
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "messages": messages,
        }
        
        if not self.enable_thinking and "Qwen/Qwen3-32B".lower() in self.model.lower():
            qwen_kwargs = {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 1.5,
                "extra_body": {
                    "top_k": 20, 
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
            completion_kwargs.update(qwen_kwargs)
        completion_kwargs.update(self.kwargs)
        completion_kwargs.update(kwargs)

        if messages == []:
            raise ValueError("messages should not be empty")
        elif isinstance(messages[0], dict):
            response = litellm.completion(**completion_kwargs).choices[0].message.content
        elif isinstance(messages[0], list):
            response = [r.choices[0].message.content for r in litellm.batch_completion(**completion_kwargs)]

        return response