# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
import sys
import requests

from .article import Article
from .jina_client import JinaClient
from .readability_extractor import ReadabilityExtractor

logger = logging.getLogger(__name__)

import aiohttp, asyncio

async def post_json(url, headers):
    t = aiohttp.ClientTimeout(total=60, connect=10, sock_connect=10, sock_read=50)
    async with aiohttp.ClientSession(timeout=t) as s:
        async with s.post(url, headers=headers) as r:
            r.raise_for_status()
            # return await r.json()
            return await r.text()

class Crawler:
    async def crawl(self, url: str) -> Article:
        # To help LLMs better understand content, we extract clean
        # articles from HTML, convert them to markdown, and split
        # them into text and image blocks for one single and unified
        # LLM message.
        #
        # Jina is not the best crawler on readability, however it's
        # much easier and free to use.
        #
        # Instead of using Jina's own markdown converter, we'll use
        # our own solution to get better readability results.
        jina_client = JinaClient()
        # html = jina_client.crawl(url, return_format="html")
        
        headers = {}
        if os.getenv("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        else:
            logger.warning(
                "Jina API key is not set. Provide your own key to access a higher rate limit. See https://jina.ai/reader for more information."
            )
        # response = requests.post(f"https://r.jina.ai/{url}", headers=headers, timeout=60)
        response = await post_json(f"https://r.jina.ai/{url}", headers=headers)
        
        article = Article(title="", html_content=response)
        article.url = url
        
        # try:
        #     raise Exception('test')
        #     extractor = ReadabilityExtractor()
        #     article = extractor.extract_article(html)
        #     article.url = url
        # except Exception as e:
        #     # logger.warning(f"Failed to crawl html from {url}. Using Jina API directly")
        #     # logger.warning(f"Using Jina API directly")
        #     headers = {}
        #     if os.getenv("JINA_API_KEY"):
        #         headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        #     else:
        #         logger.warning(
        #             "Jina API key is not set. Provide your own key to access a higher rate limit. See https://jina.ai/reader for more information."
        #         )
        #     response = requests.post(f"https://r.jina.ai/{url}", headers=headers, timeout=60)
            
        #     article = Article(title="", html_content=response.text)
        #     article.url = url
        
        # if html:
        #     extractor = ReadabilityExtractor()
        #     article = extractor.extract_article(html)
        #     article.url = url
        # else:
        #     logger.warning(f"Failed to crawl html from {url}. Using Jina API directly")
        #     headers = {}
        #     if os.getenv("JINA_API_KEY"):
        #         headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        #     else:
        #         logger.warning(
        #             "Jina API key is not set. Provide your own key to access a higher rate limit. See https://jina.ai/reader for more information."
        #         )
        #     response = requests.post(f"https://r.jina.ai/{url}", headers=headers)
            
        #     article = Article(title="", html_content=response.text)
            
        return article
