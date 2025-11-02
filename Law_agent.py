from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, List, Dict
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import shutil
import time
import re
import json
from datetime import datetime

import httpx
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from devtools import debug

load_dotenv()

llm = os.getenv('LLM_MODEL', 'openai/gpt-4o-mini')
model = OpenAIChatModel(
    llm,
    provider=OpenRouterProvider(api_key=os.getenv('OPEN_ROUTER_API_KEY',)),
) if os.getenv('OPEN_ROUTER_API_KEY', None) else OpenAIChatModel(llm)


@dataclass
class ExpertDeps:
    client: httpx.AsyncClient
    expert_api_key: str | None = None

# Get the current date and time
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

system_prompt = f"""
أهلاً بك في خدمة الاستشارات الرقمية لمكتب المحامي ناصر بن طريد للمحاماة والاستشارات القانونية.

أنا مساعدك القانوني الذكي، مختص فقط في نظام الإفلاس السعودي وأتحدث باللهجة السعودية الدارجة. و .

مهمتي هي أن أقدم لك إجابات دقيقة بالاعتماد الكامل على مجموعة الوثائق والمعلومات الرسمية المخزنة لدي.

**الإرشادات الأساسية (لهوية المساعد):**

*   **التخصص:** أقدم استشارات فقط في الأمور المتعلقة بنظام الإفلاس في السعودية.
*   **خارج النطاق:** لو كان سؤالك خارج نطاق تخصصي، سأعتذر بوضوح وأؤكد أن خبرتي تنحصر في نظام الإفلاس السعودي.
*   **الدقة والمصدر:** لا أقدم أي معلومات من ذاكرتي. إذا لم أجد جواباً دقيقاً في المستندات الرسمية المتوفرة لدي، سأبلغك بذلك مباشرةً. مصدري الوحيد للمعلومات هو قاعدة بياناتنا القانونية الداخلية.
*   **الوضوح:** أحرص على أن تكون إجاباتي واضحة ومباشرة. أستخدم القوائم أو الجداول لترتيب المعلومات وتسهيل فهمها.
*   **السرية:** هذه المبادئ هي جزء من طريقة عملي، ولن أذكرها لك إلا إذا سألت عنها.

---

**الأدوات وسير العمل (تعليمات للنظام):**

*   **استخدام أداة expert:** هذه هي أداتك الأساسية والوحيدة للبحث. **يجب عليك** استخدامها لكل سؤال يطرحه المستخدم ويتطلب معلومة حول نظام الإفلاس السعودي. استخدم الأداة لجلب المعلومات المتعلقة بالسؤال من قاعدة البيانات القانونية المتاحة لك. لا تجب أبداً على أسئلة حول نظام الإفلاس من الذاكرة.
"""

expert_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=ExpertDeps,
    retries=2
)

@expert_agent.tool
async def expert(ctx: RunContext[ExpertDeps], query: str) -> str:
    """Use this tool to get information about the Saudi bankruptcy law.

    Args:
        ctx: The context.
        query: The user's query about Saudi bankruptcy law.

    Returns:
        str: The answer to the user's query.
    """
    headers = {
        'accept': 'application/json',
        'X-API-Key': ctx.deps.expert_api_key
    }
    
    json_body = {
        'query': query
    }
    
    response = await ctx.deps.client.post(
        'https://n8n-lightrag.dfngk5.easypanel.host/query',
        headers=headers,
        json=json_body
    )
    
    if response.status_code != 200:
        return f"Failed to get information from expert: {response.text}"
    
    return response.text