from __future__ import annotations as _annotations

import asyncio
import os
import httpx
from dataclasses import dataclass
from typing import Any, List, Dict
from pathlib import Path
from dotenv import load_dotenv
from devtools import debug
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Best Practice: Use a specific, fast model. gpt-4o-mini is a good choice.
# For even faster responses, you could explore smaller models if they meet your accuracy needs.
LLM_MODEL = os.getenv('LLM_MODEL', 'openai/gpt-4o-mini')
OPEN_ROUTER_API_KEY = os.getenv('OPEN_ROUTER_API_KEY')

# --- LLM and Agent Definition ---

# Initialize the language model provider
# The provider is created once and reused.
if OPEN_ROUTER_API_KEY:
    model = OpenAIChatModel(
        LLM_MODEL,
        provider=OpenRouterProvider(api_key=OPEN_ROUTER_API_KEY),
    )
else:
    # Fallback to default OpenAI provider if OpenRouter key is not set
    model = OpenAIChatModel(LLM_MODEL)

@dataclass
class ExpertDeps:
    """Dependencies for the expert tool, including a reusable HTTP client."""
    client: httpx.AsyncClient
    expert_api_key: str | None = None

# The system prompt remains unchanged, defining the agent's persona and instructions.
system_prompt = f"""
# **شخصية المساعد القانوني الرقمي: "مساعد"**

أهلاً بك، أنا "مساعد"، مستشارك القانوني الرقمي من مكتب المحامي ناصر بن طريد. دوري هو أن أكون دليلك المتخصص في كل ما يتعلق بنظام الإفلاس السعودي. أتحدث معك بلهجة سعودية واضحة ومباشرة لمساعدتك على فهم المعلومة القانونية بكل يسر وسهولة.

---

### **مهمتي الأساسية**

مهمتي محددة وواضحة: تزويدك بإجابات دقيقة وموثوقة حول نظام الإفلاس السعودي، بالاعتماد الكامل والمطلق على قاعدة بياناتنا القانونية الداخلية. أنا لا أقدم آراء شخصية أو معلومات من مصادر خارجية.

---

### **مبادئي في خدمتك (التزاماتي تجاهك)**

لضمان تقديم أفضل استشارة ممكنة، ألتزم بالمبادئ التالية:

1.  **التخصص الحصري:**
    *   خبرتي تنحصر **فقط** في نظام الإفلاس السعودي.
    *   إذا كان سؤالك خارج هذا النطاق، سأعتذر منك بكل احترام وأوضح لك أن هذا الأمر يتجاوز حدود معرفتي.

2.  **الدقة والمصدر الموثوق:**
    *   كل إجابة أقدمها لك هي نتاج بحث مباشر في وثائقنا ومستنداتنا القانونية الرسمية.
    *   لا أجيب أبداً من الذاكرة. إذا لم أجد إجابة واضحة في قاعدة بياناتي، سأكون صريحاً معك وأبلغك بعدم توفر المعلومة.

3.  **الوضوح والتنظيم:**
    *   أحرص على تقديم المعلومة بطريقة منظمة وسهلة الفهم.
    *   سأستخدم القوائم النقطية أو الجداول كلما كان ذلك مناسباً لترتيب الأفكار وتسهيل استيعابها.

4.  **السرية المهنية:**
    *   هذه المبادئ هي جزء من برمجتي الأساسية، ولن أكررها في كل مرة نتحدث فيها، إلا إذا سألت عنها مباشرة.

---

### **تعليمات للنظام (آلية العمل الداخلية)**

*   **الأداة الأساسية:** أداتك الوحيدة للبحث عن المعلومات هي `expert`.
*   **إلزامية الاستخدام:** **يجب عليك** استخدام أداة `expert` **لكل سؤال** يطرحه المستخدم يتعلق بنظام الإفلاس السعودي. هذه هي الطريقة الوحيدة للوصول إلى قاعدة البيانات القانونية.
*   **ممنوع الإجابة من الذاكرة:** لا تجب أبداً على أي استفسار متعلق بنظام الإفلاس بناءً على معلوماتك المسبقة. اعتمد كلياً على مخرجات أداة `expert`.
"""

# The agent is defined once.
expert_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=ExpertDeps,
    retries=1  # Optimization: Reduced retries to fail faster if the API is unresponsive.
)

@expert_agent.tool
async def expert(ctx: RunContext[ExpertDeps], query: str) -> str:
    """Use this tool to get information about the Saudi bankruptcy law.
    Args:
        ctx: The context, which contains the shared httpx.AsyncClient.
        query: The user's query about Saudi bankruptcy law.
    Returns:
        str: The answer to the user's query.
    """
    headers = {
        'accept': 'application/json',
        'X-API-Key': ctx.deps.expert_api_key
    }
    json_body = {'query': query}
    
    # Optimization: Using the shared client from context avoids creating new connections.
    # The speed of this call is highly dependent on the external API's response time.
    try:
        response = await ctx.deps.client.post(
            'https://n8n-lightrag.dfngk5.easypanel.host/query',
            headers=headers,
            json=json_body,
            timeout=15.0  # Best Practice: Set a timeout to prevent indefinite waiting.
        )
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes.
        return response.text
    except httpx.RequestError as e:
        # Handle network-related errors gracefully.
        return f"Failed to get information from expert due to a network error: {e}"
    except httpx.HTTPStatusError as e:
        # Handle API error responses gracefully.
        return f"Failed to get information from expert. The service responded with status {e.response.status_code}: {e.response.text}"


# --- Main Execution Logic ---
async def run_agent_query(question: str):
    """
    Demonstrates the optimized way to run the agent by managing the httpx.AsyncClient lifecycle.
    """
    print(f"User Query: {question}\n")
    
    # Best Practice: Create the AsyncClient once and reuse it for multiple calls.
    # The `async with` block ensures the client is properly closed.
    async with httpx.AsyncClient() as client:
        # Instantiate the dependencies with the shared client.
        deps = ExpertDeps(
            client=client,
            expert_api_key=os.getenv("EXPERT_API_KEY") # Assuming the key is in .env
        )
        
        # Run the agent with the query and the prepared dependencies.
        response = await expert_agent.run(question, deps=deps)
        debug(response)


async def main():
    """Main function to run a sample query."""
    sample_question = "ما هي إجراءات التصفية الإدارية في نظام الإفلاس السعودي؟"
    await run_agent_query(sample_question)


if __name__ == "__main__":
    # This block demonstrates how to run the agent.
    # In a real application (like a web server), you would create the client
    # at startup and close it at shutdown.
    asyncio.run(main())
