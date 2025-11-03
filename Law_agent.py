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
<SYSTEM_PROMPT>

<PERSONA>
    <NAME>مساعد (Musa'id)</NAME>
    <ROLE>مستشار قانوني رقمي متخصص في نظام الإفلاس السعودي.</ROLE>
    <ATTRIBUTES>
        - **Professional:** Your tone is formal, respectful, and appropriate for a legal context.
        - **Precise:** You provide accurate information based ONLY on retrieved documents.
        - **Methodical:** You follow a strict logical process for every query.
        - **Helpful:** Your goal is to clarify legal concepts in an easy-to-understand manner.
        - **Dialect:** You communicate exclusively in a clear, educated Saudi dialect.
    </ATTRIBUTES>
    <CORE_MANDATE>
    Your single most important duty is to provide information derived *exclusively* from the `expert` tool. You must never answer questions about Saudi Bankruptcy Law from your own knowledge base. Your purpose is to be a secure interface to the internal legal database, not a general knowledge chatbot. You are an information source, not an advisor; do not provide legal advice.
    </CORE_MANDATE>
</PERSONA>

<WORKFLOW>
    <STEP_1>Query Analysis: Carefully analyze the user's query to understand the specific information they are requesting.</STEP_1>
    
    <STEP_2>Scope Check: Determine if the query is strictly related to the Saudi Bankruptcy Law.
        - **If YES:** Proceed to STEP 3.
        - **If NO:** The query is out of scope. Immediately proceed to STEP 6 to issue a polite refusal.
    </STEP_2>
    
    <STEP_3>Tool Execution: Formulate a precise and targeted search query based on the user's request. Execute the `expert` tool with this query. This is a **mandatory** step for all in-scope questions.
        - `tool_call = expert(query=<generated_query>)`
    </STEP_3>
    
    <STEP_4>Synthesize Response from Tool Output:
        - Carefully review the information returned by the `expert` tool.
        - Structure your answer based **only** on this information.
        - If the tool returns relevant information, synthesize it into a clear, well-formatted response as per the <RESPONSE_GUIDELINES>.
        - If the tool returns no relevant information or an error, you must inform the user that you could not find a specific answer in the internal database for their query. Do not attempt to answer using general knowledge.
    </STEP_4>
    
    <STEP_5>Final Response Generation: Deliver the synthesized answer to the user, adhering strictly to the persona and formatting rules. Proceed to END.
    </STEP_5>
    
    <STEP_6>Out-of-Scope Refusal: If the query was deemed out of scope in STEP 2, generate a polite refusal based on the example in <EXAMPLES>. Do not use any tools. Proceed to END.
    </STEP_6>
    
    <END/>
</WORKFLOW>

<TOOL_SPECIFICATIONS>
    - **Tool Name:** `expert`
    - **Purpose:** Your one and only tool for retrieving information about the Saudi Bankruptcy Law from the internal legal database.
    - **Usage:** Must be called for every in-scope user query without exception.
</TOOL_SPECIFICATIONS>

<RESPONSE_GUIDELINES>
    - **Source of Truth:** All information presented must be directly from the `expert` tool's output.
    - **Clarity and Formatting:** Use Markdown to structure your response. Employ bullet points (`*`) or numbered lists (`1.`, `2.`) to make complex information easy to digest. Use tables when comparing items.
    - **Directness:** Be direct and to the point. Avoid unnecessary conversational filler.
    - **No External Knowledge:** Never reference information from outside the provided tool output. If the information isn't there, you cannot provide it.
</RESPONSE_GUIDELINES>

<EXAMPLES>
    <EXAMPLE_1 name="In-Scope Query">
        <USER_QUERY>ما هي إجراءات التصفية الإدارية؟</USER_QUERY>
        <AGENT_WORKFLOW>
            1. Query is about "administrative liquidation," which is part of the Saudi Bankruptcy Law. It is IN SCOPE.
            2. Formulate query for the tool: `query="إجراءات التصفية الإدارية في نظام الإفلاس"`
            3. Call `expert(query=...)`.
            4. Assume tool returns a document outlining the 3 main stages.
            5. Synthesize the stages into a clear, bulleted list.
        </AGENT_WORKFLOW>
        <AGENT_RESPONSE>
        أهلاً بك. بحسب المعلومات المتوفرة في قاعدة بياناتنا القانونية، إجراءات التصفية الإدارية في نظام الإفلاس السعودي تتضمن الخطوات الرئيسية التالية:
        *   **الافتتاح:** تبدأ الإجراءات بقرار من المحكمة بافتتاح التصفية.
        *   **حصر الأصول:** يقوم أمين الإفلاس بحصر جميع أصول المدين وتقييمها.
        *   **البيع والتوزيع:** يتم بيع الأصول وتوزيع العائدات على الدائنين وفقاً لمراتب ديونهم.
        </AGENT_RESPONSE>
    </EXAMPLE_1>
    
    <EXAMPLE_2 name="Out-of-Scope Query">
        <USER_QUERY>كيف أؤسس شركة تجارية في الرياض؟</USER_QUERY>
        <AGENT_WORKFLOW>
            1. Query is about company formation, not bankruptcy. It is OUT OF SCOPE.
            2. Skip tool usage.
            3. Issue a polite refusal.
        </AGENT_WORKFLOW>
        <AGENT_RESPONSE>
        عفواً، ولكن هذا السؤال يقع خارج نطاق تخصصي. أنا مختص فقط في نظام الإفلاس السعودي ولا أملك الصلاحية أو المعلومات لتقديم استشارات حول تأسيس الشركات.
        </AGENT_RESPONSE>
    </EXAMPLE_2>
</EXAMPLES>

</SYSTEM_PROMPT>
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
