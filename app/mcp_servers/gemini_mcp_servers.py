import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp.server.fastmcp import FastMCP
from app.gemini.gemini_llm import GeminiLLM
from app.gemini.gemini_tools import GeminiTools

# Initialize Gemini LLM and tools
geminillm = GeminiLLM()
geminitools = GeminiTools(geminillm)

# Create an MCP server
mcp = FastMCP("Gemini Agent")

# Tool: Gemini Thinking
@mcp.tool(name="gemini_thinking_mode")
def gemini_thinking_tool(prompt: str) -> str:
    """
    Stream  reasoning process including its internal thoughts and final response.

    This tool uses Gemini's thinking configuration to stream both:
    - Internal "thoughts" (reasoning steps, intermediate ideas)
    - Final response text to the given prompt

    Use this tool when:
    - The user query requires deep reasoning or multi-step thinking
    - Interpretability or step-by-step breakdown is valuable
    - The user explicitly requests "thinking" or says "use thinking mode"

    Args:
        prompt (str): The input question or instruction for Gemini.

    Returns:
        str: A stream of thought-labeled and response-labeled output combined as a single string.
    """
    output = []
    for chunk in geminitools.stream_thinking_mode(prompt):
        prefix = "[THOUGHT] " if chunk["thought"] else ""
        output.append(f"{prefix}{chunk['text']}")
    return "\n".join(output)


# Tool: Google Search
@mcp.tool(name="google_search")
def google_search_tool(prompt: str) -> str:
    """
    Perform real-time Google Search through Gemini and return a synthesized textual result.

    This tool allows Gemini to use its Google Search plugin capability to:
    - Perform a live Google search using the given prompt
    - Extract relevant content from results
    - Return a well-formed, concise answer based on the latest information
    - Use the tool when you are uncertain about the answer or need to explore

    Ideal for questions requiring current facts, news, or real-time data.

    Args:
        prompt (str): A search query or question needing live information.

    Returns:
        str: Gemini's response based on live search results.
    """
    return "\n".join(chunk for chunk in geminitools.stream_google_search(prompt))


# Tool: Code Execution
@mcp.tool(name="gemini_code_execution")
def code_execution_tool(prompt: str) -> str:
    """
    Generate and execute Python code using Gemini's built-in code execution tool.

    This tool allows Gemini to:
    - Generate code in response to a prompt (e.g., logic, math, data processing)
    - Execute the code in a secure environment
    - Return both the code and its output

    It is useful for technical prompts where direct computation, scripting, or automation is needed.

    Args:
        prompt (str): A programming task or coding question in natural language.

    Returns:
        str: The generated Python code and the output of its execution.
    """
    return "\n".join(chunk for chunk in geminitools.stream_code_execution(prompt))

# uv run mcp dev app/mcp_servers/gemini_mcp_servers.py  