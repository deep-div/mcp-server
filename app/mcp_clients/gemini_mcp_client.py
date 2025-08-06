import asyncio
import os
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

async def main():

    client = MCPClient.from_config_file(
        os.path.join("app/config/gemini_mcp.json")
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    result = await agent.run(
        "Do google search and tell What is todays date and time in india pune, use google_search",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
    
    
# py -m app.mcp_clients.gemini_mcp_client