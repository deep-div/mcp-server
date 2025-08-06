import asyncio
import os
from mcp_use import MCPAgent, MCPClient
from langchain_google_genai import ChatGoogleGenerativeAI

async def main():

    client = MCPClient.from_config_file(
        os.path.join("app/config/playright_mcp.json")
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=10, verbose=False)

    async for chunk in agent.stream("Find the best restaurant in San Francisco"):
        print(chunk)

if __name__ == "__main__":
    asyncio.run(main())
    
    
# py -m app.mcp_clients.playright_mcp_client