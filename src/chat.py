from graph import Agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from uuid import uuid4
from langchain.chat_models import init_chat_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import nest_asyncio
from typing import List
from langchain.tools import BaseTool

nest_asyncio.apply()

memory = MemorySaver()

model = init_chat_model(
    "gemini-2.0-flash", model_provider="google_genai", temperature=0.0
)


thread = {"configurable": {"thread_id": uuid4()}}


class MCP_ChatBot:
    def __init__(self):
        self.session: ClientSession = None
        self.tools: List[BaseTool] = []
        self.agent: Agent = None

    async def stream_graph_updates(self, user_input: str):
        messages = [HumanMessage(content=user_input)]
        async for event in self.agent.graph.astream({"messages": messages}, thread):
            for value in event.values():
                print("\nAssistant:", value['messages'][-1].content)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit', 'exit', 'q' to exit.")
        while True:
            try:
                user_input = input("\nUser: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Assistant: Goodbye!")
                    break
                await self.stream_graph_updates(user_input)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "mcp_project/research_server.py"],
            env=None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.tools = await load_mcp_tools(session)
                print(
                    "\nConnected to server with tools:",
                    [tool.name for tool in self.tools],
                )
                self.agent = Agent(model, self.tools, memory)
                await self.chat_loop()

    async def connect_to_sse_server_and_run(self):
        async with sse_client("http://localhost:8120/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.tools = await load_mcp_tools(session)
                print(
                    "\nConnected to server with tools:",
                    [tool.name for tool in self.tools],
                )
                self.agent = Agent(model, self.tools, memory)
                await self.chat_loop()

    async def connect_to_multiple_servers_and_run(self):
        client = MultiServerMCPClient(
            {
                "research": {
                    "command": "uv",
                    "args": ["run", "mcp_project/research_server.py"],
                    "transport": "stdio",
                },
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                    "transport": "stdio",
                },
                "fetch": {
                    "command": "uvx",
                    "args": ["mcp-server-fetch"],
                    "transport": "stdio",
                },
            }
        )
        self.tools = await client.get_tools()
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

        self.agent = Agent(model, self.tools, memory)
        await self.chat_loop()


if __name__ == "__main__":
    chatbot = MCP_ChatBot()
    # asyncio.run(chatbot.connect_to_server_and_run())
    # asyncio.run(chatbot.connect_to_multiple_servers_and_run())
    asyncio.run(chatbot.connect_to_sse_server_and_run())
