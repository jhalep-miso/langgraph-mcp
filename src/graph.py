from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import getpass

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

nodes = {"ACTION": "ACTION", "LLM": "LLM"}


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    def call_model(self, state: AgentState):
        messages = state["messages"]
        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}

    async def call_tools(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tc in tool_calls:
            print(f"Calling tool {tc['name']} with args {tc['args']}")
            if not tc['name'] in self.tools:
                print(f"Bad tool name {tc['name']}")
                result = "bad tool name, retry"
            else:
                result = await self.tools[tc['name']].ainvoke(tc['args'])
            results.append(
                ToolMessage(tool_call_id=tc['id'], name=tc['name'], content=str(result))
            )
        print('Tools called, back to model')
        return {'messages': results}

    def exists_action(self, state: AgentState) -> bool:
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def __init__(self, model: BaseChatModel, tools: List[BaseTool], checkpointer=None):
        graph = StateGraph(AgentState)

        # adding nodes
        graph.add_node(nodes['LLM'], self.call_model)
        graph.add_node(nodes['ACTION'], self.call_tools)
        graph.add_conditional_edges(
            nodes['LLM'], self.exists_action, {True: nodes['ACTION'], False: END}
        )
        # adding edges
        graph.add_edge(nodes['ACTION'], nodes['LLM'])

        # set entrypoints and endpoints
        graph.set_entry_point(nodes['LLM'])

        self.tools = {t.name: t for t in tools}
        self.model_with_tools = model.bind_tools(tools)
        self.graph = graph.compile(checkpointer=checkpointer)
