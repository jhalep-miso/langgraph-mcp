from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage, ToolMessage
from langchain.chat_models import init_chat_model
from tools.search_papers import search_papers
from tools.extract_info import extract_info
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import getpass


load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


tools_list = [search_papers, extract_info]
tools = {t.name: t for t in tools_list}
nodes = {"ACTION": "ACTION", "LLM": "LLM"}

model = init_chat_model(
    "gemini-2.0-flash", model_provider="google_genai", temperature=0.0
)
model_with_tools = model.bind_tools(tools_list)


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


def call_model(state: AgentState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def call_tools(state: AgentState):
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for tc in tool_calls:
        print(f"Calling tool {tc['name']} with args {tc['args']}")
        if not tc['name'] in tools:
            print(f"Bad tool name {tc['name']}")
            result = "bad tool name, retry"
        else:
            result = tools[tc['name']].invoke(tc['args'])
        results.append(
            ToolMessage(tool_call_id=tc['id'], name=tc['name'], content=str(result))
        )
    print('Tools called, back to model')
    return {'messages': results}


def exists_action(state: AgentState) -> bool:
    result = state['messages'][-1]
    return len(result.tool_calls) > 0


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)

    # adding nodes
    graph.add_node(nodes['LLM'], call_model)
    graph.add_node(nodes['ACTION'], call_tools)
    graph.add_conditional_edges(
        nodes['LLM'], exists_action, {True: nodes['ACTION'], False: END}
    )
    # adding edges
    graph.add_edge(nodes['ACTION'], nodes['LLM'])

    # set entrypoints and endpoints
    graph.set_entry_point(nodes['LLM'])
    return graph.compile(checkpointer=checkpointer)
