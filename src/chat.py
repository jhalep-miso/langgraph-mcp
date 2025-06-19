from graph import build_graph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from uuid import uuid4

memory = MemorySaver()

graph = build_graph(memory)

thread = {"configurable": {"thread_id": uuid4()}}


def stream_graph_updates(user_input: str):
    messages = [HumanMessage(content=user_input)]
    for event in graph.stream({"messages": messages}, thread):
        for value in event.values():
            print("\nAssistant:", value['messages'][-1].content)


while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Assistant: Goodbye!")
        break
    stream_graph_updates(user_input)
