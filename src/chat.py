from graph import Agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from uuid import uuid4
from langchain.chat_models import init_chat_model
from tools.search_papers import search_papers
from tools.extract_info import extract_info

memory = MemorySaver()

tools = [search_papers, extract_info]
model = init_chat_model(
    "gemini-2.0-flash", model_provider="google_genai", temperature=0.0
)

agent = Agent(model, tools, memory)

thread = {"configurable": {"thread_id": uuid4()}}


def stream_graph_updates(user_input: str):
    messages = [HumanMessage(content=user_input)]
    for event in agent.graph.stream({"messages": messages}, thread):
        for value in event.values():
            print("\nAssistant:", value['messages'][-1].content)


def chat_loop():
    """Run an interactive chat loop"""
    print("\nMCP Chatbot Started!")
    print("Type your queries or 'quit', 'exit', 'q' to exit.")
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Assistant: Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    chat_loop()
