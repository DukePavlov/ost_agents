from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from agent_tools import TOOLS
from agent import SYSTEM_PROMPT

# 1. Initialize the model
model = ChatOllama(
    model="qwen3:1.7b",
    temperature=0
)

model_with_tools = model.bind_tools(TOOLS)

# 2. Define the LLM node
def call_llm(state: MessagesState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# 3. Construct the graph
builder = StateGraph(MessagesState)

builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

# 4. Compile and execute
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [{
            "role": "user",
            "content": "Compare mean glucose between groups."
        }]},
        config={"recursion_limit": 14}
    )

    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            print("[TOOL CALL]", message.tool_calls)
        if message.type == "tool":
            print("[TOOL RESULT]", message.content)

    print("[ANSWER]", result["messages"][-1].content)