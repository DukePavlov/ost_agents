
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_agent import builder

# Compile with conversation memory
graph = builder.compile(
    checkpointer=InMemorySaver()
)

def ask(question, thread_id):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 14
    }

    result = graph.invoke(
        {"messages": [
            {"role": "user", "content": question}
        ]},
        config=config
    )

    print("\nQUESTION:", question)
    print("ANSWER:", result["messages"][-1].content)
    return config

# Experiment 1: Same conversation
config = ask(
    "Compare mean glucose between groups.",
    "student-1"
)

ask("What about insulin?", "student-1")

# Experiment 2: New conversation
ask("What about insulin?", "student-2")

# Experiment 3: Inspect stored state
state = graph.get_state(config)

print("\nStored messages:")
for message in state.values["messages"]:
    print(message.type, str(message.content)[:150])