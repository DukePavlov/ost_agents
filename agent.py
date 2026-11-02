"""Local, tool-using LLM agent for exploring a small public healthcare dataset."""
from __future__ import annotations

import argparse
import json

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from agent_tools import TOOLS
from stats import load_data

SYSTEM_PROMPT = """You are a data analyst.
The dataset contains 116 observations: breast cancer patients and healthy controls.
Use the provided tools for ALL dataset statistics, counts, and comparisons.
Never invent numerical values, diagnoses, predictive accuracy, or causal relationships.
If a request cannot be answered with the available tools, explain the limitation.
You are performing descriptive educational analysis, not clinical diagnosis.
Keep responses concise; distinguish healthy controls from breast cancer patients.
"""


def show_tool_trace(messages: list) -> int:
    """Print observable tool requests and tool results, NOT private model reasoning."""
    count = 0
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                count += 1
                print(f"\n[TOOL CALL] {call['name']}({json.dumps(call['args'])})")
        if isinstance(message, ToolMessage):
            print(f"[TOOL RESULT] {message.content}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Breast Cancer Coimbra teaching agent")
    parser.add_argument("--model", default="qwen3:1.7b", help="Installed Ollama model")
    parser.add_argument("--question", help="One question; otherwise interactive mode")
    args = parser.parse_args()

    # Check dataset before the first (potentially slow) model inference.
    frame = load_data()
    print(f"Dataset loaded: {len(frame)} rows. Model: {args.model}")
    print("Note: results describe the dataset; no clinical diagnoses are made.")

    agent = create_agent(
        model=ChatOllama(model=args.model, temperature=0),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    def ask(question: str) -> None:
        # Questions are intentionally independent in this first lecture.
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": 14},
            )
        except Exception as exc:
            print(f"Agent error: {type(exc).__name__}: {exc}")
            print("Check that Ollama is running and the requested model is installed.")
            return
        calls = show_tool_trace(result["messages"])
        print("\n[ANSWER]")
        print(result["messages"][-1].content)
        if not calls:
            print("\n[NOTICE] The LLM used no tools. Do not trust unsupported numerical claims.")

    if args.question:
        ask(args.question)
    else:
        print("\nEnter a dataset question, or 'quit' to exit.")
        while True:
            try:
                q = input("\nQuestion > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in {"quit", "exit", "q"}:
                break
            if q:
                ask(q)


if __name__ == "__main__":
    main()
