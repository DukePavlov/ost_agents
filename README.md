# Healthcare Data Analysis Agent - Breast Cancer Coimbra

A small, **real LLM tool-calling agent** for the ELTE *Open-Source Technologies* master's lecture. Use **Python + pandas + LangChain + Ollama** to explore a public healthcare dataset. For Windows, with no paid API key.

**The educational goal is tool calling and evidence-grounded descriptive analysis, not cancer detection.** The agent cannot diagnose cancer and must not be used for clinical decisions.

## Project files

```text
breast-cancer-coimbra-agent/
├── README.md                 # These instructions
├── requirements.txt         # Python dependencies
├── download_data.py         # Downloads the original UCI dataset
├── stats.py                 # Deterministic pandas computations
├── agent_tools.py           # Exposes 3 analysis functions to the LLM
├── agent.py                 # Model + tools + visible execution trace
├── data/README.md           # Dataset provenance and license
├── examples/questions.txt   # Test prompts for the classroom
└── tests/test_stats.py      # Offline unit tests (synthetic fixture)
```

## 1. Install prerequisites (Windows 10/11)

Install **Python 3.11 or 3.12**, **VS Code** (optional), and **Ollama**:
- Python: https://www.python.org/downloads/windows/
- Ollama: https://ollama.com/download/windows
- VS Code: https://code.visualstudio.com/

Open **PowerShell** in this project folder. The commands below call the virtual environment's Python executable directly, so PowerShell execution policy changes are unnecessary.

```powershell
irm https://ollama.com/install.ps1 | iex (as administrator)
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Download the public dataset

```powershell
.\.venv\Scripts\python.exe download_data.py
```

This fetches the **UCI Breast Cancer Coimbra** dataset (116 observations, 9 measured features and a label): https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra

The script saves `data/dataR2.csv`. **Classification = 1** denotes healthy controls; **Classification = 2** denotes breast cancer patients. The label is an existing dataset field and is never inferred by the agent.

If the download script fails due to a network restriction, download `dataR2.csv` manually from the UCI dataset page and place it in `data/`.

## 3. Download a local LLM (one time)

Open a PowerShell terminal after installing Ollama:

```powershell
ollama pull qwen3:1.7b
```

This small model supports tool calling but may choose tools inconsistently. For the instructor demo and faster reliability when hardware allows, try `ollama pull qwen3:4b` and run the agent with `--model qwen3:4b`. Initial download requires internet; afterwards model inference is local. A CPU-only laptop may be slow.

Make sure Ollama is running before launching the Python agent.

## 4. Run the LLM agent

```powershell
.\.venv\Scripts\python.exe agent.py --question "How many people are in each group?"
```

Try a multi-tool question:

```powershell
.\.venv\Scripts\python.exe agent.py --question "How many patients are there and how does their average glucose compare with the controls?"
```

Or launch interactive question mode:

```powershell
.\.venv\Scripts\python.exe agent.py
```

The terminal shows **[TOOL CALL]**, **[TOOL RESULT]**, and **[ANSWER]**. These are observable execution events, not the LLM's private reasoning. Each question is independent; conversation memory is deliberately omitted for Lecture 1.

If the model answers without using any tools, the script explicitly warns you. Try a clearer request, retry, or switch to the larger model; LLM tool-selection behavior is not guaranteed.

## 5. What the three tools do

| Tool | Parameters | Returned analysis |
|---|---|---|
| `get_cohort_summary` | none | Total observations; counts of each class |
| `compare_group_means` | `feature` | Means in each class and their difference |
| `get_feature_statistics` | `feature`, optional `group` | n, mean, median, std, min, max |

Supported `feature` values: `Age`, `BMI`, `Glucose`, `Insulin` (case insensitive). Valid `group` values: `all`, `healthy`, `cancer`.

Example questions are in `examples/questions.txt`. A multi-part question may require the LLM to call multiple tools and combine their outputs.

## 6. Run tests without Ollama or external network

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use a **small synthetic fixture**, not real patient data. They validate calculations independently of the LLM, which is important when evaluating tool-using systems.

## 7. Student assignment

1. Run one single-tool and one multi-tool question; record which tools the agent called.
2. Check the returned statistics using pandas or the unit tests.
3. Modify the agent prompt to require numerical claims to come from tools. Does tool-selection behavior change?
4. **Optional extension:** implement and register `get_group_median(feature)`; ask about BMI median in both groups.

Discuss: When does an LLM refuse to call a tool? Can a good-sounding answer still be unsupported by the dataset?

---

## 8. Lecture 2 — Building Agentic Systems with LangGraph

In Lecture 2, we extend our healthcare AI agent with explicit graph-based orchestration and conversation memory.

### Learning Objectives

- Understand LangGraph's state, nodes, and edges.
- Implement an LLM–tool execution loop.
- Add conversational memory using checkpointing.
- Inspect execution history and evaluate agent behavior.

### New Files

- `langgraph_agent.py` — Implements the agent using `StateGraph`, `ToolNode`, and conditional edges.
- `memory_demo.py` — Demonstrates state persistence using `InMemorySaver` and conversation IDs.

The dataset and statistical tools from Lecture 1 remain unchanged.

### Running the LangGraph Agent

Install the updated dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the graph:

```powershell
.\.venv\Scripts\python.exe langgraph_agent.py
```

The agent processes questions using an explicit execution graph:

`START → LLM → Tools → LLM → END`

The tool-execution loop repeats when additional actions are requested.

### Adding Conversation Memory

Run:

```powershell
.\.venv\Scripts\python.exe memory_demo.py
```

LangGraph uses a checkpointer to retain conversation messages.

The `thread_id` identifies a conversation, allowing subsequent requests to access its previous context.

Example:

**Question 1:** Compare mean glucose between the two groups.

**Question 2:** What about insulin?

With the same `thread_id`, the agent can interpret the second question using the previous conversation.

A new `thread_id` starts an independent conversation.

**Note:** `InMemorySaver` retains state only while the Python process is running.

### Hands-On Exercise

1. Run the LangGraph implementation and inspect tool calls.
2. Ask related questions using the same `thread_id`.
3. Repeat the follow-up question using a new `thread_id`.
4. Inspect stored messages using `graph.get_state(config)`.
5. Compare the results and discuss how conversation memory affects agent behavior.

### Optional Extension

Implement `compare_group_medians(feature)` and register it as an additional agent tool.

Test whether the agent selects the new function when asked to compare median values.

### Further Reading

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

---


## Limitations and responsible use

- These are **descriptive comparisons** across two dataset groups, not diagnosis or causal inference.
- A sample of 116 observations does not establish population-level medical conclusions.
- The cohort and controls may differ in ways not captured by the chosen features.
- No patient-specific prediction or medical advice is provided.
- Tool selection is probabilistic; tool calculations are deterministic and separately testable.

## Dataset citation and license

Patrício, M., Pereira, J., Crisóstomo, J., Matafome, P., Seiça, R., & Caramelo, F. (2018). *Breast Cancer Coimbra* [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C52P59. **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/

Framework references:
- https://docs.langchain.com/oss/python/langchain/agents
- https://docs.langchain.com/oss/python/integrations/chat/ollama
- https://docs.ollama.com/capabilities/tool-calling


