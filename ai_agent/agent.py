"""
ID fraud-detection agent (LangChain + a local Ollama model), with conversation memory.

DJ's flow (Nov 10 call: one synchronous call) and where each step is:
  STEP 1  user uploads a document in the GUI     -> app.py (Streamlit page)
  STEP 2  extract features right away            -> app.py calls fraud_tool.features_from_path()
  STEP 3  agent orchestrator gets the document   -> build_agent()  (LangChain create_agent)
  STEP 4  the LLM does tool calling              -> classify_document()  (GUI)  /  check_document()  (command line)
  STEP 5  tool invokes our trained classifier    -> fraud_tool.classify_features()
  STEP 6  genuine / fake decision                -> fraud_tool.decide()  (approve / review / reject)
DJ also said feature extraction can live inside the tool instead: that is check_document() below.

MEMORY: the agent keeps the whole conversation per thread_id (LangGraph checkpointer), so
follow-up questions work: "why was it flagged?", "what was the probability?",
"which of the two was more suspicious?", "how many documents have I checked?".

    python agent.py --chat                        # chat with memory (paste image paths or ask questions)
    python agent.py path/to/document.jpg          # check one document
    python agent.py doc.jpg --model qwen2.5:7b    # use a different Ollama model
    streamlit run app.py                          # the GUI (also has follow-up chat)
"""
import argparse
import json
import os
import time

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from fraud_tool import classify_features, infer

# DJ: the system prompt gives the agent its "role, goal or mission"
SYSTEM_PROMPT = """You are an identity-document verification agent in a telecom customer-onboarding flow.

When the user gives you a NEW document (a file path or a document_id), call your verification tool exactly
once for it. Never guess a result. After the tool call, reply in this format:
Decision: <APPROVE, HUMAN_REVIEW or REJECT, copied exactly from the tool>
Probability fake: <number from the tool>
Reason: <one or two sentences based only on the tool result and its thresholds>

For follow-up questions (for example "why?", "what was the probability?", "compare the two documents",
"how many have I checked?"), answer from the earlier tool results in this conversation. Do not call the
tool again unless the user gives a new document or explicitly asks to re-check one.

Rules: you cannot see the images, so never describe their contents. Only use numbers that appear in tool
results. A probability below the approve threshold is low; above the reject threshold is high; in between
means it is borderline and a person should check it. If the tool returns an error, ask the customer to
re-upload a clear JPG or PNG photo of the document."""

# STEP 2 -> STEP 5 hand-off for the GUI: the app extracts features first and stores them here,
# so the LLM only passes a short document id instead of 295 numbers.
FEATURE_STORE = {}


# STEP 4 -> STEP 5 (GUI version): features were already extracted by the app.
@tool
def classify_document(document_id: str) -> str:
    """Classify an uploaded identity document (passport, driver's license or national ID) as genuine
    or fake. Its features were already extracted. Input: the document_id. Returns JSON with the
    decision (APPROVE, HUMAN_REVIEW or REJECT) and the probability that the document is fake."""
    item = FEATURE_STORE.get(document_id)
    if item is None:
        return json.dumps({"document_id": document_id, "error": "Unknown document_id."})
    feats, name = item
    return json.dumps(classify_features(feats, name))


# STEP 4 -> STEP 5 (command-line version): DJ's other option, feature extraction inside the tool.
@tool
def check_document(path: str) -> str:
    """Check whether an uploaded identity document image (passport, driver's license or national ID)
    is genuine or fake. Input: the file path of the image. Returns JSON with the decision
    (APPROVE, HUMAN_REVIEW or REJECT) and the probability that the document is fake."""
    return json.dumps(infer(path))


# STEP 3: the agent orchestrator. Same as DJ's example: get a reference to the LLM, then pass it
# to create_agent with our tool. Ollama instead of Gemini (free, runs locally).
# The checkpointer is the agent's MEMORY: it stores each conversation (thread_id) between calls.
def build_agent(model_name, tool_fn=check_document):
    llm = ChatOllama(model=model_name, temperature=0)  # temperature 0 = same answer every time
    return create_agent(model=llm, tools=[tool_fn], system_prompt=SYSTEM_PROMPT,
                        checkpointer=InMemorySaver())


def ask(agent, text, thread_id="default"):
    """Send one message in a conversation. Same thread_id = the agent remembers earlier messages.
    Returns (agent's answer, list of classifier results produced in THIS turn)."""
    result = agent.invoke({"messages": [("human", text)]},
                          config={"configurable": {"thread_id": thread_id}})
    msgs = result["messages"]
    last_human = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    tool_results = []
    for m in msgs[last_human + 1:]:
        if isinstance(m, ToolMessage):
            try:
                tool_results.append(json.loads(m.content))
            except (TypeError, ValueError):
                pass
    return msgs[-1].content, tool_results


# STEPS 3-6 for one document, by file path (command line).
def run(agent, path, thread_id="default"):
    return ask(agent, f"A customer uploaded this document: {path}. Verify it.", thread_id)[0]


# STEPS 3-6 for one document whose features the GUI already extracted (synchronous call).
def run_by_id(agent, document_id, name="document", thread_id="default"):
    answer, results = ask(agent, f"A customer uploaded a new document '{name}' with document_id "
                                 f"{document_id}. Verify it.", thread_id)
    return answer, (results[-1] if results else None)


def chat(agent):
    """Command-line chat with memory. Paste an image path to check it, or ask a follow-up question."""
    print("Chat with the verification agent (it remembers this conversation).")
    print("Paste an image path to check it, or ask a question. Type 'quit' to stop.\n")
    while True:
        text = input("You: ").strip().strip('"')
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        if os.path.isfile(text):   # a bare path = "please check this document"
            text = f"A customer uploaded this document: {text}. Verify it."
        print("Agent:", ask(agent, text, thread_id="cli")[0], "\n")


# Optional: stand-in for an upload event queue (DJ: only needed at large scale).
def watch(agent, folder, interval=2.0):
    print(f"Watching {folder} for uploads (Ctrl+C to stop)...")
    seen = set(os.listdir(folder))
    while True:
        for name in sorted(set(os.listdir(folder)) - seen):
            seen.add(name)
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"\n=== New upload: {name} ===")
                print(run(agent, os.path.join(folder, name), thread_id="watch"))
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="document image(s) to check")
    ap.add_argument("--chat", action="store_true", help="chat with memory")
    ap.add_argument("--watch", help="folder to watch for new uploads")
    ap.add_argument("--model", default="llama3.1", help="Ollama model that supports tool calling")
    args = ap.parse_args()

    agent = build_agent(args.model)
    for p in args.paths:
        print(f"\n=== {p} ===")
        print(run(agent, p))
    if args.chat:
        chat(agent)
    elif args.watch:
        watch(agent, args.watch)
    elif not args.paths:
        ap.print_help()
