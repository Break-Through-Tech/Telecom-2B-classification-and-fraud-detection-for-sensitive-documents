# AI Fraud-Detection Agent (Kasish Jain)

**Author:** Kasish Jain (UT Dallas) · Break Through Tech AI Studio, Fall 2026 · Verizon, Telecom-2B
**Milestone:** October, Autonomous AI Agent (stretch goal in the challenge overview)

## What I built (my contribution)

I built the autonomous AI agent around the team's binary classifier, following the flow DJ described on the Nov 10 call:

```
GUI upload -> extract features -> agent (LangChain + local Ollama LLM) -> tool call -> our classifier -> APPROVE / HUMAN_REVIEW / REJECT
```

| Piece | File / location | What I did |
|---|---|---|
| Decision thresholds (notebook section 9) | team notebook, section 9 | Turned the binary classifier into a three-outcome rule: **APPROVE** if p(fake) < t_approve, **REJECT** if p(fake) > t_reject, otherwise **HUMAN_REVIEW**. Thresholds come from 5-fold out-of-fold predictions on the training set: at most 1% of genuine documents auto-rejected (`MAX_FPR = 0.01`) and at most 2% of fakes auto-approved (`MAX_MISS = 0.02`). I kept `MAX_MISS` strict because DJ said false negatives are the critical error. |
| Model export (section 9) | team notebook, section 9 | Saved the LightGBM model, its name, both thresholds and the feature count as one `fraud_model_v2.joblib` bundle, and printed the exact package versions so the laptop can match Colab. Added `infer(path)` for single-document prediction. |
| Feature extraction as a stand-alone module | `fraud_features.py` | A copy of the notebook's section 5 `extract_features(path)`, so the agent computes the same 295 forensic features outside Colab. |
| Inference tool | `fraud_tool.py` | Loads the saved bundle, extracts features, predicts p(fake) and applies the thresholds. Returns JSON (file, decision, probability, thresholds, model). Runs from the command line too. |
| The agent | `agent.py` | LangChain `create_agent` with a local Ollama model (llama3.1). A system prompt gives the agent its role. It calls the classifier as a tool. The final decision always comes from the classifier, never from the LLM. Has `classify_document` (GUI, features already extracted) and `check_document` (command line, extraction inside the tool). |
| Conversation memory | `agent.py` | Added a LangGraph `InMemorySaver` checkpointer, one thread per conversation, so follow-up questions work ("Why was it flagged?", "Which one was more suspicious?", "How many documents have I checked?"). |
| GUI | `app.py` | Streamlit page: upload a JPG, PNG or PDF, click Verify, see each step, the colored decision, the fake probability and the agent's explanation. A chat box handles follow-up questions, and a sidebar button starts a new conversation. Falls back to the classifier alone if Ollama is not running. |
| Colab agent demo (section 11) | team notebook, section 11 | Installs Ollama and LangChain in Colab, builds the same agent and runs it on one fake and one genuine test document. |

## Results

- Verified end to end with a real local LLM (Ollama, llama3.1) in Colab: a fake driver's license returned **REJECT** (p(fake) 0.998) and a genuine Greek passport returned **HUMAN_REVIEW** (p(fake) 0.064).
- Also ran the agent and the Streamlit GUI on my own laptop.
- Open issue: with the current thresholds (approve below 0.028, reject above 0.753), 32 of 60 genuine test documents go to human review. This should shrink as the classifier improves (better features, more data, a stronger driver's-license model). I did not loosen `MAX_MISS` to reduce it.

## Design decisions

- **The classifier decides, the LLM explains.** The LLM only calls the tool and writes the explanation from the tool's output, so it cannot approve a fake by itself.
- **Local LLM (Ollama).** Free, and customer identity documents are never sent to a third-party API.
- **Swappable classifier.** If the team improves the model, rerun notebook section 9, replace `fraud_model_v2.joblib`, and copy any changed `extract_features` code into `fraud_features.py`. The agent and GUI do not change.

## Not done yet

- Agent answering analytics or ad-hoc questions over the document collection (optional stretch goal in the challenge overview, for example "how many are from California?"). My plan is to save every check to a table and add a query tool.
- Multi-class document-type detection (passport vs. driver's license vs. national ID).

---

# Setup

What you are building, in DJ's terms:

```
GUI upload  ->  extract features  ->  agent (LangChain + Ollama LLM)  ->  tool call  ->  our classifier  ->  APPROVE / HUMAN_REVIEW / REJECT
```
One synchronous call, as DJ described on Nov 10.

Files in this folder:

| File | What it does |
|---|---|
| `fraud_features.py` | The exact feature extraction from the notebook (section 5), so the agent computes the same 295 features |
| `fraud_tool.py` | The **inference tool**: image path in, decision + probability out |
| `agent.py` | The agent: DJ's "agent basic" example with `get_weather` replaced by our classifier tool |
| `app.py` | The Streamlit GUI: upload a document, see the decision and the agent's explanation |
| `requirements.txt` | Python packages |
| `fraud_model_v2.joblib` | **You create this in Colab (step 1)** |

---

## Step 1: Save the model in Colab

1. Open `Telecom_2B_agent_prep.ipynb` in Colab (the team notebook plus a new **section 9: Agent prep**).
2. Run all cells from the top through section 9. Section 10 (Demo) asks for an upload, so stop before it.
3. Section 9 does three things:
   - Picks thresholds so at most 1% of genuine documents are auto-rejected, and shows a before/after table (old 0.5 cutoff vs. approve / review / reject).
   - Saves and downloads `fraud_model_v2.joblib`.
   - Prints a `pip install ...==version` line. **Copy that line**; you need it in step 3.
4. Download 3 to 4 test images to try later: in the Colab file browser, open `/content/final_sample` and download a couple of `_genuine_` and `_fake_` files.

## Step 2: Install Ollama (the free local LLM)

1. Download and install from **https://ollama.com/download** (Mac or Windows).
2. In a terminal, download a model that supports tool calling:
   ```
   ollama pull llama3.1
   ```
   This is about 5 GB. If your laptop is low on memory, use `ollama pull qwen2.5:3b` instead and add `--model qwen2.5:3b` when running the agent.
3. Check it works: `ollama run llama3.1 "hello"`, then type `/bye` to exit.

## Step 3: Set up Python on your laptop

From inside this folder:

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install <the line copied from Colab in step 1>
```

The second `pip install` matches your laptop's scikit-learn / LightGBM versions to Colab's, so the saved model loads correctly.

Put `fraud_model_v2.joblib` in this folder.

## Step 4: Test the tool alone (no LLM yet)

```
python fraud_tool.py path/to/some_genuine.png path/to/some_fake.jpg
```

You should see JSON like:

```
{"file": "...", "decision": "APPROVE", "probability_fake": 0.031, ...}
```

If this works, the model and features are correct. Fix any error here before adding the agent.

## Step 5: Run the agent

```
python agent.py path/to/some_fake.jpg
```

The LLM reads the request, calls `check_document`, and replies:

```
Decision: REJECT
Probability fake: 0.94
Reason: The classifier's fake probability is above the rejection threshold of 0.81, ...
```

The first run is slow while Ollama loads the model.

## Step 6: Run the GUI (Streamlit)

This is DJ's synchronous flow in one click: upload → extract features → LangChain agent → classifier tool → decision.

1. Make sure Ollama is running (open the Ollama app, or run `ollama serve`).
2. From this folder, with the venv active:
   ```
   streamlit run app.py
   ```
3. Your browser opens at `http://localhost:8501`. Upload a JPG, PNG or PDF and click **Verify document**.

The page shows each step (upload, features, agent, tool call, decision), the decision in color, the fake probability and the agent's explanation. **Technical details** shows the raw classifier output.

- **Follow-up questions (memory):** after checking documents, type in the chat box at the bottom, e.g. "Why was it flagged?", "Which one was more suspicious?", "How many documents have I checked?". The agent remembers the whole conversation. **New conversation** in the sidebar clears its memory.
- **Sidebar:** switch the Ollama model, or turn the LLM off to call the classifier directly.
- **If Ollama isn't running,** the page says so and still shows the classifier's decision, so a demo never breaks completely.

## Chat from the command line (also has memory)

```
python agent.py --chat
```

Paste an image path to check it, or ask a follow-up question. Type `quit` to stop.

## Step 7 (optional): Watch a folder

To check every new file dropped into a folder, from the command line:

```
mkdir uploads
python agent.py --watch uploads
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv (step 3) and rerun `pip install -r requirements.txt` |
| Error loading the `.joblib` file | Versions differ from Colab. Run the exact `pip install ...==version` line printed in step 1 |
| `Expected 295 features, got ...` | `fraud_features.py` is out of date. Copy the new `extract_features` code from notebook section 5 |
| `Connection refused` | Ollama isn't running. Open the Ollama app, or run `ollama serve` |
| The agent answers without calling the tool | Try `--model qwen2.5:7b`, which is reliable at tool calling |

## What's next

- **LLM explanation:** already in the system prompt; it explains the decision using only the tool's output, so it can't invent details about the image.
- **PDF uploads:** supported in the GUI; the first page is converted to an image.
- **When the team updates the classifier:** rerun notebook section 9, replace `fraud_model_v2.joblib`, and copy any changed `extract_features` code into `fraud_features.py`. The agent and GUI don't change.
