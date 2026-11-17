"""
GUI for the ID fraud agent (Streamlit, runs locally), with follow-up chat and memory.

    streamlit run app.py

DJ's synchronous flow, one click per document:
  STEP 1  upload the document                 (this page)
  STEP 2  extract features right away         (fraud_tool.features_from_path)
  STEP 3  hand off to the agent orchestrator  (LangChain agent from agent.py)
  STEP 4  the LLM calls the classifier tool   (agent.classify_document)
  STEP 5  the tool runs our trained model     (fraud_tool.classify_features)
  STEP 6  decision + explanation shown here   (APPROVE / HUMAN_REVIEW / REJECT)
Then ask follow-up questions in the chat box: the agent remembers this conversation.
"""
import os
import tempfile
import time
import uuid

import streamlit as st

from agent import FEATURE_STORE, ask, build_agent, classify_document, run_by_id
from fraud_tool import classify_features, features_from_path, load_bundle

st.set_page_config(page_title="ID Document Verification", page_icon="🪪", layout="centered")

DECISION_STYLE = {
    "APPROVE": ("success", "✅ Approved: the document looks genuine"),
    "HUMAN_REVIEW": ("warning", "🟡 Sent to human review: the model isn't confident enough"),
    "REJECT": ("error", "⛔ Rejected: the document is likely fake"),
}

ss = st.session_state
ss.setdefault("thread_id", uuid.uuid4().hex[:8])   # one conversation = one memory thread
ss.setdefault("chat", [])                          # what the page shows
ss.setdefault("upload_key", 0)                     # bump to clear the uploader


@st.cache_resource
def get_agent(model_name):
    # the agent keeps a separate memory per thread_id, so each browser session has its own conversation
    return build_agent(model_name, tool_fn=classify_document)


def save_upload(uploaded):
    """STEP 1: save the upload to a temp file. PDFs: render the first page as an image."""
    suffix = os.path.splitext(uploaded.name)[1].lower()
    data = uploaded.getvalue()
    if suffix == ".pdf":
        try:
            import fitz  # pymupdf
        except ImportError:
            st.error("PDF support needs `pip install pymupdf`. Upload a JPG or PNG instead.")
            st.stop()
        page = fitz.open(stream=data, filetype="pdf")[0]
        data, suffix = page.get_pixmap(dpi=200).tobytes("png"), ".png"
    path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex[:8]}{suffix}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def show_decision(result):
    if "error" in result:
        st.error(result["error"])
        return
    kind, text = DECISION_STYLE[result["decision"]]
    getattr(st, kind)(text)
    c1, c2 = st.columns(2)
    c1.metric("Probability fake", f"{result['probability_fake']:.0%}")
    c2.metric("Decision", result["decision"].replace("_", " ").title())


def show_message(msg):
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=320)
        if msg.get("result"):
            show_decision(msg["result"])
        if msg.get("content"):
            st.markdown(msg["content"].replace("\n", "  \n"))  # keep the LLM's line breaks
        if msg.get("steps"):
            with st.expander("How the agent got this"):
                for s in msg["steps"]:
                    st.write(s)
                if msg.get("result"):
                    st.json(msg["result"])


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle("Use the LLM agent", value=True,
                        help="Off = call the classifier directly (no explanations, no follow-up chat).")
    model_name = st.text_input("Ollama model", value="llama3.1", disabled=not use_llm)
    try:
        b = load_bundle()
        st.caption(f"Classifier: **{b['model_name']}**")
        st.caption(f"Approve if p(fake) < {b['t_approve']:.3f}  \nReject if p(fake) > {b['t_reject']:.3f}  \n"
                   "Human review in between")
    except FileNotFoundError:
        st.error("fraud_model_v2.joblib not found. Put it next to app.py (see README step 1).")
        st.stop()
    st.divider()
    if st.button("🗑️ New conversation", width="stretch"):
        ss.thread_id, ss.chat = uuid.uuid4().hex[:8], []   # new thread = fresh memory
        st.rerun()

# ---------------- main page ----------------
st.title("🪪 ID Document Verification")
st.write("Upload a passport, driver's license or national ID. Then ask follow-up questions below: "
         "the agent remembers this conversation.")

with st.container(border=True):
    uploaded = st.file_uploader("Document image", type=["jpg", "jpeg", "png", "pdf"],
                                key=f"upload_{ss.upload_key}")
    verify = st.button("Verify document", type="primary", width="stretch", disabled=uploaded is None)

for msg in ss.chat:
    show_message(msg)

# ---------- a new document: DJ's synchronous flow ----------
if verify and uploaded is not None:
    path = save_upload(uploaded)
    ss.chat.append({"role": "user", "content": f"Uploaded **{uploaded.name}**", "image": path})
    steps = ["**Step 1** Upload received"]
    with st.spinner("Verifying document..."):
        t0 = time.time()
        feats, err = features_from_path(path)          # STEP 2: features first, synchronously
        if err:
            ss.chat.append({"role": "assistant", "content": err})
            ss.upload_key += 1
            st.rerun()
        steps.append(f"**Step 2** Extracted {feats.shape[1]} features ({time.time() - t0:.1f}s)")
        doc_id = uuid.uuid4().hex[:8]
        FEATURE_STORE[doc_id] = (feats, uploaded.name)  # kept so follow-ups can re-check it

        answer, result = None, None
        if use_llm:                                      # STEPS 3-6 through the agent
            t0 = time.time()
            try:
                answer, result = run_by_id(get_agent(model_name), doc_id, uploaded.name, ss.thread_id)
                steps.append(f"**Step 3** Agent orchestrator ({model_name}) received the document")
                steps.append(f"**Step 4–5** LLM called the classifier tool ({time.time() - t0:.1f}s)"
                             if result else "⚠️ The LLM answered without calling the classifier tool")
            except Exception as e:  # Ollama not running, model not pulled, etc.
                steps.append(f"⚠️ LLM unavailable ({type(e).__name__}); used the classifier directly")
        if result is None:                               # fallback / LLM off: steps 5-6 directly
            result = classify_features(feats, uploaded.name)
            steps.append("**Step 5** Classifier called directly")
        steps.append("**Step 6** Decision made")
    ss.chat.append({"role": "assistant", "content": answer, "result": result, "steps": steps})
    ss.upload_key += 1                                   # clear the uploader for the next document
    st.rerun()

# ---------- follow-up questions: same thread_id = the agent remembers ----------
question = st.chat_input("Ask a follow-up, e.g. 'Why was it flagged?' or 'Which one was more suspicious?'",
                         disabled=not use_llm)
if question:
    ss.chat.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        try:
            answer, results = ask(get_agent(model_name), question, ss.thread_id)
            msg = {"role": "assistant", "content": answer}
            if results:                                  # the agent re-checked a document
                msg["result"] = results[-1]
        except Exception as e:
            msg = {"role": "assistant", "content": f"⚠️ The LLM isn't reachable ({type(e).__name__}). "
                                                   "Open the Ollama app and try again."}
    ss.chat.append(msg)
    st.rerun()
