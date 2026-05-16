"""
app.py — TorqShift Streamlit UI.
Multi-modal RAG: text questions + optional image upload.
"""

import os
import base64
from io import BytesIO

import streamlit as st
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

COLLECTION = "torqshift_manuals"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 8

CAR_LABELS = {
    "WRX": "2017 Subaru WRX Owner's Manual",
    "Odyssey": "2016 Honda Odyssey Owner's Manual",
}


@st.cache_resource
def get_clients():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
        st.stop()
    openai_client = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(name=COLLECTION)
    return openai_client, collection


def image_analysis(client: OpenAI, image_bytes: bytes, mime: str) -> dict:
    """Send image to GPT-4o-mini Vision; return {car, subsystem}."""
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime};base64,{b64}"
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at this automotive image. "
                            "1) Identify the vehicle if possible — is it a 2017 Subaru WRX or a 2016 Honda Odyssey? "
                            "Reply with exactly 'WRX', 'Odyssey', or 'unknown'. "
                            "2) Identify the visible subsystem (e.g. engine bay, fuse box, brake assembly, transmission). "
                            "Reply in this exact format:\n"
                            "CAR: <WRX|Odyssey|unknown>\n"
                            "SUBSYSTEM: <subsystem description>"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=100,
    )
    text = response.choices[0].message.content or ""
    car = "unknown"
    subsystem = ""
    for line in text.splitlines():
        if line.startswith("CAR:"):
            val = line.split(":", 1)[1].strip()
            if val in ("WRX", "Odyssey"):
                car = val
        elif line.startswith("SUBSYSTEM:"):
            subsystem = line.split(":", 1)[1].strip()
    return {"car": car, "subsystem": subsystem}


def retrieve_chunks(client: OpenAI, collection, query: str, car: str) -> list[dict]:
    """Embed query, filter by car, return top-K chunks."""
    emb_response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    query_emb = emb_response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K,
        where={"car": car},
    )
    chunks = []
    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({"text": doc, "car": meta["car"], "type": meta["type"], "page": meta["page"]})
    return chunks


def synthesize_answer(client: OpenAI, query: str, chunks: list[dict]) -> str:
    """Build context prompt and call GPT-4o-mini."""
    context_blocks = "\n".join(
        f"[{i+1}] ({c['car']}, page {c['page']}, {c['type']}): {c['text']}"
        for i, c in enumerate(chunks)
    )
    system_prompt = (
        "You are TorqShift, an AI assistant for DIY mechanics. Answer the user's "
        "question using ONLY the following excerpts from the official owner's manual. "
        "Do not use outside knowledge. If the answer is not in the excerpts, say so. "
        "When the answer involves specific information, extract exact numeric values, units, "
        "and part names directly from the context.\n\n"
        f"Context:\n{context_blocks}"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content or ""


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="TorqShift", page_icon="🔧")
st.title("🔧 TorqShift — AI Mechanical Assistant")

openai_client, collection = get_clients()

car_option = st.selectbox("Select vehicle manual", ["WRX", "Odyssey", "Auto-detect from image"])
uploaded_image = st.file_uploader("Upload a photo (optional)", type=["jpg", "jpeg", "png"])
query = st.chat_input("Ask a question about your vehicle...")

if query:
    # Prompt injection guard
    injection_keywords = [
        "ignore previous", "ignore instructions", "disregard",
        "forget instructions", "you are now", "act as",
        "jailbreak", "pretend you are"
    ]
    if any(kw in query.lower() for kw in injection_keywords):
        st.error("Invalid input detected. Please ask a question about your vehicle.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    resolved_car = None
    subsystem_context = ""

    # ── Image path ────────────────────────────────────────────────────────────
    if uploaded_image:
        with st.spinner("Analyzing image..."):
            try:
                image_bytes = uploaded_image.read()
                mime = f"image/{uploaded_image.type.split('/')[-1]}"
                result = image_analysis(openai_client, image_bytes, mime)
                subsystem_context = result["subsystem"]
                if result["car"] != "unknown":
                    resolved_car = result["car"]
                    st.info(f"Image identified: **{resolved_car}** — {subsystem_context}")
                else:
                    st.warning("Could not identify the vehicle from the image.")
            except Exception as e:
                st.error(f"Image analysis error: {e}")

    # ── Car resolution ────────────────────────────────────────────────────────
    if resolved_car is None:
        if car_option == "Auto-detect from image":
            if not uploaded_image:
                st.warning("Please upload an image to use Auto-detect, or select a vehicle manually.")
                st.stop()
            else:
                # Image was uploaded but car not identified — let user pick
                resolved_car = st.selectbox("Could not auto-detect vehicle. Please select:", ["WRX", "Odyssey"])
        else:
            resolved_car = car_option

    # ── RAG retrieval + synthesis ─────────────────────────────────────────────
    full_query = f"{query} {subsystem_context}".strip() if subsystem_context else query

    with st.spinner("Searching manual and generating answer..."):
        try:
            chunks = retrieve_chunks(openai_client, collection, full_query, resolved_car)
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.stop()

        if len(chunks) < 2:
            st.warning(
                "Not enough manual context found. Try rephrasing or selecting a different vehicle."
            )
            st.stop()

        try:
            answer = synthesize_answer(openai_client, query, chunks)
        except Exception as e:
            st.error(f"Answer generation error: {e}")
            st.stop()

    # ── Output ────────────────────────────────────────────────────────────────
    st.markdown(answer)

    st.markdown("---")
    for chunk in chunks:
        label = CAR_LABELS.get(chunk["car"], chunk["car"])
        st.caption(f"📄 Source: {label} — Page {chunk['page']} ({chunk['type']})")

    with st.expander("View raw context chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**[{i+1}] {chunk['car']} — Page {chunk['page']} ({chunk['type']})**")
            st.text(chunk["text"])
