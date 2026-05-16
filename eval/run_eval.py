"""
eval/run_eval.py — TorqShift evaluation pipeline.
Runs each test case through the RAG pipeline and scores keyword matches.
"""

import json
import os
import sys
import datetime

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

COLLECTION = "torqshift_manuals"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def retrieve_chunks(client, collection, query: str, car: str) -> list[dict]:
    emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    results = collection.query(query_embeddings=[emb], n_results=TOP_K, where={"car": car})
    chunks = []
    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({"text": doc, "car": meta["car"], "type": meta["type"], "page": meta["page"]})
    return chunks


def synthesize_answer(client, query: str, chunks: list[dict]) -> str:
    context = "\n".join(
        f"[{i+1}] ({c['car']}, page {c['page']}, {c['type']}): {c['text']}"
        for i, c in enumerate(chunks)
    )
    system = (
        "You are TorqShift, an AI assistant for DIY mechanics. Answer the user's "
        "question using ONLY the following excerpts from the official service manual. "
        "Do not use outside knowledge. If the answer is not in the excerpts, say so.\n\n"
        f"Context:\n{context}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
    )
    return resp.choices[0].message.content or ""


def score_response(response: str, keywords: list[str]) -> tuple[float, list[str]]:
    lower = response.lower()
    matched = [kw for kw in keywords if kw.lower() in lower]
    return len(matched) / len(keywords) if keywords else 0.0, matched


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(name=COLLECTION)

    with open(TEST_CASES_PATH) as f:
        test_cases = json.load(f)

    results = []
    scores = []

    header = f"{'ID':<4} {'Query':<40} {'Car':<8} {'Score':<7} Matched Keywords"
    print(header)
    print("-" * len(header))

    for tc in test_cases:
        query = tc["query"]
        car = tc["car"]
        keywords = tc["expected_keywords"]

        chunks = retrieve_chunks(client, collection, query, car)
        if chunks:
            answer = synthesize_answer(client, query, chunks)
        else:
            answer = "No context found in manual."

        score, matched = score_response(answer, keywords)
        scores.append(score)

        truncated_query = query[:38] + ".." if len(query) > 40 else query
        print(f"{tc['id']:<4} {truncated_query:<40} {car:<8} {score:<7.2f} {matched}")

        results.append({
            "id": tc["id"],
            "query": query,
            "car": car,
            "score": round(score, 4),
            "matched_keywords": matched,
            "response": answer,
        })

    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\nMean score: {mean_score:.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"mean_score": round(mean_score, 4), "results": results}, f, indent=2)
    print(f"Results written to {RESULTS_PATH}")


    # Timestamped Results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(os.path.dirname(__file__), f"results_{timestamp}.json")
    with open(backup_path, "w") as f:
        json.dump({"mean_score": round(mean_score, 4), "results": results}, f, indent=2)
    print(f"Backup written to {backup_path}")


if __name__ == "__main__":
    main()
