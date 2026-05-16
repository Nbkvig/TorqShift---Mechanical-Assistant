"""
ingest.py — TorqShift ingestion pipeline.
Hybrid extraction: pdfplumber for tables, PyMuPDF for prose.
Embeds chunks with text-embedding-3-small and upserts into ChromaDB.
"""

import os
import sys
import hashlib

import fitz  # pymupdf
import pdfplumber
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PDFS = {
    "./data/2017_Subaru_WRX.pdf": "WRX",
    "./data/2016_honda_odyssey.pdf": "Odyssey",
}
COLLECTION = "torqshift_manuals"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50


def check_files():
    for path in PDFS:
        if not os.path.exists(path):
            print(f"ERROR: Missing required PDF: {path}")
            sys.exit(1)


def chunk_text(text: str, chunk_size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Naive word-based chunking approximating token counts (1 token ≈ 0.75 words)."""
    words = text.split()
    # Convert token counts to approximate word counts
    size = int(chunk_size * 0.75)
    step = int((chunk_size - overlap) * 0.75)
    if step < 1:
        step = 1
    chunks = []
    for i in range(0, max(1, len(words) - size + 1), step):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk.strip())
    # Catch any trailing content not covered
    if not chunks and words:
        chunks.append(" ".join(words))
    return chunks


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def chunk_id(car: str, page: int, chunk_type: str, index: int, text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{car}_{page}_{chunk_type}_{index}_{h}"


def ingest_pdf(path: str, car: str, collection, client: OpenAI):
    total_chunks = 0
    plumber_pdf = pdfplumber.open(path)
    fitz_pdf = fitz.open(path)
    num_pages = len(fitz_pdf)

    for page_num in range(num_pages):
        page_index = page_num + 1  # 1-based
        plumber_page = plumber_pdf.pages[page_num]
        tables = plumber_page.extract_tables()

        chunks_with_meta = []

        if tables:
            for table in tables:
                if not table:
                    continue
                # Format as markdown table
                rows = [[str(cell) if cell is not None else "" for cell in row] for row in table]
                if not rows:
                    continue
                header = "| " + " | ".join(rows[0]) + " |"
                separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
                body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
                md_table = "\n".join(filter(None, [header, separator, body]))
                chunks_with_meta.append((md_table, "table"))
        else:
            fitz_page = fitz_pdf[page_num]
            text = fitz_page.get_text()
            if text.strip():
                for chunk in chunk_text(text):
                    chunks_with_meta.append((chunk, "prose"))

        if not chunks_with_meta:
            continue

        # Batch embed
        texts = [c[0] for c in chunks_with_meta]
        embeddings = embed(client, texts)

        ids, docs, metas, embeds = [], [], [], []
        for i, ((text, ctype), emb) in enumerate(zip(chunks_with_meta, embeddings)):
            cid = chunk_id(car, page_index, ctype, i, text)
            ids.append(cid)
            docs.append(text)
            metas.append({"car": car, "type": ctype, "page": page_index})
            embeds.append(emb)

        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
        total_chunks += len(ids)
        print(f"  [{car}] Page {page_index}/{num_pages} — {len(ids)} chunks")

    plumber_pdf.close()
    fitz_pdf.close()
    return total_chunks


def main():
    check_files()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(name=COLLECTION)

    grand_total = 0
    for path, car in PDFS.items():
        print(f"\nIngesting {path} as car={car} ...")
        n = ingest_pdf(path, car, collection, client)
        grand_total += n
        print(f"  → {n} chunks from {path}")

    print(f"\nIngested {grand_total:,} chunks from {len(PDFS)} documents")


if __name__ == "__main__":
    main()
