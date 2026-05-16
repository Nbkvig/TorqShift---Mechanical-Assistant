# TorqShift — Kiro Implementation Spec

## System Context & Purpose
TorqShift is a specialized AI Mechanical Assistant for DIY mechanics and automotive 
hobbyists. It transforms dense factory owner's manuals into an interactive, 
context-aware assistant that answers specific questions about text specifications, 
fluid capacities, torque values, and maintenance procedures.

The app targets two factory owner's manuals:
- 2017 Subaru WRX (tagged car="WRX")
- 2016 Honda Odyssey (tagged car="Odyssey")

The user can type a question OR upload a photo (e.g., an engine bay, fuse box, or 
part label). The app identifies the relevant vehicle subsystem from the image using 
GPT-4o-mini Vision, retrieves the most relevant manual chunks from ChromaDB, and 
synthesizes a grounded, cited answer using only the retrieved context.

## PDF Source Convention
Both owner's manual PDFs are already placed in ./data/:
  ./data/2017_Subaru_WRX.pdf
  ./data/2016_honda_odyssey.pdf

ingest.py must print a clear error and exit if either file is missing, rather than 
silently failing or crashing mid-run.

## Core Architecture & Technical Constraints

### 1. Tech Stack
- Python 3.11+
- Streamlit (UI)
- ChromaDB with PersistentClient (local vector store, stored in ./chroma_db/)
- PyMuPDF — pip package name is `pymupdf`, imported in code as `fitz`
- pdfplumber (table extraction)
- OpenAI: GPT-4o-mini (chat + vision), text-embedding-3-small (embeddings)
- python-dotenv (.env loading)

### 2. Zero-Cloud Database Boundary
The vector database must be entirely local using ChromaDB's PersistentClient 
persisted to ./chroma_db/. No Pinecone, Weaviate, Supabase, or any external 
database. The grader supplies only OPENAI_API_KEY in a .env file and nothing else. 
The app must start and run fully on that alone.

### 3. Hybrid Extraction Pipeline — ingest.py
Create a standalone script ingest.py that:

1. Iterates over both PDFs in ./data/ using the exact filenames:
   - ./data/2017_Subaru_WRX.pdf  →  car="WRX"
   - ./data/2016_honda_odyssey.pdf  →  car="Odyssey"

2. For each page, attempts pdfplumber table detection first:
   - If tables are detected, extract and format as Markdown table strings.
     Tag: {"car": "WRX"|"Odyssey", "type": "table", "page": int}
   - If no tables detected, extract raw text via PyMuPDF (fitz).
     Tag: {"car": "WRX"|"Odyssey", "type": "prose", "page": int}

3. Chunks prose text into segments of ~500 tokens with ~50 token overlap.

4. Embeds every chunk using text-embedding-3-small via the OpenAI API.

5. Upserts all embeddings and metadata into a ChromaDB collection named
   "torqshift_manuals" using PersistentClient pointed at ./chroma_db/.

6. Prints progress per page and a final summary:
   e.g. "Ingested 3,412 chunks from 2 documents"

7. Is fully idempotent — running it twice does not duplicate chunks.

### 4. Metadata Schema
Every chunk stored in ChromaDB must carry this exact metadata structure:
  {
    "car":  "WRX" | "Odyssey",
    "type": "prose" | "table",
    "page": int
  }
All ChromaDB retrieval queries must filter by "car" to scope results to the 
relevant vehicle.

## Functional Features & UI — app.py

### 1. Input Area
- st.chat_input for the user's question
- st.file_uploader accepting image types (jpg, jpeg, png) for optional photo upload
- st.selectbox for manual car selection: ["WRX", "Odyssey", "Auto-detect from image"]
  - If "Auto-detect from image" is selected and no image is uploaded, show
    st.warning and prompt the user to upload one.

### 2. Image Routing (Multimodal Path)
If an image is uploaded:
1. Send the image to GPT-4o-mini Vision with a prompt asking it to identify:
   - The vehicle (WRX or Odyssey) if determinable
   - The subsystem visible (e.g., "engine bay", "fuse box", "brake assembly")
2. Parse the response to extract car and subsystem context.
3. Confidence fallback: if GPT-4o-mini Vision cannot confidently identify the vehicle,
   display a st.selectbox for the user to manually select the car before proceeding.
   Do not dead-end the user — always give them a path forward.
4. Use the identified subsystem as additional context injected into the RAG 
   retrieval query.

### 3. RAG Retrieval
- Query ChromaDB using the user's question (and subsystem context if from image)
- Apply a metadata filter: {"car": identified_car}
- Retrieve top 5 most relevant chunks
- If fewer than 2 chunks are returned, display:
  st.warning("Not enough manual context found. Try rephrasing or selecting a 
  different vehicle.")

### 4. Answer Synthesis
Construct a system prompt that:
- Instructs GPT-4o-mini to answer using ONLY the provided manual chunks
- Explicitly tells it not to use outside knowledge or hallucinate specifications
- Injects the retrieved chunks as numbered context blocks
- Requests the answer include specific values (torque specs, fluid types,
  capacities, step numbers) when present in context

System prompt structure:
  "You are TorqShift, an AI assistant for DIY mechanics. Answer the user's
  question using ONLY the following excerpts from the official Owner's manual.
  Do not use outside knowledge. If the answer is not in the excerpts, say so.

  Context:
  [1] (WRX, page 142, table): ...
  [2] (WRX, page 143, prose): ...
  ..."

### 5. Output & Citations
- Display the synthesized answer using st.markdown
- Below the answer, render each source chunk as a st.caption:
    📄 Source: 2017 Subaru WRX Owner's Manual — Page 142 (table)
- Show a st.expander("View raw context chunks") containing the full text of
  each retrieved chunk for transparency

### 6. Loading States & Error Handling
- Wrap image analysis in st.spinner("Analyzing image...")
- Wrap retrieval + synthesis in st.spinner("Searching manual and generating answer...")
- On missing OPENAI_API_KEY: display st.error("OPENAI_API_KEY not found. 
  Please add it to your .env file.") and st.stop() — do not crash with a traceback
- On OpenAI API errors: catch exceptions and display st.error with the message
- On empty query submission: display st.warning and do not trigger retrieval

## Evaluation Pipeline — eval/

### eval/test_cases.json
A JSON array of exactly 10 labeled test cases. Each object must have:
  {
    "id": int,
    "query": str,
    "car": "WRX" | "Odyssey",
    "expected_keywords": [str],
    "notes": str
  }

Test cases must cover:
- At least 3 cases from each vehicle
- At least 2 cases requiring table data (torque specs, fluid capacities)
- At least 1 intentionally out-of-scope case where the app should say it
  doesn't know rather than hallucinate

### eval/run_eval.py
A standalone script that:
1. Loads test_cases.json
2. For each test case, calls the RAG pipeline directly (not via Streamlit UI)
   using the query and car filter
3. Scores each case:
   score = keywords found in response (case-insensitive) / total expected keywords
4. Prints a per-case result table:
   ID | Query (truncated) | Car | Score | Matched Keywords
5. Prints aggregate mean score at the end
6. Writes results to eval/results.json

Run with: python eval/run_eval.py

## Repo Structure
torqshift/
├── app.py
├── ingest.py
├── requirements.txt
├── .env.example            ← contains only: OPENAI_API_KEY=your_key_here
├── .gitignore              ← must exclude .env, chroma_db/, data/
├── data/                   ← gitignored, PDFs already present
├── chroma_db/              ← auto-created by ingest.py, gitignored
└── eval/
    ├── test_cases.json
    ├── run_eval.py
    └── results.json

## requirements.txt
Pin all dependencies to exact versions. Must include:
  streamlit
  chromadb
  openai
  python-dotenv
  pymupdf
  pdfplumber
  pillow

## .gitignore
Must exclude:
  .env
  chroma_db/
  data/
  __pycache__/
  *.pyc

## Code Quality Requirements
- No secrets or API keys hardcoded anywhere
- All OpenAI calls must use the openai Python SDK v1.x client style
- ChromaDB interactions must use PersistentClient
- app.py must be modular with separate functions:
    image_analysis()
    retrieve_chunks()
    synthesize_answer()
- ingest.py must be runnable standalone: python ingest.py
- eval/run_eval.py must be runnable standalone: python eval/run_eval.py