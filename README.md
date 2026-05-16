# TorqShift---Mechanical-Assistant
    TorqShift, is an AI-powered web app designed for diy mechanics or car enthusiasts that may need fast and specific answers from their vehicle's owner's manual. Whether your in the middle of working on your car and can't remember the exact specs you need, TorqShift lets you ask a question and pull answers directly from official manuals (Currently only 2017 WRX and 2016 Odyssey). This allows you to get the exact answers you need without having to find your manual or going to many pages just to find what you need.


## Prerequisites

- Python 3.11+
- An OpenAI API key with billing enabled

---

## Setup

### 1. Clone the repo
```bash
git clone 
cd TorqShift---Mechanical-Assistant
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
```
Open `.env` and replace `your_key_here` with your actual OpenAI API key.

### 5. Run the app
The vector database is pre-built and included in the repo so no ingestion 
step is required.

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Example Queries

**2017 Subaru WRX:**
- "What is the engine oil capacity for the 2017 WRX?"
- "What type of coolant does the 2017 WRX use?"
- "What is the recommended oil viscosity for the WRX?"

**2016 Honda Odyssey:**
- "What brake fluid does the Honda Odyssey require?"
- "How do I reset the maintenance minder on the Odyssey?"
- "What is the tire pressure specification for the 2016 Odyssey?"

**Image upload:**
- Select a photo from the internet of a 2017 WRX or 2016 Odyssey, upload a photo of an engine bay or 
  component, and type your question.

---

## Running the Evaluation

```bash
python eval/run_eval.py
```

Runs 10 labeled test cases through the RAG pipeline, scores each using 
LLM-as-judge, prints a per-case result table, and writes `eval/results.json`.

---

## Project Structure

```
TorqShift/
├── app.py                  # Streamlit frontend
├── ingest.py               # PDF ingestion script (pre-run, not required)
├── requirements.txt        # Pinned dependencies
├── .env.example            # API key template
├── .gitignore
├── chroma_db/              # Pre-built vector database (included)
└── eval/
    ├── test_cases.json     # 10 labeled test cases
    ├── run_eval.py         # Evaluation script
    └── results.json        # Output from last eval run