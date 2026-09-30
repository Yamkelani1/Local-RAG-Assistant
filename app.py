import json
import os
import sqlite3

from openai import OpenAI
from pypdf import PdfReader
import streamlit as st

DB_PATH = "research.db"

# Cloud Model Identifiers (OpenRouter / OpenAI compatible)
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "google/gemma-2-9b-it"  # Or "gpt-4o-mini", "mistralai/mistral-7b-instruct", etc.

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
DEFAULT_QUESTION = "What helps distributed teams stay productive?"

# Initialize API Client using Streamlit Secrets or Environment Variable
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
base_url = st.secrets.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)


def initialize_database():
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                                                  id TEXT PRIMARY KEY,
                                                  source TEXT NOT NULL,
                                                  page INTEGER NOT NULL,
                                                  chunk_index INTEGER NOT NULL,
                                                  content TEXT NOT NULL,
                                                  embedding TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def extract_documents(uploaded_files):
    pages = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        source = uploaded_file.name
        if source.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"source": source, "page": page_number, "text": text})
        else:
            text = uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
            if text:
                pages.append({"source": source, "page": 1, "text": text})
    return pages


def chunk_pages(pages):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append({
                    "id": f"{page['source']}|{page['page']}|{chunk_index}",
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "content": content,
                })
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
            chunk_index += 1
    return chunks


def literal_phrase_search(chunks, question):
    normalized_question = question.strip().lower()
    if not normalized_question:
        return []
    return [chunk for chunk in chunks if normalized_question in chunk["content"].lower()]


def get_cloud_embeddings(texts):
    """Generates embeddings via OpenRouter/OpenAI API."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [data.embedding for data in response.data]


def store_chunks(chunks):
    texts = [chunk["content"] for chunk in chunks]
    embeddings = get_cloud_embeddings(texts)
    rows = [
        (
            chunk["id"],
            chunk["source"],
            chunk["page"],
            chunk["chunk_index"],
            chunk["content"],
            json.dumps(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    sources = sorted({chunk["source"] for chunk in chunks})
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.executemany(
            "DELETE FROM chunks WHERE source = ?",
            [(source,) for source in sources],
        )
        connection.executemany(
            "INSERT INTO chunks (id, source, page, chunk_index, content, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def count_stored_chunks():
    connection = sqlite3.connect(DB_PATH)
    try:
        return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        connection.close()


def load_sources():
    connection = sqlite3.connect(DB_PATH)
    try:
        return [row[0] for row in connection.execute("SELECT DISTINCT source FROM chunks ORDER BY source").fetchall()]
    finally:
        connection.close()


def load_rows(selected_sources=None):
    connection = sqlite3.connect(DB_PATH)
    try:
        if selected_sources:
            placeholders = ", ".join("?" for _ in selected_sources)
            sql = "SELECT source, page, chunk_index, content, embedding " + f"FROM chunks WHERE source IN ({placeholders})"
            return connection.execute(sql, tuple(selected_sources)).fetchall()
        return connection.execute("SELECT source, page, chunk_index, content, embedding FROM chunks").fetchall()
    finally:
        connection.close()


def cosine_similarity(left, right):
    dot_product = sum(a * b for a, b in zip(left, right))
    left_magnitude = sum(value * value for value in left) ** 0.5
    right_magnitude = sum(value * value for value in right) ** 0.5
    if left_magnitude == 0 or right_magnitude == 0:
        return 0.0
    return dot_product / (left_magnitude * right_magnitude)


def retrieve(question, selected_sources=None):
    rows = load_rows(selected_sources)
    if not rows:
        return []
    query_vector = get_cloud_embeddings([question])[0]
    scored = []
    for source, page, chunk_index, content, embedding_json in rows:
        scored.append({
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "content": content,
            "score": cosine_similarity(query_vector, json.loads(embedding_json)),
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:TOP_K]


def answer_question(question, evidence):
    evidence_text = "\n\n".join(
        f"[Source {index}: {item['source']}, page {item['page']}]\n{item['content']}"
        for index, item in enumerate(evidence, start=1)
    )
    system_prompt = (
        "You are a careful research assistant. Answer only from the provided evidence. "
        "Cite claims with the exact bracketed source labels. If the evidence is insufficient, say so clearly."
    )
    user_prompt = f"Question:\n{question}\n\nEvidence:\n{evidence_text}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def render_evidence(evidence):
    for index, item in enumerate(evidence, start=1):
        with st.expander(f"Source {index}: {item['source']}, page {item['page']}"):
            st.write(f"Cosine similarity: {item['score']:.3f}")
            st.write(item["content"])


# --- Database Initialization ---
initialize_database()

# --- Interface Layout ---
st.title("Cloud-Ready RAG Research Assistant")
st.write("Upload PDF or TXT files, compare literal search with semantic retrieval, and generate a cited answer via Cloud APIs.")

if not api_key:
    st.error("API Key missing! Please set `OPENROUTER_API_KEY` in `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")

uploaded_files = st.file_uploader("Upload research documents", type=["pdf", "txt"], accept_multiple_files=True)
pages = extract_documents(uploaded_files)
chunks = chunk_pages(pages)

if uploaded_files:
    st.write(f"Extracted pages: {len(pages)}")
    st.write(f"Prepared chunks: {len(chunks)}")
    if not pages:
        st.write("No readable text was extracted. Scanned PDFs need OCR, which is outside this project.")
    for page in pages[:3]:
        with st.expander(f"Preview: {page['source']}, page {page['page']}"):
            st.write(page["text"][:1000])

question = st.text_input("Research question", value=DEFAULT_QUESTION)
selected_sources = st.multiselect(
    "Limit retrieval to these sources; leave empty to search all", load_sources()
)

if st.button("1. Test literal phrase search", disabled=not chunks):
    matches = literal_phrase_search(chunks, question)
    st.write(f"Exact phrase matches: {len(matches)}")
    if not matches:
        st.write("The full question is absent even though one document discusses the idea.")

if st.button("2. Index documents", disabled=not chunks or not api_key):
    store_chunks(chunks)
    st.write(f"Stored chunks in {DB_PATH}: {count_stored_chunks()}")

if st.button("3. Preview semantic retrieval", disabled=not question or not api_key):
    if count_stored_chunks() == 0:
        st.write("Index at least one document first.")
    else:
        evidence = retrieve(question, selected_sources)
        st.write("Semantic retrieval results")
        render_evidence(evidence)

if st.button("4. Ask the research assistant", disabled=not question or not api_key):
    if count_stored_chunks() == 0:
        st.write("Index at least one document first.")
    else:
        evidence = retrieve(question, selected_sources)
        st.write("Grounded answer")
        st.write(answer_question(question, evidence))
        st.write("Retrieved evidence")
        render_evidence(evidence)