import os
import re
import glob

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv() 

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docs = load_transcripts()

    if not docs:
        raise ValueError(
            f"No .vtt transcript files found in '{DATA_DIR}'"
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    if not chunks:
        raise ValueError("No chunks created from transcripts.")

    store = Chroma(
        collection_name="transcripts",
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    if store._collection.count() == 0:
        print(f"Indexing {len(chunks)} chunks...")
        store.add_documents(chunks)
    else:
        print(
            f"Loaded {store._collection.count()} existing chunks."
        )

    return store


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- uv run python src/retriever.py
if __name__ == "__main__":
    print("1. Building retriever...")

    retriever = build_retriever()
    print("2. Retriever built successfully!")

    query = "what is regression testing?"
    print(f"3. Searching for: {query}")

    results = retriever.invoke(query)

    print(f"4. Retrieved {len(results)} documents.\n")

    for i, r in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Metadata: {r.metadata}")
        print(f"Content: {r.page_content[:150]}...\n")