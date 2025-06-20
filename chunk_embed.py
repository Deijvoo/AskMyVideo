import argparse
from pathlib import Path
import re
from typing import List, Dict, Any
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import pickle


def parse_srt(srt_path: Path) -> List[Dict[str, Any]]:
    """Parse SRT file into list of dicts with start, end, and text."""
    pattern = re.compile(
        r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]+?)(?=\n\d+\n|\Z)",
        re.MULTILINE,
    )
    with open(srt_path, "r", encoding="utf-8") as f:
        srt = f.read()
    entries = []
    for match in pattern.finditer(srt):
        idx, start, end, text = match.groups()
        entries.append(
            {
                "start": start.replace(",", "."),
                "end": end.replace(",", "."),
                "text": text.replace("\n", " ").strip(),
            }
        )
    return entries

def srt_entries_to_text(entries: List[Dict[str, Any]]) -> str:
    """Concatenate all SRT entries into a single text."""
    return " ".join(e["text"] for e in entries)

def main():
    parser = argparse.ArgumentParser(
        description="Chunk and embed SRT transcript into FAISS index."
    )
    parser.add_argument("srt_path", type=Path, help="Path to .srt transcript")
    parser.add_argument(
        "--embedder",
        choices=["llama", "mpnet"],
        default="mpnet",
        help="Embedding backend",
    )
    args = parser.parse_args()

    entries = parse_srt(args.srt_path)
    text = srt_entries_to_text(entries)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Map chunk to SRT time range
    chunk_meta = []
    entry_idx = 0
    for i, chunk in enumerate(chunks):
        # Find the first SRT entry that appears in this chunk
        while entry_idx < len(entries) and entries[entry_idx]["text"] not in chunk:
            entry_idx += 1
        start = entries[entry_idx]["start"] if entry_idx < len(entries) else "00:00:00.000"
        end = entries[entry_idx]["end"] if entry_idx < len(entries) else "00:00:00.000"
        chunk_meta.append({"start": start, "end": end, "chunk_id": i})

    # Choose embedding function
    if args.embedder == "llama":
        try:
            embedder = OllamaEmbeddings(model="llama3:embed")
        except Exception:
            print("Ollama embedding failed, falling back to SentenceTransformer.")
            embedder = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    else:
        embedder = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print(f"Embedding {len(chunks)} chunks...")
    # FAISS.from_texts expects a list of texts, an embedding function, and metadatas
    faiss_index = FAISS.from_texts(
        texts=chunks,
        embedding=embedder,
        metadatas=chunk_meta
    )
    slug = args.srt_path.stem
    out_dir = args.srt_path.parent
    out_path = out_dir / f"{slug}.faiss"
    faiss_index.save_local(str(out_path))
    print(f"Stored {len(chunks)} vectors in {out_path}")

if __name__ == "__main__":
    main()

def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Split text into chunks of chunk_size words."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks: list, model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def build_faiss_index(embeddings: np.ndarray, out_path: Path = Path("data/index.faiss")):
    """Build and save a FAISS index from embeddings."""
    arr = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, str(out_path))
    return out_path

def save_chunks(chunks: list, out_path: Path = Path("data/chunks.pkl"), txt_path: Path = Path("data/chunks.txt")):
    with open(out_path, "wb") as f:
        pickle.dump(chunks, f)
    # Save as .txt for easy copy-paste
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n") 