from pathlib import Path

def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Split text into chunks of chunk_size words."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def save_chunks(chunks: list, txt_path: Path = Path("data/chunks.txt")):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n") 