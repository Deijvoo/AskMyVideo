from pathlib import Path
from transcribe import transcribe
from chunk_and_save import chunk_text, save_chunks

AUDIO_PATH = Path("data/test-video1.mp3")

if __name__ == "__main__":
    print("Transcribing audio...")
    transcript = transcribe(AUDIO_PATH)
    print("Transcript saved to data/transcript.txt")

    print("Chunking transcript and saving chunks...")
    chunks = chunk_text(transcript)
    save_chunks(chunks)
    print("Chunks saved to data/chunks.txt")

    print("All done! You can now copy from data/transcript.txt or data/chunks.txt and use them in ChatGPT.")