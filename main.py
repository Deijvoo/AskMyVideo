import subprocess
from pathlib import Path
import sys

def run_process_video(video_path):
    print(f"[1/4] Extracting audio from {video_path}...")
    subprocess.run([sys.executable, 'process_video.py', str(video_path)], check=True, cwd=Path(__file__).parent)

def run_transcribe(audio_path):
    print(f"[2/4] Transcribing {audio_path}...")
    subprocess.run([sys.executable, 'transcribe.py', str(audio_path)], check=True, cwd=Path(__file__).parent)

def run_chunk_embed(srt_path):
    print(f"[3/4] Chunking and embedding {srt_path}...")
    subprocess.run([sys.executable, 'chunk_embed.py', str(srt_path)], check=True, cwd=Path(__file__).parent)

def run_ask(faiss_path, question):
    print(f"[4/4] Answering: {question}")
    subprocess.run([sys.executable, 'ask.py', str(faiss_path), question], check=True, cwd=Path(__file__).parent)

def main():
    video_path = Path(input("Paste the full path to your video file (.mp4): ").strip())
    if not video_path.exists():
        print(f"Error: File {video_path} does not exist.")
        return
    video_stem = video_path.stem
    out_dir = video_path.parent / video_stem
    audio_path = out_dir / f'{video_stem}.mp3'
    srt_path = out_dir / f'{video_stem}.srt'
    faiss_path = out_dir / f'{video_stem}.faiss'

    if not audio_path.exists():
        run_process_video(video_path)
    if not srt_path.exists():
        run_transcribe(audio_path)
    if not faiss_path.exists():
        run_chunk_embed(srt_path)

    question = input("Enter your question about the video: ")
    run_ask(faiss_path, question)

if __name__ == "__main__":
    main() 