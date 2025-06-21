import argparse
import subprocess
import sys
from pathlib import Path

def main():
    """
    Main script to orchestrate the video processing and Q&A pipeline.
    
    Usage:
        python main.py /path/to/your/video.mp4 [--no-images]
    """
    parser = argparse.ArgumentParser(description="Full pipeline to process a video and start an interactive Q&A session.")
    parser.add_argument("video_path", type=Path, help="Path to the video file.")
    parser.add_argument("--no-images", action="store_true", help="Run the Q&A session in text-only mode.")
    args = parser.parse_args()

    video_path = args.video_path.resolve()
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    # Define paths
    video_dir = Path(__file__).parent.resolve()
    video_stem = video_path.stem
    out_dir = video_path.parent / video_stem
    srt_path = out_dir / f"{video_stem}.srt"
    faiss_path = out_dir / f"{video_stem}.faiss"

    # --- Step 1: Process Video (Audio, Transcription, Frames) ---
    print("--- Running Step 1: Video Processing ---")
    subprocess.run([
        sys.executable, str(video_dir / 'process_video.py'), str(video_path)
    ], check=True)
    print("--- Step 1 Complete ---\n")

    # --- Step 2: Chunk and Embed Transcript ---
    print("--- Running Step 2: Chunking and Embedding ---")
    subprocess.run([
        sys.executable, str(video_dir / 'chunk_embed.py'), str(srt_path)
    ], check=True)
    print("--- Step 2 Complete ---\n")

    # --- Step 3: Start Interactive Q&A Session ---
    print("--- Running Step 3: Starting Q&A Session ---")
    ask_command = [
        sys.executable, str(video_dir / 'ask_ai.py'), str(faiss_path)
    ]
    if args.no_images:
        ask_command.append("--no-images")
    
    print(f"Starting chat for {video_path.name}...")
    if args.no_images:
        print("Running in text-only mode.")
    
    subprocess.run(ask_command)

if __name__ == "__main__":
    main() 