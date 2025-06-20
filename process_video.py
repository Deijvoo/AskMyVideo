from pathlib import Path
import subprocess
import sys
from transcribe import transcribe_to_srt

def extract_audio(video_path: Path, audio_path: Path):
    """Extract audio from video using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "libmp3lame", str(audio_path)
    ], check=True)
    print(f"Audio extracted to {audio_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_path>")
        sys.exit(1)
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: File {video_path} does not exist.")
        sys.exit(1)
    video_stem = video_path.stem
    out_dir = video_path.parent / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / f'{video_stem}.mp3'
    srt_path = out_dir / f'{video_stem}.srt'
    extract_audio(video_path, audio_path)
    transcribe_to_srt(audio_path, srt_out_path=srt_path) 