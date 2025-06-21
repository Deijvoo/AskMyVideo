from pathlib import Path
import subprocess
import sys
from transcribe import transcribe_to_srt
import cv2
import json

def extract_audio(video_path: Path, audio_path: Path):
    """Extract audio from video using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "libmp3lame", str(audio_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Audio extracted to {audio_path}")

def extract_global_frames(video_path: Path, out_dir: Path):
    """Extract 4 global keyframes from the video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps if fps > 0 else 0
    n_frames = 4
    frame_times = [i * total_duration / (n_frames - 1) if n_frames > 1 else 0 for i in range(n_frames)]
    
    img_paths = []
    for j, t in enumerate(frame_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            img_name = f"frame_{j+1:02d}.jpg"
            img_path = out_dir / img_name
            cv2.imwrite(str(img_path), frame)
            img_paths.append(str(img_path.relative_to(out_dir.parent)))
            
    cap.release()
    
    scene_infos = [{
        "scene_id": 1,
        "start": 0.0,
        "end": total_duration,
        "frames": [str(p) for p in img_paths]
    }]
    
    json_path = out_dir / "scenes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scene_infos, f, indent=2)
        
    print(f"Extracted {len(img_paths)} global frames and saved scene info to {json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_path>")
        sys.exit(1)
        
    video_path = Path(sys.argv[1]).resolve()
    if not video_path.exists():
        print(f"Error: File {video_path} does not exist.")
        sys.exit(1)
        
    video_stem = video_path.stem
    out_dir = video_path.parent / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    audio_path = out_dir / f'{video_stem}.mp3'
    srt_path = out_dir / f'{video_stem}.srt'
    
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    
    extract_audio(video_path, audio_path)
    transcribe_to_srt(audio_path, srt_out_path=srt_path)
    extract_global_frames(video_path, scenes_dir)
    print(f"\nVideo processing complete for {video_path.name}")
    print(f"Artifacts saved in: {out_dir}") 