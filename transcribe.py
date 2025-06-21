from pathlib import Path
from typing import Optional
import whisper

def write_srt(segments):
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{millis:03d}"
    srt = ""
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip().replace('-->', '->')
        srt += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt

def transcribe_to_srt(
    audio_path: Path,
    model_name: str = "base",
    srt_out_path: Optional[Path] = None
) -> Path:
    """
    Transcribe audio to SRT using Whisper and save to srt_out_path.
    Returns the path to the SRT file.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), task="transcribe", verbose=False)
    if srt_out_path is None:
        srt_out_path = audio_path.with_suffix(".srt")
    with open(srt_out_path, "w", encoding="utf-8") as f:
        f.write(write_srt(result["segments"]))
    print(f"SRT saved to {srt_out_path}")
    return srt_out_path