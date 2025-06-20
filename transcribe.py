from pathlib import Path
import whisper

def transcribe(audio_path: Path, model_name: str = "base", out_path: Path = Path("data/transcript.txt")) -> str:
    """Transcribe audio file to text using Whisper. Saves transcript to out_path."""
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    text = result["text"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text 