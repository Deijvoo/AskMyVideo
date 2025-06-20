# AskMyVideo

Minimal pipeline for extracting transcript and text chunks from a local MP4 video using Whisper ASR.

## Current Features
- Extract audio from a local MP4 file (using ffmpeg)
- Transcribe audio to text (OpenAI Whisper)
- Chunk transcript into manageable pieces
- Save transcript and chunks to a dedicated folder named after the video

## Usage
1. Place your `.mp4` file in the `data/` folder.
2. Run `process_video.py` to extract audio, transcribe, and chunk the transcript.
3. Find your results in `data/<video_name>/transcript.txt` and `data/<video_name>/chunks.txt`.

## Example
```bash
python process_video.py
```

## In Progress
- Add speaker diarization (who spoke when)
- Loader/progress indicator for long transcriptions

## Planned Features
- Local LLM Q&A (e.g., using Ollama or LM Studio instead of OpenAI API)
- Streamlined UI for selecting files and viewing results
- Support for more audio/video formats

---
This repo is a work in progress. See commit history and issues for updates. 