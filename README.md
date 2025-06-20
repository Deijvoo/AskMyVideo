# AskMyVideo

Minimal pipeline for extracting transcript and text chunks from a local MP4 video using Whisper ASR, and running retrieval-augmented Q&A (RAG) with local embeddings and LLMs.

## Features
- Extract audio from any local MP4 file (using ffmpeg)
- Transcribe audio to SRT subtitles (OpenAI Whisper)
- Chunk transcript and embed with SentenceTransformer (or Ollama, optional)
- Build a FAISS vector index for retrieval
- Ask questions about your video via CLI (local RAG)
- All helper files (.mp3, .srt, .faiss) are saved in a subdirectory named after the video

## Setup
1. Install Python 3.9+
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure ffmpeg is installed and in your PATH.

## Usage
1. Place your `.mp4` file anywhere on your system.
2. Run the main pipeline:
   ```bash
   python main.py
   ```
   - Paste the full path to your video file when prompted.
   - The pipeline will extract audio, transcribe, chunk, embed, and let you ask a question.
3. All outputs will be in a subdirectory next to your video, e.g.:
   ```
   /path/to/myvideo/myvideo.mp3
   /path/to/myvideo/myvideo.srt
   /path/to/myvideo/myvideo.faiss/
   ```

## Example
```bash
python main.py
# Paste: C:\Users\me\Videos\lecture.mp4
# Enter your question: What is the main topic?
```

## In Progress
- Add speaker diarization (who spoke when)
- Loader/progress indicator for long transcriptions
- More robust error handling and file versioning

## Planned Features
- Local LLM Q&A with more models (Ollama, LM Studio, etc.)
- Streamlined UI for selecting files and viewing results
- Support for more audio/video formats

---
This repo is a work in progress. See commit history and issues for updates. 