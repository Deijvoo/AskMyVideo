# AskMyVideo

Minimal pipeline for extracting transcript and text chunks from a local MP4 video using Whisper ASR, and running retrieval-augmented Q&A (RAG) with local embeddings and LLMs.

## Features
- Extract audio from any local MP4 file (using ffmpeg)
- Transcribe audio to SRT subtitles (OpenAI Whisper)
- Detect scene changes and extract keyframes for visual context
- Chunk transcript and embed with SentenceTransformer
- Build a FAISS vector index for retrieval
- Ask questions about your video via interactive CLI (local RAG with LLaVA)
- All helper files are saved in a subdirectory named after the video

## Setup
1. Install Python 3.9+
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure ffmpeg is installed and in your PATH.
4. Make sure you have pulled the required model from Ollama:
   ```bash
   ollama pull llava:7b-v1.6-mistral-q4_0
   ```

## Usage

Run the `main.py` script and provide the path to your video. It will automatically process the video and start an interactive chat session.

```bash
python video2rag/main.py /path/to/your/video.mp4
```

This single command will:
1.  Extract the audio, transcribe it, and pull keyframes from the video.
2.  Chunk the transcript and create a searchable FAISS vector database.
3.  Launch an interactive chat session where you can ask questions about the video.

### Text-Only Mode
If you only care about the transcript and don't need visual context, you can run in text-only mode using the `--no-images` flag. This will be faster as it doesn't load or send images to the AI.

```bash
python video2rag/main.py /path/to/your/video.mp4 --no-images
```

---
This repo is a work in progress. See commit history and issues for updates. 