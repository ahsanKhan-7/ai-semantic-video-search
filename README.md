# AI Semantic Video Search Engine

This project allows users to upload a video and search inside it using natural language.
The app returns relevant timestamps and plays the matching video segment.

## Features

- Upload a video
- Extract audio using FFmpeg
- Transcribe speech using faster-whisper
- Create timestamped transcript chunks
- Generate embeddings using Sentence Transformers
- Search using cosine similarity
- Improve results using semantic score plus keyword bonus
- Refine playback timestamp using original transcript segments
- Play a relevant preview clip

## Tech Stack

- Python
- Streamlit
- faster-whisper
- Sentence Transformers
- NumPy
- imageio-ffmpeg

## How to Run Locally

Step 1:

pip install -r requirements.txt

Step 2:

streamlit run app.py

## Recommended Video Size

For free deployment, short videos are recommended.

Best: 3 to 10 minutes

Avoid: very large videos on free hosting

## Project Type

No vector database.
No LLM API.
Uses speech-to-text, embeddings, and cosine similarity.