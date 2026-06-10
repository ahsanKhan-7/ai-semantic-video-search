
import html
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer


# =========================================================
# Page Setup
# =========================================================
st.set_page_config(
    page_title="AI Semantic Video Search",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 AI Semantic Video Search Engine")
st.caption("Upload a video, search a topic, and play the relevant video segment.")
st.caption("Whisper + Sentence Transformers + Cosine Similarity | No Vector Database | No LLM")


# =========================================================
# Runtime folders
# =========================================================
BASE_DIR = Path("runtime_data")
VIDEO_DIR = BASE_DIR / "videos"
AUDIO_DIR = BASE_DIR / "audios"
CLIP_DIR = BASE_DIR / "clips"

for folder in [VIDEO_DIR, AUDIO_DIR, CLIP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# Model loading
# =========================================================
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size):
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =========================================================
# Utility functions
# =========================================================
def format_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def save_uploaded_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == "":
        suffix = ".mp4"

    video_path = VIDEO_DIR / f"input_video{suffix}"

    # remove old videos
    for old_file in VIDEO_DIR.glob("input_video.*"):
        old_file.unlink()

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return video_path


def extract_audio(video_path):
    audio_path = AUDIO_DIR / "input_audio.wav"

    if audio_path.exists():
        audio_path.unlink()

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return audio_path


def transcribe_audio(audio_path, whisper_model, language="en"):
    segments_generator, info = whisper_model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True
    )

    transcript_segments = []

    for segment in segments_generator:
        text = segment.text.strip()

        if text:
            transcript_segments.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text
            })

    if len(transcript_segments) == 0:
        raise ValueError("No speech detected in this video.")

    return transcript_segments


def create_chunks(transcript_segments, chunk_seconds=30):
    chunks = []

    current_start = None
    current_end = None
    current_text = []

    for segment in transcript_segments:
        if current_start is None:
            current_start = segment["start"]

        current_text.append(segment["text"])
        current_end = segment["end"]

        if current_end - current_start >= chunk_seconds:
            chunks.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_text)
            })

            current_start = None
            current_end = None
            current_text = []

    if current_text:
        chunks.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_text)
        })

    return chunks


def build_embeddings(chunks, embedding_model):
    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return chunk_embeddings


def cosine_similarity(query_vector, chunk_vectors):
    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)

    similarities = np.dot(chunk_vectors, query_vector) / (chunk_norms * query_norm)

    return similarities


def get_query_terms(query):
    query = query.lower()
    terms = re.findall(r"[a-zA-Z0-9]+", query)
    terms = [term for term in terms if len(term) > 1]
    return terms


def keyword_bonus(query, text):
    query_lower = query.lower()
    text_lower = text.lower()

    terms = get_query_terms(query)
    bonus = 0.0

    if query_lower in text_lower:
        bonus += 0.35

    matched_terms = 0

    for term in terms:
        if term in text_lower:
            matched_terms += 1

    if terms:
        bonus += 0.15 * (matched_terms / len(terms))

    return bonus


def find_exact_segment_start(query, chunk, transcript_segments):
    query_terms = get_query_terms(query)

    chunk_start = chunk["start"]
    chunk_end = chunk["end"]

    candidate_segments = []

    for segment in transcript_segments:
        if segment["end"] >= chunk_start and segment["start"] <= chunk_end:
            candidate_segments.append(segment)

    for segment in candidate_segments:
        if query.lower() in segment["text"].lower():
            return max(0, segment["start"] - 1)

    for segment in candidate_segments:
        segment_text_lower = segment["text"].lower()

        for term in query_terms:
            if term in segment_text_lower:
                return max(0, segment["start"] - 1)

    return chunk_start


def search_video_v2(query, chunks, chunk_embeddings, embedding_model, transcript_segments, top_k=3, min_score=0.20):
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_vector=query_embedding,
        chunk_vectors=chunk_embeddings
    )

    all_results = []

    for index, semantic_score in enumerate(similarities):
        chunk = chunks[index]

        bonus = keyword_bonus(query, chunk["text"])
        final_score = float(semantic_score) + bonus

        refined_start = find_exact_segment_start(
            query=query,
            chunk=chunk,
            transcript_segments=transcript_segments
        )

        result = {
            "start": chunk["start"],
            "end": chunk["end"],
            "play_start": refined_start,
            "text": chunk["text"],
            "semantic_score": float(semantic_score),
            "keyword_bonus": bonus,
            "final_score": final_score
        }

        all_results.append(result)

    all_results = sorted(
        all_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    filtered_results = [
        result for result in all_results
        if result["final_score"] >= min_score
    ]

    return filtered_results[:top_k]


def create_preview_clip(video_path, start_time, end_time):
    start_time = int(start_time)
    end_time = int(end_time)

    preview_path = CLIP_DIR / f"preview_{start_time}_{end_time}.mp4"

    if preview_path.exists():
        return preview_path

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    duration = max(5, int(end_time - start_time))

    command = [
        ffmpeg_path,
        "-y",
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-movflags", "+faststart",
        str(preview_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return preview_path


def display_result_card(result, index):
    chunk_start = format_time(result["start"])
    chunk_end = format_time(result["end"])
    play_start = format_time(result["play_start"])

    semantic_score = round(float(result["semantic_score"]), 3)
    keyword_bonus_value = round(float(result["keyword_bonus"]), 3)
    final_score = round(float(result["final_score"]), 3)

    with st.container(border=True):
        st.subheader(f"Result {index}")

        st.write(f"**Chunk Time:** {chunk_start} - {chunk_end}")
        st.write(f"**Video Will Play From:** {play_start}")

        st.write(
            f"**Semantic Score:** {semantic_score} | "
            f"**Keyword Bonus:** {keyword_bonus_value} | "
            f"**Final Score:** {final_score}"
        )

        st.write("**Transcript:**")
        st.info(result["text"])


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Settings")

    whisper_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        index=0,
        help="tiny is fastest and best for free deployment. base/small are more accurate but slower."
    )

    language = st.selectbox(
        "Video language",
        ["en", "auto"],
        index=0
    )

    chunk_seconds = st.slider(
        "Chunk size",
        min_value=20,
        max_value=60,
        value=30,
        step=5
    )

    top_k = st.slider(
        "Max results",
        min_value=1,
        max_value=5,
        value=3
    )

    min_score = st.slider(
        "Min score",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05
    )


# =========================================================
# Upload + Processing
# =========================================================
uploaded_video = st.file_uploader(
    "Upload a short educational video",
    type=["mp4", "mov", "mkv", "avi"]
)

if uploaded_video is not None:
    st.info("For free deployment, use short videos. 3–10 minute videos are recommended.")

    if st.button("Process Video", type="primary"):
        with st.spinner("Processing video... This may take a few minutes."):
            try:
                video_path = save_uploaded_video(uploaded_video)

                audio_path = extract_audio(video_path)

                whisper_model = load_whisper_model(whisper_size)

                selected_language = None if language == "auto" else language

                transcript_segments = transcribe_audio(
                    audio_path=audio_path,
                    whisper_model=whisper_model,
                    language=selected_language
                )

                chunks = create_chunks(
                    transcript_segments=transcript_segments,
                    chunk_seconds=chunk_seconds
                )

                embedding_model = load_embedding_model()

                chunk_embeddings = build_embeddings(
                    chunks=chunks,
                    embedding_model=embedding_model
                )

                st.session_state["video_path"] = str(video_path)
                st.session_state["transcript_segments"] = transcript_segments
                st.session_state["chunks"] = chunks
                st.session_state["chunk_embeddings"] = chunk_embeddings
                st.session_state["selected_result"] = None

                st.success("Video processed successfully!")
                st.write("Transcript segments:", len(transcript_segments))
                st.write("Searchable chunks:", len(chunks))

            except Exception as e:
                st.error("Processing failed.")
                st.exception(e)


# =========================================================
# Search
# =========================================================
if "chunks" in st.session_state:
    st.divider()
    st.subheader("Search inside the video")

    query = st.text_input(
        "Enter topic",
        placeholder="Example: siri, email filtering, future of NLP"
    )

    if query:
        embedding_model = load_embedding_model()

        results = search_video_v2(
            query=query,
            chunks=st.session_state["chunks"],
            chunk_embeddings=st.session_state["chunk_embeddings"],
            embedding_model=embedding_model,
            transcript_segments=st.session_state["transcript_segments"],
            top_k=top_k,
            min_score=min_score
        )

        if len(results) == 0:
            st.warning("No strong result found. Try lowering Min Score or using a different query.")
        else:
            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.write("### Search Results")

                for i, result in enumerate(results, start=1):
                    display_result_card(result, i)

                    if st.button(
                        f"Play result {i} from {format_time(result['play_start'])}",
                        key=f"play_{i}_{query}_{result['play_start']}"
                    ):
                        st.session_state["selected_result"] = result

            with right_col:
                st.write("### Video Preview")

                selected_result = st.session_state.get("selected_result")

                if selected_result is None:
                    selected_result = results[0]

                st.write(
                    f"Playing from **{format_time(selected_result['play_start'])}** "
                    f"to **{format_time(selected_result['end'])}**"
                )

                try:
                    clip_path = create_preview_clip(
                        video_path=Path(st.session_state["video_path"]),
                        start_time=selected_result["play_start"],
                        end_time=selected_result["end"]
                    )

                    st.video(str(clip_path))

                except Exception as e:
                    st.error("Could not create preview clip.")
                    st.exception(e)

else:
    st.warning("Upload and process a video first.")
