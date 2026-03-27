"""
OSCAR v4 Conference Voice Assistant – Streamlit Demo
Uses:
  - audio-recorder-streamlit for browser mic recording
  - Deepgram SDK v6 for STT (pre-recorded) and TTS
  - Google Gemini via OSCARAgent for QnA

Run:  streamlit run oscar_streamlit.py
"""

import os
import io
import struct
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from deepgram import DeepgramClient
from oscar_qna_agent import OSCARAgent
from audio_recorder_streamlit import audio_recorder

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

TTS_MODEL = "aura-asteria-en"
SAMPLE_RATE = 16000


# ── Helpers ──────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes) -> str:
    """Send recorded audio to Deepgram pre-recorded STT, return transcript."""
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    response = client.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
        language="en",
        punctuate=True,
        smart_format=True,
    )
    # Extract transcript from response
    channels = response.results.channels
    if channels and channels[0].alternatives:
        return channels[0].alternatives[0].transcript
    return ""


def text_to_speech(text: str) -> bytes:
    """Convert text to raw PCM audio via Deepgram TTS, return WAV bytes."""
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    audio_chunks = []
    for chunk in client.speak.v1.audio.generate(
        text=text,
        model=TTS_MODEL,
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        container="none",
    ):
        audio_chunks.append(chunk)

    pcm_data = b"".join(audio_chunks)
    return pcm_to_wav(pcm_data, SAMPLE_RATE, 1, 16)


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int, bits: int) -> bytes:
    """Wrap raw PCM bytes in a WAV header so the browser can play it."""
    data_size = len(pcm)
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))             # chunk size
    buf.write(struct.pack("<H", 1))              # PCM format
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)

    return buf.getvalue()


# ── Streamlit App ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OSCAR v4 Voice Assistant",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ OSCAR Conference Voice Assistant")
st.caption("Speak your question about the conference – schedule, speakers, rooms, and more.")

# Validate keys
if not DEEPGRAM_API_KEY or not GEMINI_API_KEY:
    st.error("Set DEEPGRAM_API_KEY and GEMINI_API_KEY in your .env file.")
    st.stop()

# Init agent in session state (persists across reruns)
if "agent" not in st.session_state:
    st.session_state.agent = OSCARAgent(api_key=GEMINI_API_KEY)

if "messages" not in st.session_state:
    greeting = 'Hello & Welcome to Organon\'s OSCAR V4 conference! How can I help you?'
    st.session_state.messages = [{"role": "assistant", "text": greeting}]
    # Generate greeting audio
    st.session_state.greeting_audio = text_to_speech(greeting)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/wav", autoplay=False)

# Play greeting audio with autoplay on first load
if "greeting_played" not in st.session_state:
    st.session_state.greeting_played = True
    if "greeting_audio" in st.session_state:
        st.audio(st.session_state.greeting_audio, format="audio/wav", autoplay=True)

# ── Audio recorder ───────────────────────────────────────────────────
st.divider()
col1, col2 = st.columns([1, 3])

with col1:
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#2c3e50",
        icon_size="2x",
        pause_threshold=2.0,
        sample_rate=16000,
    )

with col2:
    st.markdown("**Click the mic to record your question**")

# Process recorded audio
if audio_bytes and audio_bytes != st.session_state.get("last_audio"):
    st.session_state.last_audio = audio_bytes

    # STT
    with st.spinner("Transcribing..."):
        transcript = transcribe_audio(audio_bytes)

    if not transcript.strip():
        st.warning("Couldn't catch that. Please try again.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "text": transcript})
        with st.chat_message("user"):
            st.write(transcript)

        # Get agent response
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.agent.ask(transcript)
            except Exception as e:
                answer = "I'm sorry, I'm having trouble right now. Please try again."
                st.error(f"Agent error: {e}")

        # TTS
        with st.spinner("Generating audio..."):
            response_audio = text_to_speech(answer)

        # Show assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "text": answer,
            "audio": response_audio,
        })
        with st.chat_message("assistant"):
            st.write(answer)
            st.audio(response_audio, format="audio/wav", autoplay=True)

# ── Text input fallback ──────────────────────────────────────────────
text_input = st.chat_input("Or type your question here...")
if text_input:
    st.session_state.messages.append({"role": "user", "text": text_input})
    with st.chat_message("user"):
        st.write(text_input)

    with st.spinner("Thinking..."):
        try:
            answer = st.session_state.agent.ask(text_input)
        except Exception as e:
            answer = "I'm sorry, I'm having trouble right now. Please try again."
            st.error(f"Agent error: {e}")

    with st.spinner("Generating audio..."):
        response_audio = text_to_speech(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "text": answer,
        "audio": response_audio,
    })
    with st.chat_message("assistant"):
        st.write(answer)
        st.audio(response_audio, format="audio/wav", autoplay=True)

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown(
        "**OSCAR v4** medical conference by Organon\n\n"
        "📅 April 3-4, 2026\n\n"
        "🕐 UAE Time (GMT+4)\n\n"
        "🏥 8 tracks across Women's Health, Cardiovascular, "
        "Respiratory, Migraine, Bone/Pain, Biosimilars & Pharmacists"
    )
    if st.button("🔄 Reset Conversation"):
        st.session_state.agent.reset()
        greeting = 'Hello & Welcome to Organon\'s OSCAR V4 conference! How can I help you?'
        st.session_state.messages = [{"role": "assistant", "text": greeting}]
        st.rerun()
