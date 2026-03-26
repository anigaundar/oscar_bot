"""
OSCAR v4 Conference Audio Bot
Conversational voice bot using:
  - Deepgram STT (streaming mic input via websocket) - SDK v6
  - Google Gemini (QnA agent)
  - Deepgram TTS (text-to-speech playback) - SDK v6

Setup:
  pip install deepgram-sdk sounddevice numpy google-generativeai
  export GEMINI_API_KEY="your-gemini-key"
  export DEEPGRAM_API_KEY="your-deepgram-key"
"""

import os
import threading
import time
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

from deepgram import DeepgramClient
from deepgram.core.events import EventType

from oscar_qna_agent import OSCARAgent

# ── Config ──────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Mic settings
CHANNELS = 1
RATE = 16000
CHUNK = 4096

# TTS settings
TTS_MODEL = "aura-asteria-en"

# Silence detection: seconds of silence after last final transcript
SILENCE_TIMEOUT = 2.0


# ── TTS: Convert text to audio and play it ──────────────────────────────
def speak(text: str):
    """Use Deepgram SDK v6 TTS to speak text aloud."""
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    # Collect all audio chunks
    audio_chunks = []
    for chunk in client.speak.v1.audio.generate(
        text=text,
        model=TTS_MODEL,
        encoding="linear16",
        sample_rate=RATE,
        container="none",
    ):
        audio_chunks.append(chunk)

    audio_data = b"".join(audio_chunks)

    # Play through speakers
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    sd.play(audio_array, samplerate=RATE, blocksize=CHUNK)
    sd.wait()


# ── STT: Listen for one utterance ───────────────────────────────────────
def listen_for_utterance() -> str:
    """
    Open mic, stream to Deepgram STT, and return the full utterance
    once the user stops speaking (detected by silence after final transcripts).
    """

    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    collected_parts: list[str] = []
    last_final_time = [0.0]
    stop_event = threading.Event()
    ready = threading.Event()

    with client.listen.v1.connect(
        model="nova-3",
        language="en",
        encoding="linear16",
        sample_rate=RATE,
        channels=CHANNELS,
    ) as connection:

        def on_open(_):
            ready.set()

        def on_message(result):
            channel = getattr(result, "channel", None)
            if not channel or not hasattr(channel, "alternatives"):
                return
            transcript = channel.alternatives[0].transcript
            is_final = getattr(result, "is_final", False)
            if transcript and is_final:
                collected_parts.append(transcript.strip())
                last_final_time[0] = time.time()
            elif transcript:
                print(f"  [listening] {transcript}", end="\r")

        def on_error(error):
            print(f"[STT Error] {error}")
            stop_event.set()

        connection.on(EventType.OPEN, on_open)
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)

        def stream_mic():
            ready.wait()
            mic = sd.InputStream(
                samplerate=RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK,
            )
            mic.start()
            print("\n🎤 Listening...")
            try:
                while not stop_event.is_set():
                    data, _ = mic.read(CHUNK)
                    connection.send_media(data.tobytes())
            except Exception:
                pass  # websocket closed, stop sending
            finally:
                mic.stop()
                mic.close()

        mic_thread = threading.Thread(target=stream_mic, daemon=True)
        mic_thread.start()

        # Run start_listening in a background thread (it blocks to process ws messages)
        def listen_loop():
            try:
                connection.start_listening()
            except Exception:
                stop_event.set()

        listen_thread = threading.Thread(target=listen_loop, daemon=True)
        listen_thread.start()

        # Wait for user to finish speaking:
        # After receiving final transcripts, wait for SILENCE_TIMEOUT
        # of no new finals to consider the utterance complete.
        # Max total wait: 30 seconds.
        start_time = time.time()
        while not stop_event.is_set():
            time.sleep(0.1)
            elapsed = time.time() - start_time
            if elapsed > 30:
                break
            if collected_parts and (time.time() - last_final_time[0]) > SILENCE_TIMEOUT:
                break

        stop_event.set()

    user_utterance = " ".join(collected_parts)
    print(f"\nYou: {user_utterance}")
    return " ".join(collected_parts)


# ── Main conversational loop ────────────────────────────────────────────
def main():
    if not DEEPGRAM_API_KEY:
        print("Error: Set DEEPGRAM_API_KEY environment variable")
        return
    if not GEMINI_API_KEY:
        print("Error: Set GEMINI_API_KEY environment variable")
        return

    # Init the QnA agent
    agent = OSCARAgent(api_key=GEMINI_API_KEY)

    print("=" * 50)
    print("  OSCAR v4 Conference Voice Assistant")
    print("  Speak into your mic. Say 'goodbye' to exit.")
    print("=" * 50)

    # Greet the user
    greeting = """Hello & Welcome to Organon's largest, immersive & fully virtual scientific E-Congress “Oscar V4 conference”. How can I help you?"""
    print(f"\nAssistant: {greeting}")
    speak(greeting)

    while True:
        # ── Listen phase ──
        transcript = listen_for_utterance()

        if not transcript:
            continue

        print(f"\nYou: {transcript}")

        # Check for exit
        if any(word in transcript.lower() for word in ("goodbye", "bye", "stop", "exit", "quit")):
            farewell = "Goodbye! Enjoy the conference."
            print(f"\nAssistant: {farewell}")
            speak(farewell)
            break

        # ── Think phase ──
        try:
            answer = agent.ask(transcript)
        except Exception as e:
            answer = "I'm sorry, I'm having trouble right now. Please try again."
            print(f"[Agent Error] {e}")

        print(f"\nAssistant: {answer}")

        # ── Speak phase ──
        speak(answer)


if __name__ == "__main__":
    main()
