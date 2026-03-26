"""
OSCAR v4 Conference QnA Agent
Uses: Python + Google Gemini 1.5 Flash
Approach: Structured agenda data injected into system prompt (no RAG needed)

Setup:
  pip install google-generativeai pytz
"""

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import google.generativeai as genai
from oscar_schedule_falttened import schedule


CONFERENCE_INFO = schedule["context"]
AGENDA = schedule["sessions"]

# HELPER: Get current time context
def get_time_context() -> str:
    """Returns the current UAE time + session status hints."""
    uae_tz = ZoneInfo("Asia/Dubai")
    now = datetime.now(uae_tz)
    return (
        f"Current UAE date/time: {now.strftime('%Y-%m-%d %H:%M')} (GMT+4)\n"
        f"Conference dates: MAR 23-24, 2026\n"
    )

def build_system_prompt() -> str:
    agenda_json = json.dumps(AGENDA, indent=2)
    time_ctx = get_time_context()
    print("UAE Time Context: ", time_ctx)

    return f"""You are a professional front-desk receptionist for the OSCAR v4 medical conference.\nYou speak to attendees via voice, so responses must be SHORT, POLITE, and EASY to understand.

## CORE BEHAVIOR
- Keep responses to 1–2 short sentences.
- Speak naturally, like a helpful receptionist.
- Be polite and welcoming, but not verbose.
- Give only the most relevant information.
- Do not explain or add unnecessary details.

## RESPONSE STYLE
- Use friendly phrases like:
  "Sure," "Yes," "No," "It’s in...", "It starts at..."
- Do NOT repeat the user’s question.
- Avoid lists unless absolutely necessary.

## STRICT RULES
- Answer ONLY using the agenda and conference data below.
- If info is missing, say:
  "I’m not sure about that. Please check with the help desk."
- All times are in UAE Time (GMT+4).
- For "has session X started?", compare with CURRENT TIME.
- Handle misspelled speaker names with best match.

## EXAMPLES

User: "Where is Dr. Ahmed speaking?"
Assistant: "Sure, Dr. Ahmed is in Room B at 2 PM."

User: "Has the cardiology session started?"
Assistant: "Yes, it started at 10 AM in Room A."
OR
"No, it starts at 10 AM in Room A."

User: "What’s happening now?"
Assistant: "The cardiology session is currently in Room A."

User: "Where should I go for registration?"
Assistant: "Please check the main entrance desk."

## CURRENT TIME
{time_ctx}

## CONFERENCE INFO
{CONFERENCE_INFO}

## FULL AGENDA DATA
{agenda_json}
"""

class OSCARAgent:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY env var or pass api_key")

        genai.configure(api_key=key)

        # Use Gemini 1.5 Flash for speed + low cost
        # Switch to "gemini-1.5-pro" if you need higher reasoning
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=build_system_prompt(),
        )
        self.chat = self.model.start_chat(history=[])

    def ask(self, question: str) -> str:
        """Send a question and get the agent's response."""
        response = self.chat.send_message(question)
        return response.text

    def reset(self):
        """Reset chat history (new conversation)."""
        self.chat = self.model.start_chat(history=[])




# if __name__ == "__main__":
#     main()
