"""
ai.py
-----
Talks to your local Ollama model via the OpenAI Python client pointed at
Ollama's OpenAI-compatible endpoint - same base_url/messages pattern
already used in your notebook.

chat(messages, context) is the only function the rest of the app calls.
"""

import os
from openai import OpenAI, APIConnectionError

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Set MEDASSIST_MODEL=medibot after running `ollama create medibot -f Modelfile`
# to use the persona-tuned model instead of plain llama3.2.
MODEL = os.environ.get("MEDASSIST_MODEL", "llama3.2")

client = OpenAI(base_url=BASE_URL, api_key="ollama")  # api_key is unused by Ollama

SYSTEM_PROMPT = """You are MediBot, the AI assistant embedded in a home symptom-checker \
web app. You are given RETRIEVED CONTEXT pulled directly from the app's own dataset \
(disease descriptions, precautions, commonly referenced medications, diet and lifestyle \
notes). Ground your answer in that context - do not invent facts that aren't in it or in \
well-established general medical knowledge.

Rules you always follow:
- You are informational only, not a diagnosis and not a substitute for a licensed clinician.
- Never state a disease as certain. Use language like "this pattern is often associated with".
- Never invent specific drug dosages, dosing schedules, or drug interactions. You may name \
the medication classes/names present in the retrieved context, but tell the user to confirm \
dosage and suitability with a pharmacist or doctor.
- If symptoms described sound severe or urgent (e.g. chest pain, breathing difficulty, \
signs of stroke, heavy bleeding, suicidal thoughts), clearly and immediately recommend \
emergency care first, before anything else.
- Keep answers concise, warm, and plain-language - the user is a patient, not a clinician.
- End non-trivial answers with a short reminder to see a healthcare professional for \
confirmation or treatment.
"""


def chat(messages, context: str = "", temperature: float = 0.4) -> str:
    """
    messages: list of {"role": "user"|"assistant", "content": str} - prior turns.
    context:  retrieved-context string from data.context_for() to ground this turn.
    Returns the assistant's reply, or a friendly error string if Ollama isn't
    reachable - so the rest of the app keeps working even without it running.
    """
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        full_messages.append({
            "role": "system",
            "content": f"RETRIEVED CONTEXT (from the app's dataset):\n\n{context}",
        })
    full_messages += messages

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=full_messages, temperature=temperature,
        )
        return response.choices[0].message.content
    except APIConnectionError:
        return (
            "I can't reach the local Ollama server right now, so the AI assistant is "
            "offline. Make sure Ollama is running (`ollama serve`) and that the model "
            f"is pulled (`ollama pull {MODEL}`). The symptom checker below still works "
            "without it."
        )
    except Exception as e:
        return f"AI assistant error: {e}"
