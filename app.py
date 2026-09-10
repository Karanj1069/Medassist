"""
app.py
------
Flask routes only - all data/matching logic lives in data.py, all LLM
calls in ai.py. Three routes, nothing unused:

  /            - the page
  /api/predict - symptoms in, ranked candidates + AI summary out
  /api/chat    - free-text chat, grounded in the CSV knowledge base
"""

from flask import Flask, render_template, request, jsonify

import data
import ai

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", all_symptoms=data.ALL_SYMPTOMS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    symptoms = [s.strip() for s in payload.get("symptoms", []) if s and s.strip()]
    if not symptoms:
        return jsonify({"error": "Please select at least one symptom."}), 400

    candidates = data.match_diseases(symptoms, top_n=5)
    if not candidates:
        return jsonify({"error": "No matching condition found for those symptoms."}), 404

    top_disease = candidates[0]["disease"]
    rec = data.get_disease_record(top_disease)

    ai_summary = None
    if payload.get("with_ai", True):
        shortlist = "\n".join(
            f"- {c['disease']}: matched {c['matched_count']}/{c['disease_symptom_count']} "
            f"known symptoms (matched: {', '.join(c['matched'])})"
            for c in candidates[:3]
        )
        prompt = (
            f"A user reported these symptoms: {', '.join(symptoms)}.\n\n"
            f"Symptom-overlap ranking of possible conditions:\n{shortlist}\n\n"
            "In 4-6 short sentences: explain what the top candidate likely means, note a "
            "runner-up if it's worth keeping in mind, and give general next steps. Use the "
            "retrieved context as your source of truth - don't invent facts outside it."
        )
        context = data.context_for([c["disease"] for c in candidates[:3]])
        ai_summary = ai.chat([{"role": "user", "content": prompt}], context=context)

    return jsonify({**rec, "ai_summary": ai_summary, "candidates": candidates})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    history = payload.get("history", [])  # [{role, content}, ...] prior turns from the client
    predicted_disease = payload.get("predicted_disease")  # optional, from the checker card

    if not message:
        return jsonify({"error": "Message is empty."}), 400

    sources = data.search_diseases(message, k=3)
    if predicted_disease and predicted_disease not in sources:
        sources.insert(0, predicted_disease)

    messages = [{"role": m["role"], "content": m["content"]} for m in history if m.get("content")]
    messages.append({"role": "user", "content": message})

    reply = ai.chat(messages, context=data.context_for(sources))
    return jsonify({"reply": reply, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True)
