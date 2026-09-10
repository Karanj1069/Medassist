# MediAssist — AI-powered Medical Assistance

Your original `Medical-assistance` symptom-checker, rebuilt around exactly two ideas:
**your own data** (read straight from the CSVs with Python's built-in `csv` module) and
**your local Ollama `llama3.2`**. That's it — no pandas, no numpy, no scikit-learn, no
pickled model. Two pip packages total: `Flask` and `openai`.

## Project structure

```
medassist/
├── app.py            # Flask routes only (3 routes): page, /api/predict, /api/chat
├── data.py            # The whole knowledge base: CSV loading, symptom matcher,
│                       # chat search, and LLM-context formatting - one file
├── ai.py               # ~50 lines - talks to your local Ollama over the OpenAI client
├── Modelfile            # Optional persona/safety system-prompt for Ollama
├── requirements.txt     # Flask, openai - nothing else
├── data/                 # The original CSVs
├── templates/index.html
└── static/{style.css, script.js, img.png}
```

Three Python files, no ML dependency, no framework beyond Flask itself.

## How it works

1. **`data.py`** loads the CSVs into plain dicts and sets at startup (`csv.DictReader`, no
   DataFrame). `match_diseases()` ranks diseases against the symptoms you picked using
   Jaccard similarity (`|matched| / |union|`) — pure set math, fully explainable: you get
   back *which* symptoms matched each candidate, not just a label. `search_diseases()` does
   the same idea for free-text chat questions, scoring diseases by shared words.
2. **`ai.py`** sends the matched shortlist + the relevant CSV facts to your local
   `llama3.2` (same `base_url=http://localhost:11434/v1` OpenAI-client setup as your
   notebook) and asks it to explain the result in plain language, grounded in that context.
3. **`app.py`** just wires the two together across 3 routes.

Trade-off worth knowing: this replaces the old SVC classifier, which was trained on
`Training.csv` and could pick up patterns across the whole feature space. The new matcher is
simpler and fully transparent — you can always see *why* it made a call — but it only
reasons over the symptoms literally listed per disease in `symtoms_df.csv`, so it won't be
more accurate than that dataset's coverage.

## Setup

```bash
cd medassist
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Make sure Ollama is running and the model is pulled:

```bash
ollama serve            # if not already running
ollama pull llama3.2
```

Run the app:

```bash
python app.py
```

Visit `http://127.0.0.1:5000`.

**If Ollama isn't running**, the symptom checker still works (matching + CSV lookups) — the
AI summary and chat just return a friendly "AI assistant is offline" message instead of
crashing the app.

## About "fine-tuning"

Actually retraining llama3.2's weights (LoRA/unsloth) needs a training pipeline and GPU time
— overkill here. What you actually want — an AI that stays grounded in your data and on
persona — comes from two lighter layers already built in:

1. **Retrieval (`data.context_for()`)** — every answer is grounded in facts pulled straight
   from your CSVs, not the model's general training data.
2. **`Modelfile`** — an Ollama-native way to bake in the persona + safety rules without
   retraining weights:

   ```bash
   ollama create medibot -f Modelfile
   ```

   Then set `MEDASSIST_MODEL=medibot` (env var, see `ai.py`) instead of the default
   `llama3.2` to use it.

If you want to go further into real fine-tuning later, that's a good next project — happy to
help set up a LoRA fine-tune with `unsloth` when you're ready.

## A data-quality note

`medications.csv` looks mismatched for some rows (e.g. Pneumonia's listed medications look
like they belong to a different condition). This is a flaw in the original dataset — worth
cleaning up since the matcher's context and the AI's grounding are only as good as this data.

## Safety notes baked into the AI layer

- The system prompt (`ai.py` / `Modelfile`) forbids stating a diagnosis as certain, forbids
  inventing dosages, and tells the model to push urgent-sounding symptoms toward emergency
  care first.
- All structured facts (precautions, meds, diet, workout) shown in the UI come straight from
  your CSVs — the LLM only paraphrases/contextualizes them.
- Every result and chat reply carries a "not a diagnosis, confirm with a professional"
  reminder.

## Next steps you might want

- Weight the matcher by `Symptom-severity.csv` (already loaded in `data.py` as `SEVERITY`,
  not yet used in `match_diseases()`) so a rarer/more severe matched symptom counts for more
  than a common one like fatigue.
- Containerize with a `Dockerfile` + `docker-compose.yml` that also runs Ollama, so the
  whole thing is one `docker compose up`.
