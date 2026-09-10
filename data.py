"""
data.py
-------
The app's whole "knowledge base" in one file, built with nothing but
Python's built-in `csv` module - no pandas, no numpy, no sklearn.

What's here:
  - Loading: turns the CSVs into plain dicts/sets at import time.
  - get_disease_record(disease): look up everything known about one disease.
  - match_diseases(symptoms): rank diseases by symptom overlap (replaces
    the old SVC classifier - see the README for the trade-off).
  - search_diseases(query): free-text search for the chat feature.
  - context_for(diseases): format disease facts for the LLM prompt.
"""

import ast
import csv
import os
import re
from collections import defaultdict

def _resolve_data_dir():
    """CSV folder next to this file, or next to the notebook cwd."""
    candidates = []
    try:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    except NameError:
        pass  # __file__ is undefined in Jupyter / interactive sessions
    cwd = os.getcwd()
    candidates.extend([
        os.path.join(cwd, "data"),
        os.path.join(cwd, "medassist", "data"),
    ])
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


DATA_DIR = _resolve_data_dir()


def _rows(filename):
    """Read one CSV file into a list of plain dicts (one dict per row)."""
    with open(os.path.join(DATA_DIR, filename), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _flatten_listlike(value):
    """medications.csv / diets.csv store values like "['A', 'B']" as a
    single string. Parse that into a real list of clean strings."""
    if value and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed]
        except (ValueError, SyntaxError):
            pass
    return [value.strip()] if value and value.strip() else []


# ---------------------------------------------------------------------
# Load everything once, at import time, into simple lookups.
# ---------------------------------------------------------------------
DESCRIPTIONS = {r["Disease"]: r["Description"] for r in _rows("description.csv")}

PRECAUTIONS = defaultdict(list)
for _r in _rows("precautions_df.csv"):
    PRECAUTIONS[_r["Disease"]] = [
        _r[c] for c in ("Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4")
        if _r.get(c)
    ]

MEDICATIONS = defaultdict(list)
for _r in _rows("medications.csv"):
    MEDICATIONS[_r["Disease"]].extend(_flatten_listlike(_r["Medication"]))

DIETS = defaultdict(list)
for _r in _rows("diets.csv"):
    DIETS[_r["Disease"]].extend(_flatten_listlike(_r["Diet"]))

WORKOUTS = defaultdict(list)
for _r in _rows("workout_df.csv"):
    WORKOUTS[_r["disease"]].append(_r["workout"])

# A disease has several rows/variants in symtoms_df.csv - union all the
# symptoms recorded for it into one set.
DISEASE_SYMPTOMS = defaultdict(set)
for _r in _rows("symtoms_df.csv"):
    for _col in ("Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"):
        _symptom = (_r.get(_col) or "").strip()
        if _symptom:
            DISEASE_SYMPTOMS[_r["Disease"]].add(_symptom)

# The canonical, clean 132-symptom vocabulary used to populate the
# checkbox UI (Symptom-severity.csv has no whitespace issues, unlike
# symtoms_df.csv).
_SEVERITY_ROWS = _rows("Symptom-severity.csv")
ALL_SYMPTOMS = sorted({r["Symptom"].strip() for r in _SEVERITY_ROWS})
SEVERITY = {r["Symptom"].strip(): int(r["weight"]) for r in _SEVERITY_ROWS}


def get_disease_record(disease: str) -> dict:
    """Everything known about one disease, in one dict."""
    return {
        "disease": disease,
        "description": DESCRIPTIONS.get(disease, "No description available."),
        "precautions": PRECAUTIONS.get(disease, []),
        "medications": MEDICATIONS.get(disease, []),
        "diet": DIETS.get(disease, []),
        "workout": WORKOUTS.get(disease, []),
    }


# ---------------------------------------------------------------------
# Symptom matching - replaces the old SVC classifier.
# ---------------------------------------------------------------------
def match_diseases(user_symptoms, top_n: int = 5):
    """
    Rank diseases by Jaccard similarity between the symptoms the user
    picked and the symptoms on record for each disease:

        score = |user ∩ disease| / |user ∪ disease|

    Pure set math - explainable, no trained model. Returns the top_n
    diseases with score > 0, each with the matched/missing symptoms so
    the UI and the LLM can show *why*, not just a bare label.
    """
    user_set = {s.strip() for s in user_symptoms if s and s.strip()}
    if not user_set:
        return []

    scored = []
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        matched = user_set & symptoms
        if not matched:
            continue
        union = user_set | symptoms
        scored.append({
            "disease": disease,
            "score": round(len(matched) / len(union), 3),
            "matched": sorted(matched),
            "missing": sorted(symptoms - user_set),
            "matched_count": len(matched),
            "disease_symptom_count": len(symptoms),
        })

    scored.sort(key=lambda c: (c["score"], c["matched_count"]), reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------
# Free-text search over the knowledge base - powers the chat feature.
# ---------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _build_disease_word_index():
    """One set of words per disease, built from everything the CSVs know
    about it. Built once at import time and reused for every /api/chat call."""
    index = {}
    for disease in DESCRIPTIONS:
        rec = get_disease_record(disease)
        text = " ".join([
            disease, rec["description"], " ".join(DISEASE_SYMPTOMS.get(disease, [])),
            " ".join(rec["precautions"]), " ".join(rec["medications"]),
            " ".join(rec["diet"]), " ".join(rec["workout"]),
        ])
        index[disease] = _tokenize(text)
    return index


_DISEASE_WORDS = _build_disease_word_index()


def search_diseases(query: str, k: int = 3):
    """Return up to k disease names whose word-set overlaps most with the
    query's word-set. score = number of shared words. No ML - just counting."""
    query_words = _tokenize(query or "")
    if not query_words:
        return []
    scored = [(d, len(words & query_words)) for d, words in _DISEASE_WORDS.items()]
    scored = sorted((s for s in scored if s[1] > 0), key=lambda s: s[1], reverse=True)
    return [disease for disease, _ in scored[:k]]


def context_for(diseases) -> str:
    """Format one or more disease records into a text block for the LLM."""
    blocks = []
    for disease in dict.fromkeys(diseases):  # dedupe, keep order
        rec = get_disease_record(disease)
        blocks.append(
            f"Disease: {rec['disease']}\n"
            f"Description: {rec['description']}\n"
            f"Typical precautions: {', '.join(rec['precautions']) or 'n/a'}\n"
            f"Commonly referenced medications: {', '.join(rec['medications']) or 'n/a'}\n"
            f"Suggested diet: {', '.join(rec['diet']) or 'n/a'}\n"
            f"Suggested workout/lifestyle: {', '.join(rec['workout']) or 'n/a'}"
        )
    return "\n\n".join(blocks)
