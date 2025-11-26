# governance_drift_engine.py
# FULL SELF-CONTAINED DRIFT ENGINE (CLOUD-FRIENDLY, AI-DRIVEN)
# - trains TF-IDF + RandomForestRegressor (if artifacts missing)
# - computes embedding drift using OpenAI embeddings (text-embedding-3-small)
# - uses OpenAI (gpt-4o / gpt-4o-mini) to generate root_causes & recommended_actions (no hard-coding)
# - returns final combined JSON

import os
import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ----------------- CONFIG & SEEDING -----------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CSV = "historical_windows.csv"
ARTIFACT_DIR = "artifacts"
BASELINE_FILE = os.path.join(ARTIFACT_DIR, "baseline_embedding.npy")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"   # you can switch to "gpt-4o"

# Init OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------------------------------------------------------------------------------------
# STEP 1 — DATA & ML: TRAIN REGRESSOR AUTOMATICALLY IF ARTIFACTS MISSING
# ----------------------------------------------------------------------------------------------------

def create_synthetic_csv(path: str, n: int = 60) -> pd.DataFrame:
    """
    Create a synthetic historical_windows.csv for initial training.
    This is only used if no real CSV is present.
    """
    rows = []
    for i in range(n):
        skips = round(abs(random.gauss(3, 3)) + (random.random() < 0.25) * random.uniform(4, 12), 2)
        emerg = int(np.random.poisson(0.8) + (random.random() < 0.2) * random.randint(1, 6))
        after = round(abs(random.gauss(0, 2)) + (random.random() < 0.2) * random.uniform(2, 10), 2)

        parts = []
        if skips > 0.5:
            parts.append(f"TL approver skipped in {skips}% of changes")
        if emerg > 0:
            parts.append(f"{emerg} emergency changes with no CAB note")
        if after > 0.5:
            parts.append(f"After-hours approvals for Critical CIs increased by {after}%")
        if not parts:
            parts.append("Normal operations; no obvious governance issues")

        text = " | ".join(parts)

        # Synthetic numeric drift label
        drift = min(
            1.0,
            0.02
            + (skips / 100) * 3.5
            + emerg * 0.08
            + (after / 100) * 4.0
            + random.gauss(0, 0.03)
        )

        # We still store synthetic root_causes/recs for future possible fine-tuning,
        # but we won't use them in the ML model now.
        causes = []
        if skips > 5:
            causes.append("approval_bypass")
        if emerg >= 1:
            causes.append("emergency_flag_misuse")
        if after > 3:
            causes.append("after_hours_approvals")

        recs = []
        if "approval_bypass" in causes:
            recs += [
                "Enforce second approver for critical CIs",
                "Add approval gating rules",
            ]
        if "emergency_flag_misuse" in causes:
            recs += [
                "Mandatory CAB notification when emergency flag=true",
                "Require justification for emergency flag",
            ]
        if "after_hours_approvals" in causes:
            recs += [
                "Block after-hours approvals without on-call tag",
                "Add after-hours approval review",
            ]

        rows.append({
            "window_id": f"WIN{i+1:04d}",
            "window_text": text,
            "skipped_approvals_delta": skips,
            "emergency_without_CAB_count": emerg,
            "after_hours_critical_delta": after,
            "drift_score_label": round(float(drift), 4),
            "root_causes_label": "|".join(causes),
            "recommended_actions_label": "|".join(recs),
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def train_or_load_ml_models():
    """
    Train a RandomForestRegressor to predict drift_score_label from:
      - TF-IDF(window_text)
      - numeric deltas (skipped, emergency, after_hours)
    If artifacts already exist, just load them.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    tfidf_path = os.path.join(ARTIFACT_DIR, "tfidf.joblib")
    reg_path = os.path.join(ARTIFACT_DIR, "regressor.joblib")

    models_exist = all(os.path.exists(p) for p in [tfidf_path, reg_path])

    if models_exist:
        print("Loading ML artifacts...")
        tfidf = joblib.load(tfidf_path)
        reg = joblib.load(reg_path)
        return tfidf, reg

    # If artifacts missing — train fresh model
    print("Training ML models (artifacts missing)...")

    if not Path(CSV).exists():
        print("No historical_windows.csv found — generating synthetic dataset...")
        df = create_synthetic_csv(CSV)
    else:
        df = pd.read_csv(CSV)
        print(f"Loaded {CSV} with {len(df)} rows")

    texts = df["window_text"].astype(str).tolist()
    tfidf = TfidfVectorizer(max_features=1024, stop_words="english")
    X_text = tfidf.fit_transform(texts).toarray()

    X_num = df[["skipped_approvals_delta",
                "emergency_without_CAB_count",
                "after_hours_critical_delta"]].astype(float).values
    X = np.hstack([X_text, X_num])

    y_reg = df["drift_score_label"].astype(float).values

    X_train, X_test, ytrain_reg, ytest_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=RANDOM_SEED
    )

    reg = RandomForestRegressor(n_estimators=120, random_state=RANDOM_SEED)
    reg.fit(X_train, ytrain_reg)
    print("✔ Trained RandomForestRegressor")

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(reg, reg_path)
    print("✔ ML artifacts saved")

    return tfidf, reg


# Load ML artifacts at import time
tfidf, reg = train_or_load_ml_models()


def predict_alert(window_text: str,
                  skipped: float,
                  emergency: int,
                  after_hours: float) -> Dict[str, Any]:
    """
    ML model predicts only a numeric drift_score (0–1).
    Root causes and recommended actions are delegated to the LLM.
    """
    vec_text = tfidf.transform([window_text]).toarray()
    Xr = np.hstack([vec_text, np.array([[skipped, emergency, after_hours]])])

    pred_drift = float(reg.predict(Xr)[0])
    pred_drift = max(0.0, min(1.0, pred_drift))

    # Simple heuristic confidence: stronger drift => higher confidence
    confidence = round(0.5 + 0.5 * pred_drift, 3)

    return {
        "drift_score": round(pred_drift, 4),
        "confidence": confidence,
    }


# ----------------------------------------------------------------------------------------------------
# STEP 2 — EMBEDDING DRIFT (OpenAI embeddings)
# ----------------------------------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    """
    Compute embedding using OpenAI embeddings API.
    This replaces local models so it's light for cloud hosts like Render.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    emb = resp.data[0].embedding
    return np.array(emb, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_baseline_embedding() -> np.ndarray:
    """
    Computes (or loads) baseline embedding by averaging OpenAI embeddings
    for all historical window_text rows in CSV.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if os.path.exists(BASELINE_FILE):
        print("✔ Loaded cached baseline embedding")
        return np.load(BASELINE_FILE)

    print("Computing baseline embedding with OpenAI embeddings...")
    if not Path(CSV).exists():
        df = create_synthetic_csv(CSV)
    else:
        df = pd.read_csv(CSV)

    texts = df["window_text"].astype(str).tolist()
    embeddings = [get_embedding(txt) for txt in texts]
    baseline = np.mean(np.array(embeddings), axis=0)

    np.save(BASELINE_FILE, baseline)
    print("✔ Baseline embedding saved")
    return baseline


def embedding_drift_score(current_emb: np.ndarray, baseline_emb: np.ndarray) -> float:
    """
    Convert cosine similarity to a [0,1] drift score, where higher = more drift.
    """
    sim = cosine_similarity(current_emb, baseline_emb)
    return max(0.0, min(1.0, 1.0 - sim))


# ----------------------------------------------------------------------------------------------------
# STEP 3 — CALL OPENAI (Chat Completion) FOR ROOT CAUSES & RECOMMENDATIONS
# ----------------------------------------------------------------------------------------------------

def call_llm(window_text: str,
             skipped: float,
             emergency: int,
             after_hours: float,
             ml_drift_score: float,
             embedding_drift_score: float) -> Dict[str, Any]:
    """
    Use GPT to infer:
      - drift_score (final)
      - root_causes (free-text labels, not hard-coded)
      - recommended_actions (concrete, contextual)
      - confidence
    based on evidence + metrics + model hints.
    """

    prompt = f"""
You are Governance Drift Sentinel, an expert in IT process governance and audit.

Your task:
- Read the evidence about a weekly governance window.
- Consider numeric signals and model drift scores as hints.
- Decide:
  - overall drift_score (0–1, higher = worse governance drift)
  - root_causes: short category-style strings (e.g. "approval bypass", "emergency flag misuse", "after-hours approvals", "missing CAB oversight", "weak change documentation", "insufficient segregation of duties"). You may create new labels if needed.
  - recommended_actions: 3–6 concrete, actionable recommendations (policy changes, workflow rules, training, alerts, additional approvals).

Evidence (behavior statements):
{window_text}

Numeric signals:
- skipped_approvals_delta: {skipped}        (percentage points vs baseline)
- emergency_without_CAB_count: {emergency}
- after_hours_critical_delta: {after_hours} (percentage points vs baseline)

Model hints:
- ml_drift_score: {ml_drift_score}
- embedding_drift_score: {embedding_drift_score}

Guidance:
- If signals are mild and single-dimensional, keep drift_score lower (e.g. 0.1–0.3) and recommendations lighter.
- If signals are strong, multi-dimensional, or show a pattern of bypassing governance, use higher drift_score (e.g. 0.4–0.8) and stronger controls.
- Only use 2–5 root_causes and 3–6 recommended_actions.

Return STRICT JSON only in this structure:
{{
  "drift_score": <float between 0 and 1>,
  "root_causes": [<short strings>],
  "recommended_actions": [<short actionable sentences>],
  "confidence": <float between 0 and 1>
}}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are Governance Drift Sentinel. Always respond with VALID JSON only, no markdown, no explanation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds ```json ... ```
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            if "{" in p and "}" in p:
                raw = p.replace("json", "").strip()
                break

    return json.loads(raw)


# ----------------------------------------------------------------------------------------------------
# STEP 4 — FULL PIPELINE
# ----------------------------------------------------------------------------------------------------

def analyze_governance_window(window_text: str,
                              skipped: float,
                              emergency: int,
                              after_hours: float) -> Dict[str, Any]:
    """
    Full pipeline:
    - ML drift_score from TF-IDF + RandomForestRegressor
    - Embedding drift via OpenAI embeddings vs baseline
    - LLM reasoning to generate final drift_score, root_causes, recommendations, confidence
    - Combined/meta scores attached in the final JSON
    """

    # 1) ML drift score
    ml = predict_alert(window_text, skipped, emergency, after_hours)
    ml_drift = ml["drift_score"]

    # 2) Embedding drift
    current_emb = get_embedding(window_text)
    baseline_emb = compute_baseline_embedding()
    emb_drift = embedding_drift_score(current_emb, baseline_emb)

    # 3) LLM reasoning
    llm_json = call_llm(
        window_text=window_text,
        skipped=skipped,
        emergency=emergency,
        after_hours=after_hours,
        ml_drift_score=ml_drift,
        embedding_drift_score=emb_drift
    )

    # 4) Attach meta fields for debugging/analytics
    llm_json["ml_drift_score"] = ml_drift
    llm_json["embedding_drift_score"] = round(emb_drift, 4)

    # combined_drift_score: you can choose your own policy, e.g. max of all
    final_drift = float(llm_json.get("drift_score", 0.0))
    combined_drift = max(final_drift, ml_drift, emb_drift)
    llm_json["combined_drift_score"] = round(combined_drift, 4)

    return llm_json


# ----------------------------------------------------------------------------------------------------
# DEMO RUN (local testing)
# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
    "40% of change records lacked complete implementation notes. "
    "Multiple approvals were provided without sufficient justification or reviewer comments."
)

    print("\nRunning full governance drift engine (OpenAI-only embeddings)...\n")
    result = analyze_governance_window(sample, 12, 0, 3)
    print("\nFinal Result:\n")
    print(json.dumps(result, indent=2))
