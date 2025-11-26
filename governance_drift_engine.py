# governance_drift_engine.py
# FULL SELF-CONTAINED DRIFT ENGINE (CLOUD-FRIENDLY)
# - trains TF-IDF + RandomForest (if artifacts missing)
# - computes embedding drift using OpenAI embeddings (text-embedding-3-small)
# - calls OpenAI (gpt-4o / gpt-4o-mini)
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CSV = "historical_windows.csv"
ARTIFACT_DIR = "artifacts"
BASELINE_FILE = os.path.join(ARTIFACT_DIR, "baseline_embedding.npy")

# Init OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------------------------------------------------------------------
# STEP 1 — TRAIN ML MODEL AUTOMATICALLY IF ARTIFACTS MISSING
# ----------------------------------------------------------------------------------------------------

def create_synthetic_csv(path: str, n: int = 60):
    rows = []
    for i in range(n):
        skips = round(abs(random.gauss(3, 3)) + (random.random() < 0.25) * random.uniform(4,12), 2)
        emerg = int(np.random.poisson(0.8) + (random.random() < 0.2) * random.randint(1,6))
        after = round(abs(random.gauss(0,2)) + (random.random() < 0.2) * random.uniform(2,10), 2)

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

        drift = min(1.0, 0.02 + (skips/100)*3.5 + emerg*0.08 + (after/100)*4.0 + random.gauss(0,0.03))

        causes = []
        if skips > 5: causes.append("approval_bypass")
        if emerg >= 1: causes.append("emergency_flag_misuse")
        if after > 3: causes.append("after_hours_approvals")

        recs = []
        if "approval_bypass" in causes:
            recs += ["Enforce second approver for critical CIs", "Add approval gating rules"]
        if "emergency_flag_misuse" in causes:
            recs += ["Mandatory CAB notification when emergency flag=true", "Require justification for emergency flag"]
        if "after_hours_approvals" in causes:
            recs += ["Block after-hours approvals without on-call tag", "Add after-hours approval review"]

        rows.append({
            "window_id": f"WIN{i+1:04d}",
            "window_text": text,
            "skipped_approvals_delta": skips,
            "emergency_without_CAB_count": emerg,
            "after_hours_critical_delta": after,
            "drift_score_label": round(float(drift),4),
            "root_causes_label": "|".join(causes),
            "recommended_actions_label": "|".join(recs)
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def train_or_load_ml_models():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    tfidf_path = os.path.join(ARTIFACT_DIR, "tfidf.joblib")
    reg_path = os.path.join(ARTIFACT_DIR, "regressor.joblib")
    clf_path = os.path.join(ARTIFACT_DIR, "classifier.joblib")
    mlb_path = os.path.join(ARTIFACT_DIR, "mlb.joblib")

    models_exist = all(os.path.exists(p) for p in [tfidf_path, reg_path, mlb_path])

    if models_exist:
        print("Loading ML artifacts...")
        tfidf = joblib.load(tfidf_path)
        reg = joblib.load(reg_path)
        clf = joblib.load(clf_path) if os.path.exists(clf_path) else None
        mlb = joblib.load(mlb_path)
        return tfidf, reg, clf, mlb

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

    X_num = df[["skipped_approvals_delta", "emergency_without_CAB_count", "after_hours_critical_delta"]].astype(float).values
    X = np.hstack([X_text, X_num])

    y_reg = df["drift_score_label"].astype(float).values
    df["root_causes_list"] = df["root_causes_label"].fillna("").apply(lambda s: [x for x in s.split("|") if x])

    mlb = MultiLabelBinarizer()
    y_multi = mlb.fit_transform(df["root_causes_list"])

    X_train, X_test, ytrain_reg, ytest_reg, ytrain_multi, ytest_multi = train_test_split(
        X, y_reg, y_multi, test_size=0.2, random_state=RANDOM_SEED
    )

    reg = RandomForestRegressor(n_estimators=120, random_state=RANDOM_SEED)
    reg.fit(X_train, ytrain_reg)
    print("✔ Trained RandomForestRegressor")

    clf = None
    if ytrain_multi.size > 0 and ytrain_multi.sum() > 0:
        clf = RandomForestClassifier(n_estimators=120, random_state=RANDOM_SEED)
        clf.fit(X_train, ytrain_multi)
        print("✔ Trained RandomForestClassifier")

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(reg, reg_path)
    if clf is not None:
        joblib.dump(clf, clf_path)
    joblib.dump(mlb, mlb_path)
    print("✔ ML artifacts saved")

    return tfidf, reg, clf, mlb


# Load ML models
tfidf, reg, clf, mlb = train_or_load_ml_models()

RECOMM_MAP = {
    "approval_bypass": ["Enforce second approver for critical CIs", "Add approval gating rules"],
    "emergency_flag_misuse": ["Mandatory CAB notification when emergency flag=true", "Require justification for emergency flag"],
    "after_hours_approvals": ["Block after-hours approvals without on-call tag", "Add after-hours approval review"]
}

# ----------------------------------------------------------------------------------------------------
# STEP 2 — PREDICT ALERT (ML Only)
# ----------------------------------------------------------------------------------------------------

def predict_alert(window_text, skipped, emergency, after_hours):
    vec_text = tfidf.transform([window_text]).toarray()
    Xr = np.hstack([vec_text, np.array([[skipped, emergency, after_hours]])])

    pred_drift = float(reg.predict(Xr)[0])
    pred_drift = max(0, min(1, pred_drift))

    causes = []
    cause_conf = {}

    if clf is not None:
        proba = clf.predict_proba(Xr)

        try:
            class_probs = [float(p[0][1]) if p.shape[1] > 1 else float(p[0][0]) for p in proba]
        except Exception:
            class_probs = [float(p[0]) for p in proba]

        label_probs = list(zip(mlb.classes_, class_probs))
        label_probs.sort(key=lambda x: x[1], reverse=True)

        for lbl, p in label_probs:
            if p > 0.15:
                causes.append(lbl)
                cause_conf[lbl] = round(p, 3)
    else:
        txt = window_text.lower()
        if "skip" in txt: causes.append("approval_bypass")
        if "emergency" in txt: causes.append("emergency_flag_misuse")
        if "after" in txt: causes.append("after_hours_approvals")

    recs = []
    for c in causes:
        recs += RECOMM_MAP.get(c, [])

    avg_conf = float(np.mean(list(cause_conf.values()))) if cause_conf else 0.55
    overall_conf = round(max(0, min(1, 0.3 * (1 - pred_drift) + 0.7 * avg_conf)), 3)

    return {
        "drift_score": round(pred_drift, 4),
        "root_causes": causes,
        "recommended_actions": recs,
        "confidence": overall_conf
    }


# ----------------------------------------------------------------------------------------------------
# STEP 3 — EMBEDDING DRIFT (OpenAI embeddings — no local MiniLM)
# ----------------------------------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text: str):
    """
    Compute embedding using OpenAI embeddings API.
    This replaces local SentenceTransformer (MiniLM) so it's light for Render.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    emb = resp.data[0].embedding
    return np.array(emb, dtype=np.float32)


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_baseline_embedding():
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
        # If CSV missing, reuse synthetic generator
        df = create_synthetic_csv(CSV)
    else:
        df = pd.read_csv(CSV)

    texts = df["window_text"].astype(str).tolist()
    embeddings = [get_embedding(txt) for txt in texts]
    baseline = np.mean(np.array(embeddings), axis=0)

    np.save(BASELINE_FILE, baseline)
    print("✔ Baseline embedding saved")
    return baseline


def embedding_drift_score(current_emb, baseline_emb):
    sim = cosine_similarity(current_emb, baseline_emb)
    # Convert similarity to drift score in [0,1]
    return max(0.0, min(1.0, 1.0 - sim))


# ----------------------------------------------------------------------------------------------------
# STEP 4 — CALL OPENAI (Chat Completion)
# ----------------------------------------------------------------------------------------------------

def call_llm(window_text, drift_score, causes, actions, confidence):
    prompt = f"""
You are Governance Drift Sentinel.
Return valid JSON only.

Analyze this governance window:

{window_text}

ML signals:
- drift_score: {drift_score}
- root_causes: {causes}
- recommended_actions: {actions}
- confidence: {confidence}

Return STRICT JSON only, in this structure:
{{
 "drift_score": <float>,
 "root_causes": [<strings>],
 "recommended_actions": [<strings>],
 "confidence": <float>
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # or "gpt-4o"
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

    # Strip markdown fences if model adds ```json
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            if "{" in p and "}" in p:
                raw = p.replace("json", "").strip()
                break

    return json.loads(raw)


# ----------------------------------------------------------------------------------------------------
# STEP 5 — FULL PIPELINE
# ----------------------------------------------------------------------------------------------------

def analyze_governance_window(window_text, skipped, emergency, after_hours):

    # 1) ML regression + multi-label classification
    ml = predict_alert(window_text, skipped, emergency, after_hours)

    # 2) Embedding drift using OpenAI embeddings
    current_emb = get_embedding(window_text)
    baseline_emb = compute_baseline_embedding()
    emb_drift = embedding_drift_score(current_emb, baseline_emb)

    # 3) Combine ML + embedding drift
    combined_drift = max(ml["drift_score"], emb_drift)

    # 4) Call LLM for final JSON
    llm_json = call_llm(
        window_text,
        drift_score=combined_drift,
        causes=ml["root_causes"],
        actions=ml["recommended_actions"],
        confidence=ml["confidence"]
    )

    # 5) Attach extra debug / meta fields
    llm_json["ml_drift_score"] = ml["drift_score"]
    llm_json["embedding_drift_score"] = round(emb_drift, 4)
    llm_json["combined_drift_score"] = round(combined_drift, 4)

    return llm_json


# ----------------------------------------------------------------------------------------------------
# DEMO RUN
# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "TL approver skipped in 12% of changes. 6 emergency CAB misses. After-hours approvals increased by 10%."

    print("\nRunning full governance drift engine (OpenAI-only embeddings)...\n")
    result = analyze_governance_window(sample, 12, 6, 10)

    print("\nFinal Result:\n")
    print(json.dumps(result, indent=2))
