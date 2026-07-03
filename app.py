import os
import csv 
import sys
import json
import pickle
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from train_model import train_autonomous_model

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

MODELS_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'dark_pattern_model.pkl')
DATASET_PATH = os.path.join('dataset', 'dark-patterns-v2.csv')

GLOBAL_VECTORIZER = None
GLOBAL_MODEL = None

# Only surface predictions above this probability threshold.
# Keeps false-positive rate low — critical for user trust.
CONFIDENCE_THRESHOLD = 0.42


# ── Model lifecycle ────────────────────────────────────────────────────────────

def load_pipeline():
    global GLOBAL_VECTORIZER, GLOBAL_MODEL
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
        try:
            with open(VECTORIZER_PATH, 'rb') as f:
                GLOBAL_VECTORIZER = pickle.load(f)
            with open(MODEL_PATH, 'rb') as f:
                GLOBAL_MODEL = pickle.load(f)
            print("[+] ML model loaded successfully.")
            return True
        except Exception as e:
            print(f"[-] Error loading model: {e}", file=sys.stderr)
            return False
    else:
        print("[!] Model files not found. Training from dataset...")
        if train_autonomous_model():
            return load_pipeline()
        return False


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    ready = GLOBAL_VECTORIZER is not None and GLOBAL_MODEL is not None
    return jsonify({
        "app": "MindShield ML Server",
        "version": "3.2.0",
        "status": "online",
        "model_state": "READY" if ready else "UNINITIALIZED",
        "endpoints": ["/api/health", "/api/classify", "/api/report-pattern", "/api/analyze-terms"]
    })


@app.route('/api/health', methods=['GET'])
def health():
    ready = GLOBAL_VECTORIZER is not None and GLOBAL_MODEL is not None
    return jsonify({"status": "online", "model_state": "READY" if ready else "UNINITIALIZED"})


@app.route('/api/classify', methods=['POST'])
def classify():
    if not GLOBAL_VECTORIZER or not GLOBAL_MODEL:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json()
    if not data or 'elements' not in data:
        return jsonify({"error": "Missing 'elements' in request body."}), 400

    try:
        elements = data['elements']

        valid_items, valid_texts = [], []
        for el in elements:
            text = el.get('text', '').strip()
            if text and len(text) >= 8:
                valid_items.append(el)
                valid_texts.append(text)

        if not valid_texts:
            return jsonify({"results": []})

        # Batch vectorize + predict
        features = GLOBAL_VECTORIZER.transform(valid_texts)
        predictions = GLOBAL_MODEL.predict(features)
        probabilities = GLOBAL_MODEL.predict_proba(features)
        class_labels = list(GLOBAL_MODEL.classes_)

        results = []
        for el, pred, probs in zip(valid_items, predictions, probabilities):
            if pred == "Not Dark Pattern":
                continue

            pred_idx = class_labels.index(pred)
            confidence = float(probs[pred_idx])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            results.append({
                "id": el.get('id', 'unknown'),
                "text": el.get('text', '').strip(),
                "selector": el.get('selector', ''),
                "is_dark_pattern": True,
                "category": pred,
                "confidence": round(confidence * 100, 1),
                "explanation": f"Classified as '{pred}' with {round(confidence * 100, 1)}% confidence"
            })

        return jsonify({"results": results})

    except Exception as e:
        print(f"[!] Classification error: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.route('/api/report-pattern', methods=['POST'])
def report_pattern():
    data = request.get_json()
    text = data.get('text', '').strip()
    category = data.get('category', '').strip()

    if not text or not category:
        return jsonify({"error": "Both 'text' and 'category' are required."}), 400

    try:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        file_exists = os.path.exists(DATASET_PATH)
        deceptive = 'No' if category == 'Not Dark Pattern' else 'Yes'

        with open(DATASET_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Pattern String', 'Comment', 'Pattern Category',
                                 'Pattern Type', 'Where in website?', 'Deceptive?', 'Website Page'])
            writer.writerow([text, 'User Report', category, 'Custom',
                             'Dynamic', deceptive, 'User Submitted'])

        if train_autonomous_model():
            if load_pipeline():
                return jsonify({"message": "Pattern saved and model retrained successfully.", "status": "retrained"})
            return jsonify({"message": "Retrained but failed to reload model."}), 500
        return jsonify({"message": "Pattern saved. Dataset too small for retraining."}), 202

    except Exception as e:
        print(f"[!] Report error: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze-terms', methods=['POST'])
def analyze_terms():
    """
    Summarizes Terms of Service / Privacy Policy text using the Claude API.
    Extracts only clauses that touch high-risk categories (data collection,
    third-party sharing, arbitration, etc.) before sending to the model,
    keeping token usage minimal.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY missing from .env file."}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body."}), 400

    raw_text = data['text'].strip()
    if len(raw_text) < 100:
        return jsonify({"error": "Page text too short to analyze."}), 400

    try:
        # ── Pre-filter: keep only clauses that touch high-risk categories ──────
        RISK_KEYWORDS = [
            "privacy", "track", "sell", "third-party", "arbitration", "waive",
            "opt-out", "opt-in", "cookies", "share", "location", "marketing",
            "advertise", "collect", "retention", "terminate", "liability",
            "class action", "binding", "data", "personal information", "disclose",
            "transfer", "license", "indemnif", "jurisdiction"
        ]

        paragraphs = raw_text.split("\n\n")
        relevant_clauses = [
            p.strip() for p in paragraphs
            if len(p.strip()) >= 40
            and any(kw in p.lower() for kw in RISK_KEYWORDS)
        ]

        filtered_text = "\n\n".join(relevant_clauses) if len("\n\n".join(relevant_clauses)) >= 200 else raw_text

        # Hard cap: Claude's context is large but we want fast, cheap responses
        MAX_CHARS = 18000
        if len(filtered_text) > MAX_CHARS:
            filtered_text = filtered_text[:MAX_CHARS] + "\n\n[Text truncated at 18,000 characters]"

        # ── Claude API call ────────────────────────────────────────────────────
        client = Groq(api_key=api_key)

        system_prompt = (
            "You are a neutral data privacy auditor writing for a general audience. "
            "Your reader has no legal background. Analyze the policy clauses provided and produce "
            "a concise, scannable summary in plain English. Focus strictly on:\n"
            "1. **Data collection** — what is collected and why\n"
            "2. **Third-party sharing** — who data is shared with and under what conditions\n"
            "3. **User risks** — arbitration clauses, auto-renewals, waived rights, liability limits\n\n"
            "Format your response with clear ### headers and bullet points. "
            "Ignore cookie-banner boilerplate and generic consent language. "
            "If a clause is genuinely alarming, note it explicitly. Keep the entire response under 500 words."
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=700,
            temperature=0.15,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze these policy clauses and summarize the key user risks:\n\n{filtered_text}"}
            ]
        )

        summary = completion.choices[0].message.content
        return jsonify({"summary": summary})

    except Exception as api_err:
        print(f"[!] Groq API error: {api_err}", file=sys.stderr)
        return jsonify({"error": f"Groq API error: {str(api_err)}"}), 500
    except Exception as e:
        print(f"[!] Terms analysis error: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    load_pipeline()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
