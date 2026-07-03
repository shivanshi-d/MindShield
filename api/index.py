import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "app": "MindShield API",
        "status": "online",
        "endpoints": ["/api/analyze-terms"]
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "online"})

@app.route('/api/analyze-terms', methods=['POST'])
def analyze_terms():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY not configured."}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body."}), 400

    raw_text = data['text'].strip()
    if len(raw_text) < 100:
        return jsonify({"error": "Page text too short to analyze."}), 400

    try:
        # Pre-filter: keep only clauses touching high-risk categories
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

        MAX_CHARS = 18000
        if len(filtered_text) > MAX_CHARS:
            filtered_text = filtered_text[:MAX_CHARS] + "\n\n[Text truncated]"

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
                {"role": "user", "content": f"Analyze these policy clauses:\n\n{filtered_text}"}
            ]
        )

        summary = completion.choices[0].message.content
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Vercel needs the app object exposed as `app`
# No `if __name__ == '__main__'` block needed
 
