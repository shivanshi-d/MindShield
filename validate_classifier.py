"""
validate_classifier.py
Runs both sklearn and the JS classifier on the same test texts,
then checks that predictions and probabilities match within tolerance.
Prints a clear PASS / FAIL for each test case.
"""
import pickle, json, subprocess, sys, math
import numpy as np 

# ── Load sklearn model ─────────────────────────────────────────────────────────
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/dark_pattern_model.pkl', 'rb') as f:
    model = pickle.load(f)

classes = model.classes_.tolist()

TEST_TEXTS = [
    "Only 2 left in stock!",
    "Hurry! Sale ends in 4 minutes",
    "3 people are viewing this right now",
    "Add to cart",
    "No thanks, I don't want to save money",
    "243 people bought this today",
    "Free returns within 30 days",
    "Limited time offer",
    "Selling fast order now",
    "Subscribe to our newsletter",
]

# ── sklearn predictions ────────────────────────────────────────────────────────
features = vectorizer.transform(TEST_TEXTS)
sk_preds = model.predict(features)
sk_probs = model.predict_proba(features)

ground_truth = []
for text, pred, prob in zip(TEST_TEXTS, sk_preds, sk_probs):
    ground_truth.append({
        "text": text,
        "expected_class": pred,
        "expected_probs": {c: float(p) for c, p in zip(classes, prob)}
    })

# ── JS predictions via Node.js ─────────────────────────────────────────────────
# We load model_data.json and run our JS tokenize/vectorize/classify logic
# through a small Node.js wrapper script.

node_script = """
const fs = require('fs');

// Load model
const MODEL = JSON.parse(fs.readFileSync('model_data.json', 'utf8'));

// ── paste classifier internals (no chrome.runtime dependency here) ─────────
const STOP_WORDS = new Set([
  "a","about","above","across","after","afterwards","again","against","all","almost",
  "alone","along","already","also","although","always","am","among","amongst","amoungst",
  "amount","an","and","another","any","anyhow","anyone","anything","anyway","anywhere",
  "are","around","as","at","back","be","became","because","become","becomes","becoming",
  "been","before","beforehand","behind","being","below","beside","besides","between",
  "beyond","bill","both","bottom","but","by","call","can","cannot","cant","co","con",
  "could","cry","de","describe","detail","do","done","down","due","during","each","eg",
  "eight","either","eleven","else","elsewhere","empty","enough","etc","even","ever",
  "every","everyone","everything","everywhere","except","few","fifteen","fify","fill",
  "find","fire","first","five","for","former","formerly","forty","found","four","from",
  "front","full","further","get","give","go","had","has","have","he","hence","her",
  "here","hereafter","hereby","herein","hereupon","hers","herself","him","himself",
  "his","how","however","hundred","i","ie","if","in","inc","indeed","interest","into",
  "is","it","its","itself","keep","last","latter","latterly","least","less","ltd",
  "made","many","may","me","meanwhile","might","mill","mine","more","moreover","most",
  "mostly","move","much","must","my","myself","name","namely","neither","never",
  "nevertheless","next","nine","no","nobody","none","noone","nor","not","nothing",
  "now","nowhere","of","off","often","on","once","one","only","onto","or","other",
  "others","otherwise","our","ours","ourselves","out","over","own","part","per",
  "perhaps","please","put","rather","re","same","see","seem","seemed","seeming",
  "seems","serious","several","she","should","show","side","since","six","sixty",
  "so","some","somehow","someone","something","sometime","sometimes","somewhere",
  "still","such","take","ten","than","that","the","their","them","themselves","then",
  "thence","there","thereafter","thereby","therefore","therein","thereupon","these",
  "they","thick","thin","third","this","those","though","three","through","throughout",
  "thru","thus","to","together","too","top","toward","towards","twelve","twenty","two",
  "un","under","until","up","upon","us","very","via","was","we","well","were","what",
  "whatever","when","whence","whenever","where","whereafter","whereas","whereby",
  "wherein","whereupon","wherever","whether","which","while","whither","who","whoever",
  "whole","whom","whose","why","will","with","within","without","would","yet","you",
  "your","yours","yourself","yourselves"
]);

function tokenize(text) {
  return text.toLowerCase().split(/[^a-z0-9]+/).filter(t => t.length >= 2 && !STOP_WORDS.has(t));
}

function tfidfVectorize(text, vocab, idf) {
  const vocabSize = idf.length;
  const vec = new Float64Array(vocabSize);
  const tokens = tokenize(text);
  const counts = {};
  for (let i = 0; i < tokens.length; i++) {
    const u = tokens[i];
    if (u in vocab) counts[u] = (counts[u] || 0) + 1;
    if (i + 1 < tokens.length) {
      const b = u + " " + tokens[i+1];
      if (b in vocab) counts[b] = (counts[b] || 0) + 1;
    }
  }
  let sumSq = 0;
  for (const [token, count] of Object.entries(counts)) {
    const idx = vocab[token];
    const val = (1.0 + Math.log(count)) * idf[idx];
    vec[idx] = val;
    sumSq += val * val;
  }
  if (sumSq > 0) { const n = Math.sqrt(sumSq); for (let i = 0; i < vocabSize; i++) if (vec[i]) vec[i] /= n; }
  return vec;
}

function linearScores(vec, coef, intercept) {
  const scores = new Float64Array(coef.length);
  for (let c = 0; c < coef.length; c++) {
    let dot = intercept[c];
    const row = coef[c];
    for (let j = 0; j < vec.length; j++) if (vec[j]) dot += row[j] * vec[j];
    scores[c] = dot;
  }
  return scores;
}

function softmax(scores) {
  let mx = scores[0];
  for (let i = 1; i < scores.length; i++) if (scores[i] > mx) mx = scores[i];
  const exps = new Float64Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) { exps[i] = Math.exp(scores[i] - mx); sum += exps[i]; }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}

const texts = JSON.parse(process.argv[2]);
const results = texts.map(text => {
  const vec = tfidfVectorize(text, MODEL.vocab, MODEL.idf);
  const probs = softmax(linearScores(vec, MODEL.coef, MODEL.intercept));
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
  const probObj = {};
  MODEL.classes.forEach((c, i) => probObj[c] = probs[i]);
  return { text, predicted_class: MODEL.classes[maxIdx], probs: probObj };
});

console.log(JSON.stringify(results));
"""

with open('/tmp/validate_node.js', 'w') as f:
    f.write(node_script)

result = subprocess.run(
    ['node', '/tmp/validate_node.js', json.dumps(TEST_TEXTS)],
    capture_output=True, text=True,
    cwd='/home/claude/mindshield-v3'
)

if result.returncode != 0:
    print("Node.js error:", result.stderr)
    sys.exit(1)

js_results = json.loads(result.stdout)

# ── Compare ────────────────────────────────────────────────────────────────────
TOL = 1e-4   # allow up to 0.01% difference in probabilities
all_pass = True

print(f"{'Text':<42} {'sklearn':>14} {'JS':>14} {'Match':>8}")
print("-" * 82)

for gt, js in zip(ground_truth, js_results):
    text = gt['text'][:40]
    sk_class = gt['expected_class']
    js_class = js['predicted_class']
    class_match = sk_class == js_class

    # Check all probabilities are within tolerance
    prob_match = all(
        abs(gt['expected_probs'][c] - js['probs'][c]) < TOL
        for c in classes
    )

    ok = class_match and prob_match
    if not ok:
        all_pass = False

    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"{text:<42} {sk_class:>14} {js_class:>14} {status:>8}")

    if not ok:
        print("  Probability deltas:")
        for c in classes:
            delta = abs(gt['expected_probs'][c] - js['probs'][c])
            if delta > TOL:
                print(f"    {c}: sklearn={gt['expected_probs'][c]:.6f}  js={js['probs'][c]:.6f}  delta={delta:.2e}")

print("-" * 82)
print("ALL PASS" if all_pass else "SOME TESTS FAILED")
sys.exit(0 if all_pass else 1)
