import os
import pandas as pd
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_autonomous_model():
    csv_path = os.path.join('dataset', 'dark-patterns-v2.csv')

    if not os.path.exists(csv_path):
        print(f"[-] Dataset not found: {csv_path}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(columns=['Pattern String', 'Pattern Category']).to_csv(csv_path, index=False)
        return False

    print(f"[*] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    df['Pattern String'] = df['Pattern String'].fillna('').astype(str)

    # Determine target column
    if 'Pattern Category' in df.columns:
        target_col = 'Pattern Category'
    elif 'Category' in df.columns:
        target_col = 'Category'
    else:
        target_col = df.columns[2]

    df[target_col] = df[target_col].fillna('').astype(str).str.strip()
    df = df[df['Pattern String'].str.strip() != '']
    df = df[df[target_col] != '']

    # NOTE: We intentionally do NOT use the Deceptive? column to override labels.
    # That column is inconsistently annotated — 97% of Scarcity rows are marked
    # Deceptive?=No even though "Only 2 left!" is clearly a dark pattern.
    # We trust Pattern Category as the ground truth label.

    # Merge very rare classes (< 10 samples) into 'Other Dark Pattern'
    min_class_size = 10
    counts = df[target_col].value_counts()
    tiny_classes = counts[counts < min_class_size].index.tolist()
    # Never merge Not Dark Pattern — it's a first-class label
    tiny_classes = [c for c in tiny_classes if c != 'Not Dark Pattern']
    if tiny_classes:
        print(f"[*] Merging rare classes (< {min_class_size} samples) into 'Other Dark Pattern': {tiny_classes}")
        df.loc[df[target_col].isin(tiny_classes), target_col] = 'Other Dark Pattern'
        if df[target_col].value_counts().get('Other Dark Pattern', 0) < min_class_size:
            df = df[df[target_col] != 'Other Dark Pattern']

    # NOTE: Benign "Not Dark Pattern" examples are now in the CSV directly.
    # No synthetic injection needed here — the dataset is the single source of truth.

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df['Pattern String']
    y = df[target_col]

    print(f"[*] Total training records: {len(df)}")
    print(y.value_counts())

    if len(df) < 10:
        print("[-] Too few records to train.")
        return False

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
        min_df=2
    )

    can_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42,
        stratify=y if can_stratify else None
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = LogisticRegression(
        class_weight='balanced',
        max_iter=3000,
        random_state=42,
        C=1.2,
        solver='lbfgs'
    )
    model.fit(X_train_vec, y_train)
    print("\n=== Evaluation (held-out 15%) ===")
    print(classification_report(y_test, model.predict(X_test_vec), zero_division=0))

    print("[*] Training final model on full dataset...")
    X_full_vec  = vectorizer.fit_transform(X)
    model_final = LogisticRegression(
        class_weight='balanced',
        max_iter=3000,
        random_state=42,
        C=1.2,
        solver='lbfgs'
    )
    model_final.fit(X_full_vec, y)

    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(models_dir, 'dark_pattern_model.pkl'), 'wb') as f:
        pickle.dump(model_final, f)

    print(f"[+] Model saved to {models_dir}/")
    return True


if __name__ == '__main__':
    train_autonomous_model()
