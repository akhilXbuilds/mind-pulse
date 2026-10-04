from pathlib import Path
import math
import re
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "emotion_model.pkl"


# ============================================================
# GOEMOTIONS → MIND PULSE MAPPING
# ============================================================

EMOTION_MAP = {
    # Positive
    "admiration": "Positive",
    "amusement": "Positive",
    "approval": "Positive",
    "caring": "Positive",
    "excitement": "Positive",
    "gratitude": "Positive",
    "joy": "Positive",
    "love": "Positive",
    "optimism": "Positive",
    "pride": "Positive",
    "relief": "Positive",

    # Sad
    "disappointment": "Sad",
    "grief": "Sad",
    "remorse": "Sad",
    "sadness": "Sad",

    # Angry
    "anger": "Angry",
    "annoyance": "Angry",
    "disapproval": "Angry",
    "disgust": "Angry",

    # Fear
    "fear": "Fear",
    "nervousness": "Fear",

    # Confused
    "confusion": "Confused",

    # Neutral / mixed
    "realization": "Neutral",
    "surprise": "Neutral",
    "neutral": "Neutral",
    "curiosity": "Neutral",
    "desire": "Neutral",
    "embarrassment": "Neutral",
}


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert text into simple lowercase word tokens.
    """

    text = text.lower()

    return re.findall(r"\b[a-zA-Z']+\b", text)


# ============================================================
# BUILD VOCABULARY
# ============================================================

def build_vocabulary(texts, min_df=2, max_features=30000):
    """
    Build vocabulary from training text.

    min_df:
        Ignore words appearing in fewer than min_df documents.

    max_features:
        Keep only the most frequent features.
    """

    document_frequency = {}
    word_frequency = {}

    for text in texts:

        tokens = tokenize(text)

        unique_tokens = set(tokens)

        for token in unique_tokens:
            document_frequency[token] = (
                document_frequency.get(token, 0) + 1
            )

        for token in tokens:
            word_frequency[token] = (
                word_frequency.get(token, 0) + 1
            )

    valid_words = [
        word
        for word, count in document_frequency.items()
        if count >= min_df
    ]

    # Keep the most frequent words
    valid_words.sort(
        key=lambda word: word_frequency[word],
        reverse=True
    )

    valid_words = valid_words[:max_features]

    vocabulary = {
        word: index
        for index, word in enumerate(valid_words)
    }

    return vocabulary


# ============================================================
# TF-IDF
# ============================================================

def compute_tf_idf(texts, vocabulary, idf=None):
    """
    Convert text into TF-IDF vectors.

    TF  = Term Frequency
    IDF = Inverse Document Frequency
    """

    num_documents = len(texts)
    vocab_size = len(vocabulary)

    matrix = np.zeros(
        (num_documents, vocab_size),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Term Frequency
    # --------------------------------------------------------

    for row, text in enumerate(texts):

        tokens = tokenize(text)

        if not tokens:
            continue

        counts = {}

        for token in tokens:

            if token in vocabulary:
                counts[token] = counts.get(token, 0) + 1

        total_words = sum(counts.values())

        if total_words == 0:
            continue

        for token, count in counts.items():

            column = vocabulary[token]

            matrix[row, column] = count / total_words

    # --------------------------------------------------------
    # IDF
    # --------------------------------------------------------

    if idf is None:

        document_frequency = np.zeros(
            vocab_size,
            dtype=np.float32
        )

        for text in texts:

            tokens = set(tokenize(text))

            for token in tokens:

                if token in vocabulary:

                    document_frequency[
                        vocabulary[token]
                    ] += 1

        idf = np.log(
            (1 + num_documents)
            / (1 + document_frequency)
        ) + 1

    # --------------------------------------------------------
    # TF × IDF
    # --------------------------------------------------------

    matrix *= idf

    # L2 normalization
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    matrix = matrix / norms

    return matrix, idf


# ============================================================
# MULTINOMIAL NAIVE BAYES
# ============================================================

class MultinomialNaiveBayes:

    def __init__(self):

        self.classes = None
        self.class_log_prior = None
        self.feature_log_prob = None

    def fit(self, X, y):

        self.classes = np.unique(y)

        num_classes = len(self.classes)
        num_features = X.shape[1]

        self.class_log_prior = np.zeros(
            num_classes
        )

        self.feature_log_prob = np.zeros(
            (num_classes, num_features)
        )

        # ----------------------------------------------------
        # Train each class
        # ----------------------------------------------------

        for class_index, class_name in enumerate(
            self.classes
        ):

            class_mask = y == class_name

            X_class = X[class_mask]

            # Prior probability
            class_count = X_class.shape[0]

            self.class_log_prior[class_index] = math.log(
                class_count / len(y)
            )

            # Feature counts
            feature_counts = X_class.sum(
                axis=0
            )

            # Laplace smoothing
            feature_counts = feature_counts + 1.0

            total_features = feature_counts.sum()

            probabilities = (
                feature_counts
                / total_features
            )

            self.feature_log_prob[
                class_index
            ] = np.log(probabilities)

        return self

    def predict_proba(self, X):

        scores = (
            X @ self.feature_log_prob.T
            + self.class_log_prior
        )

        # Numerical stability
        max_scores = np.max(
            scores,
            axis=1,
            keepdims=True
        )

        exp_scores = np.exp(
            scores - max_scores
        )

        probabilities = (
            exp_scores
            / exp_scores.sum(
                axis=1,
                keepdims=True
            )
        )

        return probabilities

    def predict(self, X):

        probabilities = self.predict_proba(X)

        indices = np.argmax(
            probabilities,
            axis=1
        )

        return self.classes[indices]


# ============================================================
# DATASET CONVERSION
# ============================================================

def convert_split(dataset_split, label_names):

    texts = []
    emotions = []

    for item in dataset_split:

        label_id = item["labels"][0]

        original_label = label_names[label_id]

        if original_label not in EMOTION_MAP:
            continue

        texts.append(item["text"])

        emotions.append(
            EMOTION_MAP[original_label]
        )

    return pd.DataFrame(
        {
            "text": texts,
            "emotion": emotions
        }
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, split_name):

    predictions = model.predict(X)

    accuracy = np.mean(
        predictions == y
    )

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Correct: {(predictions == y).sum()}"
        f" / {len(y)}"
    )

    print("\nPer-class results:")

    for class_name in model.classes:

        mask = y == class_name

        if mask.sum() == 0:
            continue

        class_accuracy = np.mean(
            predictions[mask] == y[mask]
        )

        print(
            f"{class_name:10s} "
            f"{class_accuracy:.4f} "
            f"({mask.sum()} samples)"
        )

    return accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("             MIND PULSE ML TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load GoEmotions
    # --------------------------------------------------------

    print("\n📥 Loading GoEmotions...")

    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified"
    )

    label_names = (
        dataset["train"]
        .features["labels"]
        .feature
        .names
    )

    print("✅ Dataset loaded")

    print(dataset)

    # --------------------------------------------------------
    # 2. Convert labels
    # --------------------------------------------------------

    print("\n🔄 Converting emotion labels...")

    train_df = convert_split(
        dataset["train"],
        label_names
    )

    validation_df = convert_split(
        dataset["validation"],
        label_names
    )

    test_df = convert_split(
        dataset["test"],
        label_names
    )

    print(
        f"\nTraining examples:   {len(train_df)}"
    )

    print(
        f"Validation examples: {len(validation_df)}"
    )

    print(
        f"Test examples:       {len(test_df)}"
    )

    # --------------------------------------------------------
    # 3. Class distribution
    # --------------------------------------------------------

    print("\n📊 Training class distribution:")

    print(
        train_df["emotion"]
        .value_counts()
    )

    # --------------------------------------------------------
    # 4. Build vocabulary
    # --------------------------------------------------------

    print("\n🧠 Building vocabulary...")

    vocabulary = build_vocabulary(
        train_df["text"].tolist(),
        min_df=2,
        max_features=30000
    )

    print(
        f"Vocabulary size: {len(vocabulary)}"
    )

    # --------------------------------------------------------
    # 5. TF-IDF training
    # --------------------------------------------------------

    print("\n⚙️ Creating training TF-IDF...")

    X_train, idf = compute_tf_idf(
        train_df["text"].tolist(),
        vocabulary
    )

    print(
        f"Training matrix: {X_train.shape}"
    )

    # --------------------------------------------------------
    # 6. TF-IDF validation
    # --------------------------------------------------------

    print("\n⚙️ Creating validation TF-IDF...")

    X_validation, _ = compute_tf_idf(
        validation_df["text"].tolist(),
        vocabulary,
        idf
    )

    # --------------------------------------------------------
    # 7. TF-IDF test
    # --------------------------------------------------------

    print("\n⚙️ Creating test TF-IDF...")

    X_test, _ = compute_tf_idf(
        test_df["text"].tolist(),
        vocabulary,
        idf
    )

    # --------------------------------------------------------
    # 8. Train model
    # --------------------------------------------------------

    print("\n🤖 Training Multinomial Naive Bayes...")

    model = MultinomialNaiveBayes()

    model.fit(
        X_train,
        train_df["emotion"].to_numpy()
    )

    print("✅ Training complete!")

    # --------------------------------------------------------
    # 9. Validation
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_validation,
        validation_df["emotion"].to_numpy(),
        "Validation"
    )

    # --------------------------------------------------------
    # 10. Test
    # --------------------------------------------------------

    test_accuracy = evaluate_model(
        model,
        X_test,
        test_df["emotion"].to_numpy(),
        "Test"
    )

    # --------------------------------------------------------
    # 11. Save model
    # --------------------------------------------------------

    model_package = {
        "model": model,
        "vocabulary": vocabulary,
        "idf": idf,
        "classes": model.classes,
    }

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model_package,
            file
        )

    print(
        f"\n💾 Model saved to:\n"
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # 12. Test real sentences
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("             QUICK MODEL TEST")
    print("=" * 60)

    examples = [
        "I am really happy today!",
        "I feel completely hopeless and sad.",
        "I am so angry about what happened.",
        "I am nervous about tomorrow.",
        "I don't understand what is happening.",
        "Today was a normal day.",
    ]

    example_matrix, _ = compute_tf_idf(
        examples,
        vocabulary,
        idf
    )

    predictions = model.predict(
        example_matrix
    )

    probabilities = model.predict_proba(
        example_matrix
    )

    for text, prediction, probability in zip(
        examples,
        predictions,
        probabilities
    ):

        confidence = np.max(
            probability
        )

        print(
            f"\nText: {text}"
        )

        print(
            f"Prediction: {prediction}"
        )

        print(
            f"Confidence: {confidence:.2%}"
        )

    print("\n")
    print("=" * 60)
    print("          🎉 TRAINING FINISHED")
    print("=" * 60)

    print(
        f"\nFinal test accuracy: "
        f"{test_accuracy:.2%}"
    )

    print("\n")


if __name__ == "__main__":
    main()