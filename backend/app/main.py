from pathlib import Path
from collections import Counter
import re
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_model_v2.pkl"

RANDOM_SEED = 42
MIN_DF = 2
MAX_FEATURES = 30000

np.random.seed(RANDOM_SEED)


# ============================================================
# GOEMOTIONS -> MIND PULSE MAPPING
# ============================================================

EMOTION_GROUPS = {
    "Positive": [
        "admiration",
        "amusement",
        "approval",
        "caring",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
    ],

    "Sad": [
        "disappointment",
        "grief",
        "remorse",
        "sadness",
    ],

    "Angry": [
        "anger",
        "annoyance",
        "disapproval",
        "disgust",
    ],

    "Fear": [
        "fear",
        "nervousness",
    ],

    "Confused": [
        "confusion",
    ],

    "Neutral": [
        "neutral",
        "realization",
    ],
}


# These emotions are intentionally excluded because
# forcing them into our six classes could introduce noise.
EXCLUDED_EMOTIONS = {
    "curiosity",
    "desire",
    "surprise",
    "embarrassment",
}


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert text into lowercase word tokens.
    Keeps simple contractions such as don't, can't, I'm.
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-z]+(?:'[a-z]+)?",
        text
    )

    return tokens


# ============================================================
# CUSTOM TF-IDF VECTORIZER
# ============================================================

class TfidfVectorizerCustom:

    def __init__(
        self,
        min_df=2,
        max_features=30000
    ):
        self.min_df = min_df
        self.max_features = max_features

        self.vocabulary = {}
        self.idf = None

    def create_features(self, tokens):
        """
        Create:
        - unigrams: happy
        - bigrams: very__happy
        """

        features = []

        # -----------------------------
        # Unigrams
        # -----------------------------

        for token in tokens:
            features.append(token)

        # -----------------------------
        # Bigrams
        # -----------------------------

        for i in range(len(tokens) - 1):

            bigram = (
                tokens[i]
                + "__"
                + tokens[i + 1]
            )

            features.append(bigram)

        return features

    def fit(self, texts):

        document_frequency = Counter()

        print("\nBuilding document frequencies...")

        for text in texts:

            tokens = tokenize(text)

            features = set(
                self.create_features(tokens)
            )

            for feature in features:
                document_frequency[feature] += 1

        # Keep features appearing in at least MIN_DF documents.
        valid_features = [
            feature
            for feature, count
            in document_frequency.items()
            if count >= self.min_df
        ]

        # Most frequent features first.
        valid_features.sort(
            key=lambda feature:
            document_frequency[feature],
            reverse=True
        )

        # Limit vocabulary size.
        valid_features = valid_features[
            :self.max_features
        ]

        self.vocabulary = {
            feature: index
            for index, feature
            in enumerate(valid_features)
        }

        n_documents = len(texts)

        document_counts = np.array(
            [
                document_frequency[feature]
                for feature in valid_features
            ],
            dtype=np.float32
        )

        # Smoothed IDF.
        self.idf = (
            np.log(
                (1 + n_documents)
                / (1 + document_counts)
            )
            + 1
        )

        print(
            f"Vocabulary size: "
            f"{len(self.vocabulary)}"
        )

        return self

    def transform(self, texts):

        matrix = np.zeros(
            (
                len(texts),
                len(self.vocabulary)
            ),
            dtype=np.float32
        )

        for row, text in enumerate(texts):

            tokens = tokenize(text)

            features = self.create_features(
                tokens
            )

            if not features:
                continue

            feature_counts = Counter(
                features
            )

            total_features = len(features)

            for feature, count in feature_counts.items():

                if feature not in self.vocabulary:
                    continue

                column = self.vocabulary[
                    feature
                ]

                tf = (
                    count
                    / total_features
                )

                matrix[row, column] = (
                    tf
                    * self.idf[column]
                )

        return matrix

    def fit_transform(self, texts):

        self.fit(texts)

        return self.transform(texts)


# ============================================================
# MULTINOMIAL NAIVE BAYES
# ============================================================

class MultinomialNaiveBayes:

    def __init__(self, alpha=1.0):

        self.alpha = alpha

        self.classes = None
        self.class_log_prior = None
        self.feature_log_prob = None

    def fit(self, X, y):

        self.classes = np.unique(y)

        number_of_classes = len(
            self.classes
        )

        number_of_features = X.shape[1]

        self.class_log_prior = np.zeros(
            number_of_classes
        )

        self.feature_log_prob = np.zeros(
            (
                number_of_classes,
                number_of_features
            )
        )

        for class_index, class_name in enumerate(
            self.classes
        ):

            X_class = X[
                y == class_name
            ]

            # Prior probability.
            self.class_log_prior[
                class_index
            ] = np.log(
                len(X_class)
                / len(X)
            )

            # Count feature occurrences.
            feature_counts = (
                X_class.sum(axis=0)
            )

            # Laplace smoothing.
            smoothed_counts = (
                feature_counts
                + self.alpha
            )

            total_count = (
                smoothed_counts.sum()
            )

            self.feature_log_prob[
                class_index
            ] = np.log(
                smoothed_counts
                / total_count
            )

        return self

    def predict_log_proba(self, X):

        scores = (
            X
            @ self.feature_log_prob.T
        )

        scores += (
            self.class_log_prior
        )

        # Numerical stability.
        max_scores = scores.max(
            axis=1,
            keepdims=True
        )

        log_sum_exp = (
            max_scores
            + np.log(
                np.exp(
                    scores - max_scores
                ).sum(
                    axis=1,
                    keepdims=True
                )
            )
        )

        return (
            scores
            - log_sum_exp
        )

    def predict_proba(self, X):

        return np.exp(
            self.predict_log_proba(X)
        )

    def predict(self, X):

        log_probabilities = (
            self.predict_log_proba(X)
        )

        indexes = np.argmax(
            log_probabilities,
            axis=1
        )

        return self.classes[indexes]


# ============================================================
# EMOTION MAPPING
# ============================================================

def create_label_mapping():

    mapping = {}

    for group, emotions in (
        EMOTION_GROUPS.items()
    ):

        for emotion in emotions:
            mapping[emotion] = group

    return mapping


# ============================================================
# DATA CLEANING
# ============================================================

def prepare_dataset(
    split,
    label_names
):

    label_mapping = (
        create_label_mapping()
    )

    rows = []

    for item in split:

        text = item["text"]

        if text is None:
            continue

        text = text.strip()

        # Remove empty text.
        if not text:
            continue

        # Get original GoEmotions label.
        original_label = (
            label_names[
                item["labels"][0]
            ]
        )

        # Ignore ambiguous classes.
        if original_label not in label_mapping:
            continue

        new_label = (
            label_mapping[
                original_label
            ]
        )

        rows.append(
            {
                "text": text,
                "emotion": new_label
            }
        )

    df = pd.DataFrame(rows)

    before = len(df)

    # Remove exact duplicate texts.
    df = df.drop_duplicates(
        subset=["text"]
    )

    df = df.reset_index(
        drop=True
    )

    removed = (
        before - len(df)
    )

    print(
        f"Removed duplicate texts: "
        f"{removed}"
    )

    return df


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

def balance_training_data(df):

    counts = (
        df["emotion"]
        .value_counts()
    )

    print(
        "\nClass distribution "
        "before balancing:"
    )

    print(counts)

    # Smallest class determines
    # the balanced target size.
    target_size = counts.min()

    print(
        f"\nTarget samples per class: "
        f"{target_size}"
    )

    balanced_parts = []

    for emotion in sorted(
        counts.index
    ):

        class_df = df[
            df["emotion"] == emotion
        ]

        sampled = class_df.sample(
            n=target_size,
            random_state=RANDOM_SEED
        )

        balanced_parts.append(
            sampled
        )

    balanced_df = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    # Shuffle balanced dataset.
    balanced_df = balanced_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    )

    balanced_df = (
        balanced_df
        .reset_index(drop=True)
    )

    print(
        "\nClass distribution "
        "after balancing:"
    )

    print(
        balanced_df[
            "emotion"
        ].value_counts()
    )

    return balanced_df


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    y_true,
    y_pred,
    classes
):

    matrix = np.zeros(
        (
            len(classes),
            len(classes)
        ),
        dtype=int
    )

    class_to_index = {
        name: index
        for index, name
        in enumerate(classes)
    }

    for actual, predicted in zip(
        y_true,
        y_pred
    ):

        actual_index = (
            class_to_index[actual]
        )

        predicted_index = (
            class_to_index[predicted]
        )

        matrix[
            actual_index,
            predicted_index
        ] += 1

    return matrix


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    classes
):

    matrix = create_confusion_matrix(
        y_true,
        y_pred,
        classes
    )

    results = []

    for index, class_name in enumerate(
        classes
    ):

        true_positive = (
            matrix[index, index]
        )

        false_positive = (
            matrix[:, index].sum()
            - true_positive
        )

        false_negative = (
            matrix[index, :].sum()
            - true_positive
        )

        if (
            true_positive
            + false_positive
        ) > 0:

            precision = (
                true_positive
                /
                (
                    true_positive
                    + false_positive
                )
            )

        else:

            precision = 0.0

        if (
            true_positive
            + false_negative
        ) > 0:

            recall = (
                true_positive
                /
                (
                    true_positive
                    + false_negative
                )
            )

        else:

            recall = 0.0

        if (
            precision + recall
        ) > 0:

            f1 = (
                2
                * precision
                * recall
                /
                (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0

        results.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        )

    accuracy = np.mean(
        np.array(y_true)
        == np.array(y_pred)
    )

    macro_f1 = np.mean(
        [
            result["f1"]
            for result in results
        ]
    )

    return (
        accuracy,
        macro_f1,
        results,
        matrix
    )


# ============================================================
# PRINT EVALUATION
# ============================================================

def print_report(
    title,
    y_true,
    y_pred,
    classes
):

    (
        accuracy,
        macro_f1,
        results,
        matrix
    ) = calculate_metrics(
        y_true,
        y_pred,
        classes
    )

    print(
        "\n"
        + "=" * 65
    )

    print(title)

    print(
        "=" * 65
    )

    print(
        f"Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{macro_f1:.4f}"
    )

    print(
        "\nPer-class metrics:"
    )

    print(
        f"{'Class':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    for result in results:

        print(
            f"{result['class']:<12}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
        )

    print(
        "\nConfusion Matrix"
    )

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(
        f"{'':<12}"
        + "".join(
            f"{class_name:<12}"
            for class_name in classes
        )
    )

    for class_name, row in zip(
        classes,
        matrix
    ):

        print(
            f"{class_name:<12}"
            + "".join(
                f"{value:<12}"
                for value in row
            )
        )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "        MIND PULSE — V2 ML TRAINING"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print(
        "\n📥 Loading GoEmotions..."
    )

    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified"
    )

    print(
        "✅ Dataset loaded"
    )

    print(dataset)

    label_names = (
        dataset[
            "train"
        ]
        .features[
            "labels"
        ]
        .feature
        .names
    )

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    print(
        "\n🧹 Cleaning training data..."
    )

    train_df = prepare_dataset(
        dataset["train"],
        label_names
    )

    print(
        "\n🧹 Cleaning validation data..."
    )

    validation_df = prepare_dataset(
        dataset["validation"],
        label_names
    )

    print(
        "\n🧹 Cleaning test data..."
    )

    test_df = prepare_dataset(
        dataset["test"],
        label_names
    )

    print(
        "\nClean dataset sizes:"
    )

    print(
        f"Training   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation : "
        f"{len(validation_df)}"
    )

    print(
        f"Test       : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # BALANCE ONLY TRAINING DATA
    # --------------------------------------------------------

    print(
        "\n⚖️ Balancing training data..."
    )

    train_df = balance_training_data(
        train_df
    )

    # IMPORTANT:
    # Validation and test datasets
    # remain untouched.
    #
    # This prevents data leakage and
    # gives us a more honest evaluation.

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print(
        "\n🧠 Building TF-IDF vocabulary..."
    )

    vectorizer = (
        TfidfVectorizerCustom(
            min_df=MIN_DF,
            max_features=MAX_FEATURES
        )
    )

    print(
        "\n⚙️ Creating training TF-IDF..."
    )

    X_train = (
        vectorizer.fit_transform(
            train_df["text"].tolist()
        )
    )

    print(
        f"Training matrix: "
        f"{X_train.shape}"
    )

    print(
        "\n⚙️ Creating validation TF-IDF..."
    )

    X_validation = (
        vectorizer.transform(
            validation_df["text"].tolist()
        )
    )

    print(
        f"Validation matrix: "
        f"{X_validation.shape}"
    )

    print(
        "\n⚙️ Creating test TF-IDF..."
    )

    X_test = (
        vectorizer.transform(
            test_df["text"].tolist()
        )
    )

    print(
        f"Test matrix: "
        f"{X_test.shape}"
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    y_train = (
        train_df[
            "emotion"
        ].to_numpy()
    )

    y_validation = (
        validation_df[
            "emotion"
        ].to_numpy()
    )

    y_test = (
        test_df[
            "emotion"
        ].to_numpy()
    )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    print(
        "\n🤖 Training Multinomial "
        "Naive Bayes..."
    )

    model = MultinomialNaiveBayes(
        alpha=1.0
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "✅ Training complete!"
    )

    # --------------------------------------------------------
    # CLASS ORDER
    # --------------------------------------------------------

    classes = [
        "Angry",
        "Confused",
        "Fear",
        "Neutral",
        "Positive",
        "Sad"
    ]

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_predictions = (
        model.predict(
            X_validation
        )
    )

    print_report(
        "VALIDATION RESULTS",
        y_validation,
        validation_predictions,
        classes
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = (
        model.predict(
            X_test
        )
    )

    print_report(
        "TEST RESULTS",
        y_test,
        test_predictions,
        classes
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_data = {
        "model": model,
        "vectorizer": vectorizer,
        "classes": classes,
        "version": "v2"
    }

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            model_data,
            file
        )

    print(
        "\n💾 V2 model saved to:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # QUICK MODEL TEST
    # --------------------------------------------------------

    examples = [
        "I am really happy today!",
        "I feel completely hopeless and sad.",
        "I am so angry about what happened.",
        "I am nervous about tomorrow.",
        "I don't understand what is happening.",
        "Today was a normal day.",
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "             V2 QUICK TEST"
    )

    print(
        "=" * 60
    )

    example_matrix = (
        vectorizer.transform(
            examples
        )
    )

    predictions = (
        model.predict(
            example_matrix
        )
    )

    probabilities = (
        model.predict_proba(
            example_matrix
        )
    )

    for text, prediction, probability in zip(
        examples,
        predictions,
        probabilities
    ):

        confidence = (
            probability.max()
            * 100
        )

        print(
            f"\nText: {text}"
        )

        print(
            f"Prediction: "
            f"{prediction}"
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}%"
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "       🎉 V2 TRAINING FINISHED"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()