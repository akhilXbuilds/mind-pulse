from pathlib import Path
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ============================================================
# MIND PULSE V4
# DistilBERT Emotion Classifier
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "emotion_model_v4"

MODEL_NAME = "distilbert-base-uncased"

RANDOM_SEED = 42

# CPU-friendly first experiment
MAX_TRAIN_PER_CLASS = 2500
MAX_VALIDATION_PER_CLASS = 600
MAX_TEST_PER_CLASS = 600

MAX_LENGTH = 96

NUM_LABELS = 6


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# EMOTION MAPPING
# ============================================================

EMOTION_GROUPS = {
    "Positive": {
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
    },

    "Sad": {
        "disappointment",
        "grief",
        "remorse",
        "sadness",
    },

    "Angry": {
        "anger",
        "annoyance",
        "disapproval",
        "disgust",
    },

    "Fear": {
        "fear",
        "nervousness",
    },

    "Confused": {
        "confusion",
    },

    "Neutral": {
        "neutral",
        "realization",
    },
}


LABEL_NAMES = [
    "Angry",
    "Confused",
    "Fear",
    "Neutral",
    "Positive",
    "Sad",
]


LABEL2ID = {
    label: index
    for index, label in enumerate(LABEL_NAMES)
}

ID2LABEL = {
    index: label
    for index, label in enumerate(LABEL_NAMES)
}


# ============================================================
# DATA PREPARATION
# ============================================================

def create_label_mapping(dataset):

    label_names = dataset.features["labels"].feature.names

    mapping = {}

    for index, name in enumerate(label_names):

        for target_emotion, source_emotions in EMOTION_GROUPS.items():

            if name in source_emotions:

                mapping[index] = target_emotion

    return mapping


def prepare_dataframe(dataset, label_mapping):

    rows = []

    for item in dataset:

        text = str(item["text"]).strip()

        if not text:
            continue

        labels = item["labels"]

        if labels is None or len(labels) == 0:
            continue

        label_id = labels[0]

        if label_id not in label_mapping:
            continue

        emotion = label_mapping[label_id]

        rows.append({
            "text": text,
            "emotion": emotion,
        })

    # Remove exact duplicate texts
    unique_rows = {}

    for row in rows:

        text_key = row["text"].lower().strip()

        if text_key not in unique_rows:

            unique_rows[text_key] = row

    rows = list(unique_rows.values())

    return rows


# ============================================================
# BALANCED SUBSET
# ============================================================

def create_balanced_subset(rows, max_per_class):

    grouped = {
        label: []
        for label in LABEL_NAMES
    }

    for row in rows:

        emotion = row["emotion"]

        if emotion in grouped:
            grouped[emotion].append(row)

    rng = random.Random(RANDOM_SEED)

    selected = []

    print("\nClass distribution:")

    for label in LABEL_NAMES:

        examples = grouped[label]

        rng.shuffle(examples)

        selected_examples = examples[:max_per_class]

        selected.extend(selected_examples)

        print(
            f"{label:<12}: "
            f"{len(examples):>6} available → "
            f"{len(selected_examples):>6} selected"
        )

    rng.shuffle(selected)

    return selected


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_dataset(rows, tokenizer):

    texts = [
        row["text"]
        for row in rows
    ]

    labels = [
        LABEL2ID[row["emotion"]]
        for row in rows
    ]

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    encoded["labels"] = labels

    return encoded


# ============================================================
# PYTORCH DATASET
# ============================================================

class EmotionDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):

        self.encodings = encodings

    def __len__(self):

        return len(self.encodings["labels"])

    def __getitem__(self, index):

        item = {}

        for key in self.encodings:

            item[key] = torch.tensor(
                self.encodings[key][index]
            )

        return item


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = np.mean(
        predictions == labels
    )

    f1_scores = []

    precision_scores = []

    recall_scores = []

    for class_id in range(NUM_LABELS):

        true_positive = np.sum(
            (predictions == class_id)
            & (labels == class_id)
        )

        false_positive = np.sum(
            (predictions == class_id)
            & (labels != class_id)
        )

        false_negative = np.sum(
            (predictions != class_id)
            & (labels == class_id)
        )

        precision = (
            true_positive
            / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive
            / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precision_scores.append(precision)

        recall_scores.append(recall)

        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(
            np.mean(precision_scores)
        ),
        "macro_recall": float(
            np.mean(recall_scores)
        ),
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(predictions, labels):

    predictions = np.argmax(
        predictions,
        axis=1
    )

    matrix = np.zeros(
        (NUM_LABELS, NUM_LABELS),
        dtype=int
    )

    for actual, predicted in zip(
        labels,
        predictions
    ):

        matrix[actual][predicted] += 1

    print("\nConfusion Matrix")
    print("Rows = Actual")
    print("Columns = Predicted\n")

    print(
        f"{'':<12}"
        + "".join(
            f"{label:<12}"
            for label in LABEL_NAMES
        )
    )

    for i, label in enumerate(LABEL_NAMES):

        print(
            f"{label:<12}"
            + "".join(
                f"{matrix[i][j]:<12}"
                for j in range(NUM_LABELS)
            )
        )


# ============================================================
# PER-CLASS REPORT
# ============================================================

def print_per_class_report(predictions, labels):

    predictions = np.argmax(
        predictions,
        axis=1
    )

    print("\nPer-class metrics:")

    print(
        f"{'Class':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    for class_id, label in enumerate(LABEL_NAMES):

        tp = np.sum(
            (predictions == class_id)
            & (labels == class_id)
        )

        fp = np.sum(
            (predictions == class_id)
            & (labels != class_id)
        )

        fn = np.sum(
            (predictions != class_id)
            & (labels == class_id)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0
        )

        print(
            f"{label:<12}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("          MIND PULSE — V4 DISTILBERT")
    print("=" * 65)

    print("\n🧠 Device configuration")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device == "cpu":

        print(
            "⚠️ CPU training enabled."
        )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\n📥 Loading GoEmotions...")

    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified"
    )

    print("✅ Dataset loaded")

    # --------------------------------------------------------
    # LABEL MAPPING
    # --------------------------------------------------------

    print("\n🔄 Creating emotion mapping...")

    label_mapping = create_label_mapping(
        dataset["train"]
    )

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    print("\n🧹 Cleaning datasets...")

    train_rows = prepare_dataframe(
        dataset["train"],
        label_mapping
    )

    validation_rows = prepare_dataframe(
        dataset["validation"],
        label_mapping
    )

    test_rows = prepare_dataframe(
        dataset["test"],
        label_mapping
    )

    print("\nClean dataset sizes:")

    print(
        f"Training   : {len(train_rows)}"
    )

    print(
        f"Validation : {len(validation_rows)}"
    )

    print(
        f"Test       : {len(test_rows)}"
    )

    # --------------------------------------------------------
    # CREATE CONTROLLED BALANCED SUBSETS
    # --------------------------------------------------------

    print("\n⚖️ Creating balanced training subset...")

    train_rows = create_balanced_subset(
        train_rows,
        MAX_TRAIN_PER_CLASS
    )

    print("\n⚖️ Creating validation subset...")

    validation_rows = create_balanced_subset(
        validation_rows,
        MAX_VALIDATION_PER_CLASS
    )

    print("\n⚖️ Creating test subset...")

    test_rows = create_balanced_subset(
        test_rows,
        MAX_TEST_PER_CLASS
    )

    print("\nFinal experiment sizes:")

    print(
        f"Training   : {len(train_rows)}"
    )

    print(
        f"Validation : {len(validation_rows)}"
    )

    print(
        f"Test       : {len(test_rows)}"
    )

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    print("\n🔤 Loading DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("✅ Tokenizer loaded")

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    print("\n⚙️ Tokenizing training data...")

    train_encodings = tokenize_dataset(
        train_rows,
        tokenizer
    )

    print("⚙️ Tokenizing validation data...")

    validation_encodings = tokenize_dataset(
        validation_rows,
        tokenizer
    )

    print("⚙️ Tokenizing test data...")

    test_encodings = tokenize_dataset(
        test_rows,
        tokenizer
    )

    train_dataset = EmotionDataset(
        train_encodings
    )

    validation_dataset = EmotionDataset(
        validation_encodings
    )

    test_dataset = EmotionDataset(
        test_encodings
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\n🤖 Loading pretrained DistilBERT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("✅ Model loaded")

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    print("\n⚙️ Configuring training...")

    training_args = TrainingArguments(

        output_dir=str(
            BASE_DIR / "v4_checkpoints"
        ),

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="steps",

        logging_steps=100,

        learning_rate=2e-5,

        per_device_train_batch_size=8,

        per_device_eval_batch_size=16,

        gradient_accumulation_steps=1,

        num_train_epochs=2,

        weight_decay=0.01,

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True,

        save_total_limit=1,

        report_to="none",

        fp16=False,

        dataloader_num_workers=0,

        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        compute_metrics=calculate_metrics,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("                 🚀 TRAINING V4")
    print("=" * 65)

    trainer.train()

    print("\n✅ V4 training complete!")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("              VALIDATION RESULTS")
    print("=" * 65)

    validation_result = trainer.evaluate(
        validation_dataset
    )

    for key, value in validation_result.items():

        if isinstance(value, float):

            print(
                f"{key:<25}: {value:.4f}"
            )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("                 TEST RESULTS")
    print("=" * 65)

    test_output = trainer.predict(
        test_dataset
    )

    test_metrics = calculate_metrics(
        (
            test_output.predictions,
            test_output.label_ids
        )
    )

    print(
        f"\nAccuracy         : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1         : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro Precision  : "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall     : "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print_per_class_report(
        test_output.predictions,
        test_output.label_ids
    )

    print_confusion_matrix(
        test_output.predictions,
        test_output.label_ids
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print("\n💾 Saving V4 model...")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trainer.save_model(
        str(MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(MODEL_DIR)
    )

    metadata = {

        "model_version": "v4",

        "base_model": MODEL_NAME,

        "labels": LABEL_NAMES,

        "max_length": MAX_LENGTH,

        "training_samples": len(train_rows),

        "validation_samples": len(validation_rows),

        "test_samples": len(test_rows),

        "test_accuracy": test_metrics["accuracy"],

        "test_macro_f1": test_metrics["macro_f1"],

        "test_macro_precision": test_metrics[
            "macro_precision"
        ],

        "test_macro_recall": test_metrics[
            "macro_recall"
        ],
    }

    with open(
        MODEL_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\n✅ V4 model saved to:"
    )

    print(MODEL_DIR)

    # --------------------------------------------------------
    # QUICK TEST
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("                   V4 QUICK TEST")
    print("=" * 65)

    quick_sentences = [

        "I am really happy today!",

        "I feel completely hopeless and sad.",

        "I am so angry about what happened.",

        "I am nervous about tomorrow.",

        "I don't understand what is happening.",

        "Today was a normal day.",

    ]

    encoded = tokenizer(
        quick_sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(model.device)
        for key, value in encoded.items()
    }

    model.eval()

    with torch.no_grad():

        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

    for text, prediction, probability in zip(
        quick_sentences,
        predictions,
        probabilities
    ):

        predicted_label = ID2LABEL[
            prediction.item()
        ]

        confidence = (
            probability[prediction].item()
            * 100
        )

        print(f"\nText: {text}")

        print(
            f"Prediction: {predicted_label}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("              🎉 V4 FINISHED")
    print("=" * 65)

    print(
        f"\nFinal Test Accuracy: "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Final Test Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"\nModel directory:"
    )

    print(MODEL_DIR)

    print("\n")


if __name__ == "__main__":
    main()