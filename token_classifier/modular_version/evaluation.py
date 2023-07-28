
import numpy as np

import evaluate

seqeval = evaluate.load("seqeval")


label_list = []
labels = []

def initialize_labeling(train_dataset):
    label_list = train_dataset.features[f"ner_tags"].feature.names
    labels = [label_list[i] for i in train_dataset[0][f"ner_tags"]]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

