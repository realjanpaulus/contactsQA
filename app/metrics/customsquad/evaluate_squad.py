"""
    Modified script,
    original: https://github.com/huggingface/datasets/blob/master/metrics/squad/evaluate.py
"""

import argparse
import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    """Remove extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(s)


def get_precision_recall_f1(prediction, ground_truth, metric_type="f1"):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)

    if metric_type == "precision":
        return precision
    elif metric_type == "recall":
        return recall
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


def f1_score(prediction, ground_truth):
    return get_precision_recall_f1(prediction, ground_truth, metric_type="f1")


def precision_score(prediction, ground_truth):
    return get_precision_recall_f1(prediction, ground_truth, metric_type="precision")


def recall_score(prediction, ground_truth):
    return get_precision_recall_f1(prediction, ground_truth, metric_type="recall")


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = precision = recall = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = (
                        "Unanswered question " + qa["id"] + " will receive score 0."
                    )
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths
                )
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                precision += metric_max_over_ground_truths(
                    precision_score, prediction, ground_truths
                )
                recall += metric_max_over_ground_truths(
                    recall_score, prediction, ground_truths
                )

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    precision = 100.0 * precision / total
    recall = 100.0 * recall / total

    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    expected_version = "1.1"
    parser = argparse.ArgumentParser(
        description="Evaluation for SQuAD " + expected_version
    )
    parser.add_argument("dataset_file", help="Dataset file")
    parser.add_argument("prediction_file", help="Prediction File")
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json["version"] != expected_version:
            print(
                "Evaluation expects v-"
                + expected_version
                + ", but got dataset with v-"
                + dataset_json["version"],
                file=sys.stderr,
            )
        dataset = dataset_json["data"]
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
