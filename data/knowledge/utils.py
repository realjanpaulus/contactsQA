# Most of the code is copied and modified from this script:
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py

import collections
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import EvalPrediction


def add_question_mark(example):
    """Adds a question mark to the question."""
    example["question"] = example["question"] + "?"
    return example


def compute_metrics(pred: EvalPrediction, no_answers: bool = True):
    """Compute all relevant QA metrics"""
    metric = load_metric(
        "./metrics/customsquadv2" if no_answers else "./metrics/customsquad"
    )
    return metric.compute(predictions=pred.predictions, references=pred.label_ids)


def convert_empty_answers(example):
    """Convert empty example (-1) to the empty list form (expected by Huggingface)."""

    if example["answers"]["answer_start"][0] < 0:
        example["answers"] = {"text": [], "answer_start": []}
    return example


def limit_question(example, question):
    """Only returns example if question equal to input question."""
    possible_questions = [
        "",
        "city",
        "email",
        "fax",
        "firstName",
        "lastName",
        "mobile",
        "organization",
        "phone",
        "poBox",
        "position",
        "street",
        "street2",
        "title",
        "vat",
        "website",
        "zip",
    ]
    if question not in possible_questions:
        logging.info(
            f"Question '{question}' is not valid, no filter will be applied!!!"
        )
        return True
    else:
        return example["question"] == question


def replace_rune(example):
    """Replace Newline rune ᛉ with a space."""
    example["context"] = example["context"].replace("ᛉ", " ")
    return example


def prepare_train_features(
    example, tokenizer, max_length=384, doc_stride=128, pad_on_right=True
):
    """Tokenize the example with truncation and padding, but keep the overflows using a stride.
    This results in one example possible giving several features when a context is long,
    each of those features having a context that overlaps a bit the context of the
    previous feature.
    """
    tokenized_example = tokenizer(
        example["question" if pad_on_right else "context"],
        example["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # mapping from character position to original context
    offset_mapping = tokenized_example.pop("offset_mapping")
    # map from a feature to its corresponding example
    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")

    tokenized_example["start_positions"] = []
    tokenized_example["end_positions"] = []

    # impossible answers are labeled with the index of the CLS token
    # all datasets (train, val, test) are tokenized here
    for i, offsets in enumerate(offset_mapping):

        input_ids = tokenized_example["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_example.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = example["answers"][sample_index]

        # If no answers are given, set the `cls_index` as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_example["start_positions"].append(cls_index)
            tokenized_example["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span
            # (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_example["start_positions"].append(cls_index)
                tokenized_example["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of
                # the answer. NOTE: normally it is possible to go AFTER the last offset
                # if the answer is the last word (edge case) but here the last word is always
                # the rune ᛉ.
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_example["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_example["end_positions"].append(token_end_index + 1)

    return tokenized_example


def prepare_validation_features(
    example, tokenizer, max_length=384, doc_stride=128, pad_on_right=True
):
    """Tokenize example with truncation and maybe padding, but keep the overflows
    using a stride. This results in one example possible giving several features
    when a context is long, each of those features having a
    context that overlaps a bit the context of the previous feature.
    """
    tokenized_example = tokenizer(
        example["question" if pad_on_right else "context"],
        example["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # map from a feature to its corresponding example
    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_example["example_id"] = []

    for i in range(len(tokenized_example["input_ids"])):
        # Grab the sequence corresponding to that example
        # (to know what is the context and what is the question).
        sequence_ids = tokenized_example.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans,
        # this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_example["example_id"].append(example["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context
        # so it's easy to determine if a token position is part of the context or not.
        tokenized_example["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_example["offset_mapping"][i])
        ]

    return tokenized_example


def postprocess_qa_predictions(
    examples: dict,
    features: dict,
    predictions: tuple[np.ndarray, np.ndarray],
    max_answer_length: int = 30,
    n_best_size: int = 20,
    no_answers: bool = True,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = "../detailed-results",
    prefix: Optional[str] = None,
) -> collections.OrderedDict:
    """Post-processes the predictions of a question-answering model to convert them to answers
        that are substrings of the original contexts.
        This is the base postprocessing functions for models that only return start and end logits.

    Parameters
    ----------
        examples : dict
            The non-preprocessed dataset.
        features : dict
            The processed dataset.
        predictions : tuple[np.ndarray, np.ndarray]
            The predictions of the model: two arrays containing the start logits
            and the end logits respectively. Its first dimension must match the number of elements
            of `features`.
        max_answer_length : int, default=20
            The maximum length of an answer that can be generated. This is needed because the
            start and end predictions are not conditioned on one another.
        n_best_size : int, default=20
            The total number of n-best predictions to generate when looking for an answer.
        no_answers : bool, default=True
            Whether or not the underlying dataset contains examples with no answers.
        null_score_diff_threshold : float, default=0.0
            The threshold used to select the null answer: if the best answer has a score that is
            less than the score of the null answer minus this threshold, the null answer is
            selected for this example (note that the score of the null answer for an example
            giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact
            they `want` to predict a null answer). Only useful when `no_answers=True`.
        output_dir : str, default='../detailed-results'
            If provided, the dictionaries of predictions, n_best predictions
            (with their scores and logits) and, if `no_answers=True`, the dictionary of the scores
            differences between best and null answers, are saved in `output_dir`.
        prefix : str, default=None
            If provided, the dictionaries mentioned above are saved with `prefix`
            added to their names.

    Returns
    -------
        all_predictions : collections.OrderedDict
            All predictions as OrderedDict.
    """
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if no_answers:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logging.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # Loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions
            # in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            # pylint disable=unsubscriptable-object
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers,
                    # either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0
                    # or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if no_answers:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if no_answers and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # Compute the softmax of all scores (we do it with numpy to stay independent
        # from torch/tf in this file, using the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not no_answers:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0

            if predictions:
                while predictions[i]["text"] == "":
                    i += 1
                    if i >= len(predictions):
                        i = i - 1
                        break
                best_non_null_pred = predictions[i]
            else:
                best_non_null_pred = {
                    "text": "ERROR",
                    "start_logit": 0.0,
                    "end_logit": 0.0,
                    "score": 0.0,
                }

            # Then we compare to the null prediction using the threshold.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(
                score_diff
            )  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"{prefix}_predictions.json",
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"{prefix}_nbest_predictions.json",
        )
        if no_answers:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"{prefix}_null_odds.json",
            )

        logging.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )
        logging.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(
                json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
            )
        if no_answers:
            logging.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(
                    json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n"
                )

    return all_predictions


def post_processing_function(
    examples: dict,
    features: dict,
    predictions: tuple[np.ndarray, np.ndarray],
    max_answer_length: int = 30,
    n_best_size: int = 20,
    no_answers: bool = True,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = "../detailed-results",
    prefix: Optional[str] = None,
):
    """Post-processing: we match the start logits and end logits to answers
        in the original context.

    Parameters
    ----------
        examples : dict
            The non-preprocessed dataset.
        features : dict
            The processed dataset.
        predictions : tuple[np.ndarray, np.ndarray]
            The predictions of the model: two arrays containing the start logits
            and the end logits respectively. Its first dimension must match the number of elements
            of `features`.
        max_answer_length : int, default=20
            The maximum length of an answer that can be generated. This is needed because the
            start and end predictions are not conditioned on one another.
        n_best_size : int, default=20
            The total number of n-best predictions to generate when looking for an answer.
        no_answers : bool, default=True
            Whether or not the underlying dataset contains examples with no answers.
        null_score_diff_threshold : float, default=0.0
            The threshold used to select the null answer: if the best answer has a score that is
            less than the score of the null answer minus this threshold, the null answer is
            selected for this example (note that the score of the null answer for an example
            giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact
            they `want` to predict a null answer). Only useful when `no_answers=True`.
        output_dir : str, default='../detailed-results'
            If provided, the dictionaries of predictions, n_best predictions
            (with their scores and logits) and, if `no_answers=True`, the dictionary of the scores
            differences between best and null answers, are saved in `output_dir`.
        prefix : str, default=None
            If provided, the dictionaries mentioned above are saved with `prefix`
            added to their names.

    Returns
    -------
         : EvalPrediction
            Predictions.
    """

    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=max_answer_length,
        n_best_size=n_best_size,
        no_answers=no_answers,
        null_score_diff_threshold=null_score_diff_threshold,
        output_dir=output_dir,
        prefix=prefix,
    )
    # Format the result to the format the metric expects.
    if no_answers:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
            for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
