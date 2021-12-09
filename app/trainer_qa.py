# coding=utf-8

# copied and slightly modified from:
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py


import collections
import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import IntervalStrategy, PredictionOutput
from transformers.training_args import TrainingArguments


class QuestionAnsweringTrainer(Trainer):
    """A subclass of `Trainer` specific to Question-Answering tasks."""

    def __init__(
        self,
        *args,
        eval_examples=None,
        max_answer_length=30,
        n_best_size=20,
        no_answers=True,
        null_score_diff_threshold=0.0,
        post_process_function=None,
        output_dir="../detailed-results",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self.no_answers = no_answers
        self.null_score_diff_threshold = null_score_diff_threshold
        self.output_dir = output_dir
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics,
                # otherwise we defer to self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                examples=eval_examples,
                features=eval_dataset,
                predictions=output.predictions,
                max_answer_length=self.max_answer_length,
                n_best_size=self.n_best_size,
                no_answers=self.no_answers,
                null_score_diff_threshold=self.null_score_diff_threshold,
                output_dir=self.output_dir,
                prefix=metric_key_prefix,
            )
            metrics = self.compute_metrics(eval_preds, self.no_answers)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(
        self,
        predict_dataset,
        predict_examples,
        ignore_keys=None,
        metric_key_prefix: str = "test",
    ):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics,
                # otherwise we defer to self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(
            examples=predict_examples,
            features=predict_dataset,
            predictions=output.predictions,
            max_answer_length=self.max_answer_length,
            n_best_size=self.n_best_size,
            no_answers=self.no_answers,
            null_score_diff_threshold=self.null_score_diff_threshold,
            output_dir=self.output_dir,
            prefix=metric_key_prefix,
        )
        metrics = self.compute_metrics(predictions, no_answers=self.no_answers)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(
            predictions=predictions.predictions,
            label_ids=predictions.label_ids,
            metrics=metrics,
        )
