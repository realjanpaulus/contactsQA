"""
    Custom SQuAD v2 metric,
    original: https://github.com/huggingface/datasets/blob/master/metrics/squad_v2/squad_v2.py
"""

import datasets

from .evaluate_squad_v2 import (apply_no_ans_threshold, find_all_best_thresh,
                                get_raw_scores, make_eval_dict,
                                make_qid_to_has_ans, merge_eval)

_CITATION = ""

_DESCRIPTION = ""

_KWARGS_DESCRIPTION = ""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SquadV2(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {
                        "id": datasets.Value("string"),
                        "prediction_text": datasets.Value("string"),
                        "no_answer_probability": datasets.Value("float32"),
                    },
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            codebase_urls=[""],
            reference_urls=[""],
        )

    def _compute(self, predictions, references, no_answer_threshold=1.0):
        no_answer_probabilities = dict(
            (p["id"], p["no_answer_probability"]) for p in predictions
        )
        dataset = [{"paragraphs": [{"qas": references}]}]
        predictions = dict((p["id"], p["prediction_text"]) for p in predictions)

        qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

        exact_raw, f1_raw, precision_raw, recall_raw = get_raw_scores(
            dataset, predictions
        )
        exact_thresh = apply_no_ans_threshold(
            exact_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold
        )
        precision_thresh = apply_no_ans_threshold(
            precision_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold
        )
        recall_thresh = apply_no_ans_threshold(
            recall_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold
        )
        f1_thresh = apply_no_ans_threshold(
            f1_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold
        )
        out_eval = make_eval_dict(
            exact_thresh, f1_thresh, precision_thresh, recall_thresh
        )

        if has_ans_qids:
            has_ans_eval = make_eval_dict(
                exact_thresh,
                f1_thresh,
                precision_thresh,
                recall_thresh,
                qid_list=has_ans_qids,
            )
            merge_eval(out_eval, has_ans_eval, "HasAns")
        if no_ans_qids:
            no_ans_eval = make_eval_dict(
                exact_thresh,
                f1_thresh,
                precision_thresh,
                recall_thresh,
                qid_list=no_ans_qids,
            )
            merge_eval(out_eval, no_ans_eval, "NoAns")
        find_all_best_thresh(
            out_eval,
            predictions,
            exact_raw,
            f1_raw,
            precision_raw,
            recall_raw,
            no_answer_probabilities,
            qid_to_has_ans,
        )
        return dict(out_eval)
