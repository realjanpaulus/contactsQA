"""
    Custom SQuAD metric,
    original: https://github.com/huggingface/datasets/blob/master/metrics/squad/squad.py
"""

import datasets

from .evaluate_squad import evaluate

_CITATION = ""

_DESCRIPTION = ""

_KWARGS_DESCRIPTION = ""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Squad(datasets.Metric):
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

    def _compute(self, predictions, references):
        pred_dict = {
            prediction["id"]: prediction["prediction_text"]
            for prediction in predictions
        }
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [
                                    {"text": answer_text}
                                    for answer_text in ref["answers"]["text"]
                                ],
                                "id": ref["id"],
                            }
                            for ref in references
                        ]
                    }
                ]
            }
        ]
        score = evaluate(dataset=dataset, predictions=pred_dict)
        return score
