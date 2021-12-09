import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import dataset_utils


def parse_arguments():
    """Initialize argument parser and return arguments."""
    parser = argparse.ArgumentParser(
        prog="create-dataset",
        description="Pipeline for creating question answering datasets.",
    )
    parser.add_argument(
        "--cased",
        "-ca",
        action="store_true",
        help="Use cased instead of lowercase (default: False).",
    )
    parser.add_argument(
        "--cross_validation",
        "-cv",
        type=int,
        default=1,
        help="Number of cross validation folds (default: 1).",
    )
    parser.add_argument(
        "--dataset_name",
        "-dn",
        type=str,
        choices=["crawl", "email", "expected", "grab", "scan"],
        default="expected",
        help="Name of the dataset (default: 'expected').",
    )
    parser.add_argument(
        "--input_path",
        "-ip",
        type=str,
        default=".",
        help="File input path (default: '.').",
    )
    parser.add_argument(
        "--limit_testcases",
        "-lt",
        type=int,
        default=None,
        help="Limit the number of testcases which adds a '-short' flag to the name "
        "(default: None).",
    )
    parser.add_argument(
        "--masked_language_modeling",
        "-mlm",
        action="store_true",
        help="Creates additional train and val json files for "
        "Masked Language Modeling (default: False).",
    )
    parser.add_argument(
        "--no_answers",
        "-na",
        action="store_true",
        help="Indicates if no answers are possible, like in SQuAD v2.0 (default: False).",
    )
    parser.add_argument(
        "--output_path",
        "-op",
        type=str,
        default=".",
        help="File output path (default: '.').",
    )
    parser.add_argument(
        "--random_removal",
        "-rr",
        action="store_true",
        help="Activate random removal of entities (default: False).",
    )
    parser.add_argument(
        "--random_seed",
        "-rs",
        type=int,
        default=42,
        help="Set random seed (default: 42).",
    )
    parser.add_argument(
        "--synthetic_all_splits",
        "-sas",
        action="store_true",
        help="Indicates if all splits or only the train split should contain synthetic testcases "
        "(default: False).",
    )
    parser.add_argument(
        "--synthetic_data_path",
        "-sdp",
        type=str,
        default=None,
        help="Path to the split files for the appending of synthetic data, "
        "e.g. 'synthetic/scan/scan' (default: None).",
    )
    parser.add_argument(
        "--synthetic_external_sleeves_path",
        "-sesp",
        type=str,
        default=None,
        help="Path to external sleeves, e.g. 'knowledge/type-data/raw-sleeves.json' "
        "(default: None).",
    )
    parser.add_argument(
        "--testcase_json_name",
        "-tjn",
        type=str,
        default="testcases",
        help="Testcases json name (default: 'testcases').",
    )

    parser.add_argument(
        "--test_share",
        "-tes",
        type=float,
        default=0.5,
        help="Test share of the remaining instances after 1-train_size (default: 0.5).",
    )
    parser.add_argument(
        "--train_size", "-ts", type=float, default=0.8, help="Train size (default: 0.8)"
    )

    return parser.parse_args()


def main(args):

    # predefined parameters
    CV = args.cross_validation
    DATASET_NAME = args.dataset_name
    INPUT_PATH = args.input_path
    LIMIT_TESTCASES = args.limit_testcases
    LOWERCASE = True
    if args.cased:
        LOWERCASE = False
    MASKED_LANGUAGE_MODELING = args.masked_language_modeling
    NO_ANSWERS = args.no_answers
    OUTPUT_PATH = args.output_path
    RANDOM_REMOVAL = args.random_removal
    RANDOM_SEED = args.random_seed
    SYNTHETIC_ALL_SPLITS = args.synthetic_all_splits
    SYNTHETIC_DATA_PATH = args.synthetic_data_path
    SYNTHETIC_EXTERNAL_SLEEVES_PATH = args.synthetic_external_sleeves_path
    TESTCASES_JSON_NAME = args.testcase_json_name
    TEST_SHARE = args.test_share
    TRAIN_SIZE = args.train_size

    # Logger
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = f"logs/create_dataset_{DATASET_NAME}.log"
    logging.basicConfig(level=logging.INFO, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    # loading whole testcase file
    logging.info("Loading testcases.")
    with open(f"{INPUT_PATH}/{TESTCASES_JSON_NAME}.json") as f:
        testcases = json.load(f)

    logging.info("Transform testcases.")
    transformed_testcases = dataset_utils.testcases_to_squad_format(
        testcases,
        DATASET_NAME,
        allow_empty_answers=NO_ANSWERS,
        lowercase=LOWERCASE,
        random_removal=RANDOM_REMOVAL,
    )

    if NO_ANSWERS:
        logging.info("Allowing no answer testcases.")
        DATASET_NAME = f"{DATASET_NAME}-na"

    if LIMIT_TESTCASES is not None:
        logging.info("Creating shorten testcases.")
        transformed_testcases = transformed_testcases[:LIMIT_TESTCASES]
        DATASET_NAME = f"{DATASET_NAME}-short"

    if CV > 1:
        logging.info(f"Cross validation with {CV} splits.")
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    train, val, _ = dataset_utils.split_save_jsonl(
        transformed_testcases,
        DATASET_NAME,
        OUTPUT_PATH,
        n_splits=CV,
        seed=RANDOM_SEED,
        synthetic_all_splits=SYNTHETIC_ALL_SPLITS,
        synthetic_data_path=SYNTHETIC_DATA_PATH,
        synthetic_external_sleeves_path=SYNTHETIC_EXTERNAL_SLEEVES_PATH,
        test_share=TEST_SHARE,
        train_size=TRAIN_SIZE,
    )

    if MASKED_LANGUAGE_MODELING:
        logging.info("Create Masked Language Modeling (MLM) datasets.")
        Path(f"{OUTPUT_PATH}/{DATASET_NAME}-mlm").mkdir(parents=True, exist_ok=True)

        _ = dataset_utils.save_mlm_jsonl(
            train, f"{OUTPUT_PATH}/{DATASET_NAME}-mlm/{DATASET_NAME}-mlm-train.jsonl"
        )
        _ = dataset_utils.save_mlm_jsonl(
            val, f"{OUTPUT_PATH}/{DATASET_NAME}-mlm/{DATASET_NAME}-mlm-val.jsonl"
        )

    logging.info("Done.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
