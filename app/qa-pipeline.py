import argparse
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import datasets
import numpy as np
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

import utils
from trainer_qa import QuestionAnsweringTrainer


def parse_arguments():
    """Initialize argument parser and return arguments."""
    parser = argparse.ArgumentParser(
        prog="qa-pipeline",
        description="Pipeline for question answering with transformers.",
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=8, help="Batch size (default: 8)."
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="distilbert-base-uncased",
        help="Indicates model checkpoint (default: 'distilbert-base-uncased').",
    )
    parser.add_argument(
        "--cross_validation",
        "-cv",
        type=int,
        default=None,
        help="Set the cross validation fold (default: None).",
    )
    parser.add_argument(
        "--dataset_name",
        "-dn",
        type=str,
        choices=["crawl", "crawl-new-split", "crawl-thin", "email", "expected"],
        default="expected",
        help="Name of the dataset (default: 'expected').",
    )
    parser.add_argument(
        "--dataset_path",
        "-dp",
        default="../data",
        type=str,
        help="Set path to dataset directory (default: '../data').",
    )
    parser.add_argument(
        "--deactivate_map_caching",
        "-dmc",
        action="store_true",
        help="Doesn't use cached files for the dataset functions '.map' and '.filter' "
        "(default: False).",
    )
    parser.add_argument(
        "--doc_stride",
        "-dc",
        type=int,
        default=128,
        help="The authorized overlap between two part of the context when splitting it is needed"
        "(default: 128).",
    )
    parser.add_argument(
        "--epochs", "-e", type=float, default=1.0, help="Number of epochs (default: 1)."
    )
    parser.add_argument(
        "--gpu_device",
        "-gd",
        type=int,
        default=2,
        help="Set the GPU DEVICE number (default: 2).",
    )
    parser.add_argument(
        "--limit_questions",
        "-lq",
        default="",
        type=str,
        help="Limits datasets to specific question, empty string = no limit (default: ''). "
        "Possible values are: '', 'city', 'email', 'fax', 'firstName', 'lastName', 'mobile', "
        "'organization', 'phone', 'position', 'street', 'title', 'website', 'zip'. "
        "If a value is given which isn't in this list, all questions will be selected.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=2e-5,
        help="Set learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--masked_language_modeling_path",
        "-mlmp",
        default=None,
        type=str,
        help="Path to the domain adapted model (default: None).",
    )
    parser.add_argument(
        "--max_answer_length",
        "-mal",
        type=int,
        default=30,
        help="Set the maximum answer length (default: 30).",
    )
    parser.add_argument(
        "--max_examples",
        "-me",
        type=int,
        default=None,
        help="Maximum training instances (default: None).",
    )
    parser.add_argument(
        "--max_length",
        "-ml",
        type=int,
        default=384,
        help="The maximum length of a feature (question and context) (default: 384).",
    )
    parser.add_argument(
        "--n_best_size",
        "-nbs",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer"
        "(default: 20).",
    )
    parser.add_argument(
        "--n_jobs",
        "-nj",
        type=int,
        default=1,
        help="Indicates the number of processors used for computation, especially for `.map`"
        "(default: 1).",
    )
    parser.add_argument(
        "--no_answers",
        "-na",
        action="store_true",
        help="Indicates if no answers are possible (SQuAD v2.0) (default: False).",
    )
    parser.add_argument(
        "--no_cuda",
        "-nc",
        action="store_true",
        help="Indicates if CPU should be used (default: False).",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        "-nsdt",
        type=float,
        default=0.0,
        help="The threshold used to select the null answer: if the best answer has a score"
        "that is less than the score of the null answer minus this threshold, the null answer is"
        "selected for this example. Only useful when `--no_answers` is selected (default: 0.0).",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        "-ood",
        action="store_true",
        help="Overwrite output directory (default: False).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        "-rfc",
        type=str,
        default=None,
        help="Resume training from given checkpoint (default: None).",
    )
    parser.add_argument(
        "--send_extra_message",
        "-sem",
        type=str,
        default="",
        help="Sends an additional custom message within the mail (default: '').",
    )
    parser.add_argument(
        "--send_mail",
        "-sm",
        action="store_true",
        help="Sends an email after the training is done. 'From' and 'to' has to be specified in a "
        "`mail.json` file in 'dataset_path' with the key 'email' and the password with the key "
        "'password' (default: False).",
    )
    parser.add_argument(
        "--squad",
        "-s",
        action="store_true",
        help="Use SQuAD v2.0 dataset (for testing purpose) (default: False).",
    )
    parser.add_argument(
        "--skip_training",
        "-st",
        action="store_true",
        help="Skip the training (default: False).",
    )
    parser.add_argument(
        "--replace_rune",
        "-rr",
        action="store_true",
        help="Replace the Newline Rune ᛉ with a space (default: False).",
    )
    parser.add_argument(
        "--result_path",
        "-rp",
        default="../results",
        type=str,
        help="Path to results (default: '../results')",
    )
    parser.add_argument(
        "--use_only_synthetic_train",
        "-uost",
        action="store_true",
        help="Uses a synthetic train file but without the original testcases, only the synthesized"
        " ones (default: False).",
    )
    parser.add_argument(
        "--use_synthetic_splits",
        "-uss",
        action="store_true",
        help="Uses a synthetic splits (default: False).",
    )
    parser.add_argument(
        "--use_synthetic_train",
        "-ust",
        action="store_true",
        help="Uses a synthetic train file (default: False).",
    )
    parser.add_argument(
        "--use_question_mark",
        "-uqm",
        action="store_true",
        help="Appends a question mark at the end of the question (default: False).",
    )

    return parser.parse_args()


def main(args):

    # ========================== #
    # Hyperparameter / Constants #
    # ========================== #

    ### environment variables ###

    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    ### args parameters ###

    ALLOW_CACHE_MAPPING = not (args.deactivate_map_caching)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    LIMIT_QUESTIONS = args.limit_questions
    MAX_ANSWER_LENGTH = args.max_answer_length
    MAX_EXAMPLES = args.max_examples
    MLM_PATH = args.masked_language_modeling_path
    N_BEST_SIZE = args.n_best_size
    N_JOBS = args.n_jobs
    NO_ANSWERS = args.no_answers
    NO_CUDA = args.no_cuda
    NULL_SCORE_DIFF_THRESHOLD = args.null_score_diff_threshold
    OVERWRITE_OUTPUT_DIR = args.overwrite_output_dir
    REPLACE_RUNE = args.replace_rune
    RESULT_PATH = args.result_path
    RESUME_FROM_CHECKPOINT = args.resume_from_checkpoint
    SKIP_TRAINING = args.skip_training
    USE_SQUAD = args.squad
    USE_SYNTHETIC_SPLITS = args.use_synthetic_splits
    USE_ONLY_SYNTHETIC_TRAIN = args.use_only_synthetic_train
    USE_SYNTHETIC_TRAIN = args.use_synthetic_train
    USE_QUESTION_MARK = args.use_question_mark

    if args.max_length <= 0:
        DOC_STRIDE = 0
        MAX_LENGTH = None
    else:
        DOC_STRIDE = args.doc_stride
        MAX_LENGTH = args.max_length

    # dataset name parameters
    if NO_ANSWERS:
        DATASET_NAME = args.dataset_name + "-na"
    else:
        DATASET_NAME = args.dataset_name

    if args.cross_validation is not None:
        DATASET_DIR_NAME = f"{DATASET_NAME}-cv"
        DATASET_FILE_NAME = f"{DATASET_NAME}-cv{args.cross_validation}"
    else:
        DATASET_DIR_NAME = DATASET_NAME
        DATASET_FILE_NAME = DATASET_NAME

    DATASET_PATH = args.dataset_path

    # use checkpoint
    if MLM_PATH:
        CHECKPOINT = MLM_PATH
        CHECKPOINT_DIR = args.dataset_name + "-mlm"
    else:
        CHECKPOINT = args.checkpoint
        CHECKPOINT_DIR = CHECKPOINT.replace("/", "-")

    ### non args parameters ###
    DATA_COLLATOR = default_data_collator
    PROGRAM_START_TIME = time.time()
    START_DATE = f"{datetime.now():%d.%m.%y}_{datetime.now():%H:%M:%S}"

    # creating model ceckpoint / result directories
    if LIMIT_QUESTIONS:
        EXPANDED_MODEL_OUTPUT_PATH = (
            f"models/{DATASET_DIR_NAME}-{LIMIT_QUESTIONS}/{CHECKPOINT_DIR}"
        )
        RESULT_OUTPUT_DIR = (
            f"{RESULT_PATH}/{DATASET_DIR_NAME}-{LIMIT_QUESTIONS}/{CHECKPOINT_DIR}"
        )
        Path("models").mkdir(parents=True, exist_ok=True)
        Path(f"models/{DATASET_DIR_NAME}-{LIMIT_QUESTIONS}").mkdir(
            parents=True, exist_ok=True
        )
        Path(EXPANDED_MODEL_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)
        Path(RESULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    elif MLM_PATH:
        EXPANDED_MODEL_OUTPUT_PATH = f"models/{CHECKPOINT_DIR}"
        RESULT_OUTPUT_DIR = f"{RESULT_PATH}/{CHECKPOINT_DIR}"
        Path("models").mkdir(parents=True, exist_ok=True)
        Path(f"models/{DATASET_DIR_NAME}").mkdir(parents=True, exist_ok=True)
        Path(EXPANDED_MODEL_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)
        Path(RESULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    else:
        EXPANDED_MODEL_OUTPUT_PATH = f"models/{DATASET_DIR_NAME}/{CHECKPOINT_DIR}"
        RESULT_OUTPUT_DIR = f"{RESULT_PATH}/{DATASET_DIR_NAME}/{CHECKPOINT_DIR}"
        Path("models").mkdir(parents=True, exist_ok=True)
        Path(f"models/{DATASET_DIR_NAME}").mkdir(parents=True, exist_ok=True)
        Path(EXPANDED_MODEL_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)
        Path(RESULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    TRAINING_ARGUMENTS = TrainingArguments(
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        load_best_model_at_end=True,
        logging_strategy="epoch",
        metric_for_best_model="eval_f1",
        num_train_epochs=EPOCHS,
        output_dir=EXPANDED_MODEL_OUTPUT_PATH,
        overwrite_output_dir=OVERWRITE_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_strategy="epoch",
        save_total_limit=2,
        weight_decay=0.01,
    )

    ### model loading ###
    config = AutoConfig.from_pretrained(
        CHECKPOINT, output_attentions=False, output_hidden_states=False
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        CHECKPOINT,
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    # adding new token
    if not REPLACE_RUNE:
        tokenizer.add_tokens("ᛉ")
        model.resize_token_embeddings(len(tokenizer))

    PAD_ON_RIGHT = tokenizer.padding_side == "right"

    ### Logger ###
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = (
        f"logs/questionanswering_{DATASET_DIR_NAME}_{CHECKPOINT_DIR} ({START_DATE}).log"
    )
    logging.basicConfig(level=logging.INFO, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    # check if GPU available
    import torch

    if torch.cuda.is_available():
        logging.info(f"There are {torch.cuda.device_count()} GPU(s) available.")
        logging.info(f"Used GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("No GPU available, using the CPU instead.")

    # =============== #
    # dataset loading #
    # =============== #

    logging.info("Load dataset.")

    if USE_SQUAD:
        train, validation = load_dataset(
            "squad_v2" if NO_ANSWERS else "squad",
            split=[f"train[:{MAX_EXAMPLES}]", f"validation[:{MAX_EXAMPLES}]"],
        )
        raw_datasets = datasets.DatasetDict({"train": train, "validation": validation})
    else:
        base_path = f"{DATASET_PATH}/{DATASET_DIR_NAME}/{DATASET_FILE_NAME}"

        if USE_SYNTHETIC_SPLITS:
            logging.info("Synthetic split files will be used.")
            train_fn = f"{base_path}-synth-train.jsonl"
            val_fn = f"{base_path}-synth-val.jsonl"
            test_fn = f"{base_path}-synth-test.jsonl"
        else:
            train_fn = f"{base_path}-train.jsonl"
            if USE_SYNTHETIC_TRAIN:
                logging.info("Synthetic train file (whole version) will be used.")
                train_fn = f"{base_path}-synth-train-whole.jsonl"
            elif USE_ONLY_SYNTHETIC_TRAIN:
                logging.info("Synthetic train file (only version) will be used.")
                train_fn = f"{base_path}-synth-train-only.jsonl"
            val_fn = f"{base_path}-val.jsonl"
            test_fn = f"{base_path}-test.jsonl"

        raw_datasets = load_dataset(
            "json",
            data_files={
                "train": train_fn,
                "validation": val_fn,
                "test": test_fn,
            },
        )

        # remove fixed column from all splits
        if "fixed" in raw_datasets["train"].column_names:
            raw_datasets = raw_datasets.remove_columns("fixed")

        # limit dataset to MAX_EXAMPLES
        if MAX_EXAMPLES is None:
            MAX_EXAMPLES = max(
                [
                    raw_datasets.num_rows["train"],
                    raw_datasets.num_rows["validation"],
                    raw_datasets.num_rows["test"],
                ]
            )
        else:
            # prevent N_JOBs to be bigger than MAX_EXAMPLES
            if MAX_EXAMPLES < N_JOBS:
                N_JOBS = MAX_EXAMPLES
                logging.info(
                    f"`N_JOBS` is set to '{N_JOBS}' to prevent an `IndexError`."
                )

        raw_datasets = raw_datasets.filter(
            lambda _, i: i < MAX_EXAMPLES,
            load_from_cache_file=ALLOW_CACHE_MAPPING,
            num_proc=N_JOBS,
            with_indices=True,
        )

        if NO_ANSWERS:
            # convert empty answers to a valid format
            raw_datasets = raw_datasets.map(
                utils.convert_empty_answers,
                load_from_cache_file=ALLOW_CACHE_MAPPING,
                num_proc=N_JOBS,
            )

        if LIMIT_QUESTIONS:
            # filter questions to a chosen one
            raw_datasets = raw_datasets.filter(
                lambda example: utils.limit_question(example, LIMIT_QUESTIONS),
                load_from_cache_file=ALLOW_CACHE_MAPPING,
                num_proc=N_JOBS,
            )

        if REPLACE_RUNE:
            # replaces Newline rune ᛉ with a space.
            raw_datasets = raw_datasets.map(
                utils.replace_rune,
                load_from_cache_file=ALLOW_CACHE_MAPPING,
                num_proc=N_JOBS,
            )

        if USE_QUESTION_MARK:
            # add a question mark to the question
            raw_datasets = raw_datasets.map(
                utils.add_question_mark,
                load_from_cache_file=ALLOW_CACHE_MAPPING,
                num_proc=N_JOBS,
            )

    logging.info("Done loading.")
    dataset_sizes_info = f"""
    Datasets sizes
    --------------
    Train:\t{len(raw_datasets['train'])}
    Val:\t{len(raw_datasets['validation'])}
    Test:\t{len(raw_datasets['test'])}\n\n
    """
    logging.info(dataset_sizes_info)

    # ============= #
    # preprocessing #
    # ============= #

    logging.info("Start tokenizing.")

    # remove old train columns because they changed (stride)
    tokenized_train_dataset = raw_datasets["train"].map(
        lambda example: utils.prepare_train_features(
            example,
            tokenizer,
            max_length=MAX_LENGTH,
            doc_stride=DOC_STRIDE,
            pad_on_right=PAD_ON_RIGHT,
        ),
        batched=True,
        num_proc=N_JOBS,
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=ALLOW_CACHE_MAPPING,
        desc="Running tokenizer on train dataset.",
    )

    # remove old validation columns because they changed (stride)
    tokenized_val_dataset = raw_datasets["validation"].map(
        lambda example: utils.prepare_validation_features(
            example,
            tokenizer,
            max_length=MAX_LENGTH,
            doc_stride=DOC_STRIDE,
            pad_on_right=PAD_ON_RIGHT,
        ),
        batched=True,
        num_proc=N_JOBS,
        remove_columns=raw_datasets["validation"].column_names,
        load_from_cache_file=ALLOW_CACHE_MAPPING,
        desc="Running tokenizer on validation dataset.",
    )

    # remove old test columns because they changed (stride)
    tokenized_test_dataset = raw_datasets["test"].map(
        lambda example: utils.prepare_validation_features(
            example,
            tokenizer,
            max_length=MAX_LENGTH,
            doc_stride=DOC_STRIDE,
            pad_on_right=PAD_ON_RIGHT,
        ),
        batched=True,
        num_proc=N_JOBS,
        remove_columns=raw_datasets["test"].column_names,
        load_from_cache_file=ALLOW_CACHE_MAPPING,
        desc="Running tokenizer on test dataset.",
    )
    logging.info("Done tokenizing.")

    # =========== #
    # fine tuning #
    # =========== #

    logging.info("Start training.")

    if NO_CUDA:
        model = model.cpu()

    trainer = QuestionAnsweringTrainer(
        args=TRAINING_ARGUMENTS,
        compute_metrics=utils.compute_metrics,
        data_collator=DATA_COLLATOR,
        eval_dataset=tokenized_val_dataset,
        eval_examples=raw_datasets["validation"],
        max_answer_length=MAX_ANSWER_LENGTH,
        model=model,
        n_best_size=N_BEST_SIZE,
        no_answers=NO_ANSWERS,
        null_score_diff_threshold=NULL_SCORE_DIFF_THRESHOLD,
        output_dir=RESULT_OUTPUT_DIR,
        post_process_function=utils.post_processing_function,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    ### Detecting last checkpoint. ###
    last_train_checkpoint = None
    if os.path.isdir(EXPANDED_MODEL_OUTPUT_PATH) and not OVERWRITE_OUTPUT_DIR:
        last_train_checkpoint = get_last_checkpoint(EXPANDED_MODEL_OUTPUT_PATH)
        if (
            last_train_checkpoint is None
            and len(os.listdir(EXPANDED_MODEL_OUTPUT_PATH)) > 0
        ):
            raise ValueError(
                f"Output directory ({EXPANDED_MODEL_OUTPUT_PATH}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome."
            )
        elif last_train_checkpoint is not None and RESUME_FROM_CHECKPOINT is None:
            logging.info(
                f"Checkpoint detected, resuming training at {last_train_checkpoint}."
                "To avoid this behavior, add `--overwrite_output_dir` to train from scratch."
            )

    train_checkpoint = None
    if RESUME_FROM_CHECKPOINT is not None:
        train_checkpoint = RESUME_FROM_CHECKPOINT
    elif last_train_checkpoint is not None:
        train_checkpoint = last_train_checkpoint

    ### training ###
    if not SKIP_TRAINING:
        train_result = trainer.train(resume_from_checkpoint=train_checkpoint)
        trainer.save_model()
        train_metrics = train_result.metrics
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    ### validation ###
    validation_metrics = trainer.evaluate()
    logging.info(f"Val metrics: {validation_metrics}")
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)
    with open(f"{RESULT_OUTPUT_DIR}/eval_scores.json", "w+") as f:
        json.dump(validation_metrics, f, indent=4, ensure_ascii=False)
    logging.info("Done training")

    # ====================== #
    # prediction on test set #
    # ====================== #

    logging.info("Start predicting.")

    ### predictions ###
    test_results = trainer.predict(tokenized_test_dataset, raw_datasets["test"])
    test_metrics = test_results.metrics
    logging.info(f"Test metrics: {test_metrics}")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    with open(f"{RESULT_OUTPUT_DIR}/test_scores.json", "w+") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)
    logging.info("End predicting.")

    PROGRAM_DURATION = float(time.time() - PROGRAM_START_TIME)
    logging.info(f"Total duration: {int(PROGRAM_DURATION)/60} minute(s).")

    ### sending reminder mail ###
    if args.send_mail:
        import yagmail

        splitted_val_metrics = "\n".join(
            [f"{k}: {v}" for k, v in validation_metrics.items()]
        )
        splitted_test_metrics = "\n".join(
            [f"{k}: {v}" for k, v in test_metrics.items()]
        )

        with open(f"{DATASET_PATH}/mail.json", "r") as f:
            if LIMIT_QUESTIONS:
                if LIMIT_QUESTIONS == "email":
                    LIMIT_QUESTIONS = "mail"
                subject_name = (
                    f"Done: {DATASET_FILE_NAME}({LIMIT_QUESTIONS})-{CHECKPOINT}"
                )
                f" ({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
            else:
                subject_name = f"Done: {DATASET_FILE_NAME}-{CHECKPOINT}"
                f" ({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
            mail = json.load(f)
            yag = yagmail.SMTP(mail["email"], mail["password"])
            contents = (
                f"Training done (duration: {int(PROGRAM_DURATION)/60} minute(s)).\n\n"
            )
            contents += f"{DATASET_FILE_NAME}-{CHECKPOINT}\n\n"
            contents += str(args.send_extra_message)
            contents += "\n\n"
            contents += dataset_sizes_info
            contents += f"{splitted_val_metrics}\n\n"
            contents += f"{splitted_test_metrics}\n\n"
            contents += f"Val metrics: {validation_metrics}\n"
            contents += f"Test metrics: {test_metrics}\n\n"
            contents += f"Args:\n"

            for k, v in vars(args).items():
                contents = contents + f"{k}: {v}\n"

            yag.send(
                to=mail["email"],
                subject=subject_name,
                contents=contents,
            )


if __name__ == "__main__":
    args = parse_arguments()
    try:
        main(args)
    except Exception as error:
        from datetime import datetime

        import yagmail

        print("Error: " + str(error))
        formatted_traceback = traceback.format_exc()
        print(formatted_traceback)

        contents = f"Training failed!\n\n"
        contents += str(args.send_extra_message)
        contents += "\n\n"
        contents += f"{datetime.now():%d.%m.%y}_{datetime.now():%H:%M}\n\n"
        contents += f"Error: {str(error)}\n\n"
        try:
            contents += f"Traceback: {str(formatted_traceback)}"
        except:
            contents += f"Traceback: {str(formatted_traceback[0])}"
        with open(f"{args.dataset_path}/mail.json", "r") as f:
            mail = json.load(f)
            yag = yagmail.SMTP(mail["email"], mail["password"])
            yag.send(
                to=mail["email"],
                subject=f"Training failed! ({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})",
                contents=contents,
            )
