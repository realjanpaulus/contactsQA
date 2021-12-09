import argparse
import json
import logging
import random
import textwrap
from pathlib import Path

from tqdm import tqdm


def parse_arguments():
    """Initialize argument parser and return arguments."""
    parser = argparse.ArgumentParser(
        prog="type-knowledge-extraction",
        description="Pipeline for extraction knowledge from data types like imprints.",
    )
    parser.add_argument(
        "--exception_ids_path",
        "-eio",
        type=str,
        default="type-data/dataset-ids.txt",
        help="Path to an exception ids txt file (default: 'type-data/dataset-ids.txt').",
    )
    parser.add_argument(
        "--fileextension",
        "-fe",
        type=str,
        default="ndjson",
        help="Input file extension (default: ndjson).",
    )
    parser.add_argument(
        "--input_filename",
        "-ifn",
        type=str,
        default="imprints",
        help="Input file name without extension (default: imprints).",
    )
    parser.add_argument(
        "--input_path",
        "-ip",
        type=str,
        default="type-data",
        help="Input path to data type folders (default: 'type-data').",
    )
    parser.add_argument(
        "--max_data",
        "-md",
        default=10000,
        type=int,
        help="Maximum of testcases to use (default: 100000).",
    )
    parser.add_argument(
        "--max_length",
        "-ml",
        default=None,
        type=int,
        help="Truncate documents to max length. Words will not be splitted so a truncated document"
        " could be shorter (default: None).",
    )
    parser.add_argument(
        "--output_filename",
        "-ofn",
        type=str,
        default="crawl-mlm-raw-imprint",
        help="Output file name without extension (default: 'crawl-mlm-raw-imprint').",
    )
    parser.add_argument(
        "--output_path",
        "-op",
        type=str,
        default="../crawl-mlm-raw-imprint",
        help="Output path for train and val files (default: '../crawl-mlm-raw-imprint').",
    )

    return parser.parse_args()


def main(args):

    EXCEPTION_IDS_PATH = args.exception_ids_path
    FILE_EXTENSION = args.fileextension
    INPUT_FILENAME = args.input_filename
    INPUT_PATH = args.input_path
    MAX_DATA = args.max_data
    MAX_LENGTH = args.max_length
    OUTPUT_FILENAME = args.output_filename
    OUTPUT_PATH = args.output_path

    # Logger
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = f"logs/type-knowledge-extraction.log"
    logging.basicConfig(level=logging.INFO, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    texts = []
    counter = 0

    if EXCEPTION_IDS_PATH:
        with open(EXCEPTION_IDS_PATH, "r") as f:
            exception_ids = f.read().split("\n")[:-1]
            exception_ids = list(map(int, exception_ids))

    with open(Path(f"{INPUT_PATH}/{INPUT_FILENAME}.{FILE_EXTENSION}")) as f:
        if FILE_EXTENSION == "ndjson":
            for line in tqdm(f, desc="Parsing lines"):
                if counter > MAX_DATA:
                    break
                data = json.loads(line)
                if data["chunked_text"] is not None and data["id"] not in exception_ids:
                    texts.append(
                        {
                            "id": data["id"],
                            "text": " ᛉ ".join(data["chunked_text"].splitlines()),
                        }
                    )
                    counter += 1

            logging.info("Parsed lines!")
            random.shuffle(texts)

            if MAX_LENGTH:
                truncated_texts = []
                for element in tqdm(texts, desc="Truncating lines"):
                    truncated_text = textwrap.wrap(element["text"], width=MAX_LENGTH)
                    for idx, text in enumerate(truncated_text):
                        new_idx = int(str(element["id"]) + f"00{idx}")
                        truncated_texts.append({"id": new_idx, "text": text})

                train = truncated_texts[: int(len(truncated_texts) * 0.8)]
                val = truncated_texts[int(len(truncated_texts) * 0.8) :]
            else:
                train = texts[: int(len(texts) * 0.8)]
                val = texts[int(len(texts) * 0.8) :]

            logging.info(f"Count of train texts: {len(train)}")
            logging.info(f"Count of val texts: {len(val)}")
            logging.info("Write json files.")

            Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

            with open(f"{OUTPUT_PATH}/{OUTPUT_FILENAME}-train.jsonl", "w+") as f:
                for element in train:
                    f.write(json.dumps(element, ensure_ascii=False))
                    f.write("\n")

            with open(f"{OUTPUT_PATH}/{OUTPUT_FILENAME}-val.jsonl", "w+") as f:
                for element in val:
                    f.write(json.dumps(element, ensure_ascii=False))
                    f.write("\n")
        else:
            logging.WARNING(f"File extension '{FILE_EXTENSION}' is unknown.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
