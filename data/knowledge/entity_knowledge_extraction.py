import argparse
import itertools
import json
import logging
import random
from pathlib import Path

from tqdm import tqdm


def parse_arguments():
    """Initialize argument parser and return arguments."""
    parser = argparse.ArgumentParser(
        prog="entity-knowledge-extraction",
        description="Pipeline for the extraction of entity knowledge from lists and dicts.",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="entity-data",
        help="Path to entity folders (default: 'entity-data').",
    )
    parser.add_argument(
        "--train_size", "-ts", type=float, default=0.9, help="Train size (default: 0.9)"
    )
    return parser.parse_args()


def main(args):

    # Logger
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = f"logs/entity-knowledge-extraction.log"
    logging.basicConfig(level=logging.INFO, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    # predefined parameters
    ENTITY_NAMES = {
        "first_name": "firstName",
        "last_name": "lastName",
        "position": "position",
        "title": "title",
        "zips": "zip",
    }
    FOLDER_NAMES = ["first_name", "last_name", "position", "title", "zips"]
    PATH = args.path
    TRAIN_SIZE = args.train_size

    knowledge_sentences = []

    for folder_name in tqdm(FOLDER_NAMES, desc="Folder names"):
        folder_path = Path(f"{PATH}/fixed-entities/{folder_name}")
        for file_path in tqdm(folder_path.iterdir(), desc="JSON files"):
            if file_path.suffix == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if folder_name in ["first_name", "last_name", "position", "title"]:
                        entities = data["words"]
                        for entity in entities:
                            knowledge_sentences.append(
                                [
                                    f"{entity.lower()} is a {ENTITY_NAMES[folder_name]}",
                                    f"{entity.capitalize()} is a {ENTITY_NAMES[folder_name]}",
                                ]
                            )
                    elif folder_name == "zips":
                        for zip, values in data.items():
                            knowledge_sentences.append(
                                [
                                    f"{zip.lower()} is a {ENTITY_NAMES[folder_name]}",
                                    f"{zip.capitalize()} is a {ENTITY_NAMES[folder_name]}",
                                ]
                            )

                            cities = values["city"]
                            for city in cities:
                                knowledge_sentences.append(
                                    [
                                        f"{city.lower()} is a city",
                                        f"{city.capitalize()} is a city",
                                    ]
                                )

    # remove duplicates
    knowledge_sentences.sort()
    unique_knowledge_sentences = list(
        knowledge_sentences
        for knowledge_sentences, _ in itertools.groupby(knowledge_sentences)
    )
    logging.info(f"Number of sentences: {len(unique_knowledge_sentences)*2}")

    random.shuffle(unique_knowledge_sentences)

    train = unique_knowledge_sentences[
        : int(len(unique_knowledge_sentences) * TRAIN_SIZE)
    ]
    val = unique_knowledge_sentences[
        int(len(unique_knowledge_sentences) * TRAIN_SIZE) :
    ]

    train_string = "\n".join(["\n".join(list(set(l))) for l in train])
    val_string = "\n".join(["\n".join(list(set(l))) for l in val])

    logging.info("Write train and test to disk.")
    logging.info("Number of train sentences: " + str(len(train_string.split("\n"))))
    logging.info("Number of val sentences: " + str(len(val_string.split("\n"))))
    with open(f"{PATH}/knowledge_sentences_train.txt", "w+") as f:
        f.write(train_string)
    with open(f"{PATH}/knowledge_sentences_val.txt", "w+") as f:
        f.write(val_string)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
