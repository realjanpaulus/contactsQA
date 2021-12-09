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
        prog="entity-knowledge-reverse-extraction",
        description="Pipeline for the reverse extraction of entity knowledge from lists and dicts.",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="entity-data",
        help="Path to entity folders (default: 'entity-data').",
    )
    parser.add_argument(
        "--sample_size",
        "-ss",
        type=int,
        default=10000,
        help="Random sample from sentences (default: 10000).",
    )
    parser.add_argument(
        "--train_size", "-ts", type=float, default=0.9, help="Train size (default: 0.9)"
    )
    return parser.parse_args()


def main(args):

    # Logger
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = f"logs/entity-knowledge-reverse-extraction.log"
    logging.basicConfig(level=logging.INFO, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    # predefined parameters
    ENTITY_NAMES = {
        "first_name": "a firstName",
        "last_name": "a lastName",
        "position": "a position",
        "title": "a title",
        "zips": "a zip",
        "phone": "a phone number",
        "mobile": "a mobile number",
        "fax": "a fax number",
        "organization": "an organization",
        "website": "a website",
        "email": "an email",
        "city": "a city",
        "street": "a street",
    }
    FOLDER_NAMES = ["first_name", "last_name", "position", "title", "zips"]
    PATH = args.path
    SAMPLE_SIZE = args.sample_size
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
                            all_entities = [
                                entity_type
                                for entity_type in ENTITY_NAMES.keys()
                                if entity_type != folder_name
                            ]
                            random_entity = random.choice(all_entities)
                            knowledge_sentences.append(
                                [
                                    f"{entity.lower()} is not {ENTITY_NAMES[random_entity]}",
                                    f"{entity.capitalize()} is not {ENTITY_NAMES[random_entity]}",
                                ]
                            )
                    elif folder_name == "zips":
                        for zipcode, values in data.items():

                            all_entities = [
                                entity_type
                                for entity_type in ENTITY_NAMES.keys()
                                if entity_type != folder_name
                            ]
                            random_entity = random.choice(all_entities)

                            knowledge_sentences.append(
                                [
                                    f"{zipcode.lower()} is not {ENTITY_NAMES[random_entity]}",
                                    f"{zipcode.capitalize()} is not {ENTITY_NAMES[random_entity]}",
                                ]
                            )

                            cities = values["city"]
                            for city in cities:
                                all_entities = [
                                    entity_type
                                    for entity_type in ENTITY_NAMES.keys()
                                    if entity_type != folder_name
                                ]
                                random_entity = random.choice(all_entities)
                                knowledge_sentences.append(
                                    [
                                        f"{city.lower()} is not {ENTITY_NAMES[random_entity]}",
                                        f"{city.capitalize()} is not {ENTITY_NAMES[random_entity]}",
                                    ]
                                )

    # remove duplicates
    knowledge_sentences.sort()
    unique_knowledge_sentences = list(
        knowledge_sentences
        for knowledge_sentences, _ in itertools.groupby(knowledge_sentences)
    )
    random.shuffle(unique_knowledge_sentences)
    unique_knowledge_sentences_shorten = unique_knowledge_sentences[:SAMPLE_SIZE]

    logging.info(f"Number of sentences: {len(unique_knowledge_sentences_shorten)}")

    train = unique_knowledge_sentences_shorten[
        : int(len(unique_knowledge_sentences_shorten) * TRAIN_SIZE)
    ]
    val = unique_knowledge_sentences_shorten[
        int(len(unique_knowledge_sentences_shorten) * TRAIN_SIZE) :
    ]

    train_string = "\n".join(["\n".join(list(set(l))) for l in train])
    val_string = "\n".join(["\n".join(list(set(l))) for l in val])

    logging.info("Write train and test to disk.")
    logging.info("Number of train sentences: " + str(len(train_string.split("\n"))))
    logging.info("Number of val sentences: " + str(len(val_string.split("\n"))))
    with open(f"{PATH}/reverse_knowledge_sentences_train.txt", "w+") as f:
        f.write(train_string)
    with open(f"{PATH}/reverse_knowledge_sentences_val.txt", "w+") as f:
        f.write(val_string)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
