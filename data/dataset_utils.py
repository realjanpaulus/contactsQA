import difflib
import itertools
import json
import logging
import random
import re
from pathlib import Path

import pandas as pd
import plainhtml
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

import text_preprocessing

# ======================== #
# general helper functions #
# ======================== #


def find_fixed_input(fixed_input: str, text: str) -> tuple[bool, str, str]:
    """Trying to find fixed input in a string by applying different methods.

    Examples
    --------
    >>> find_fixed_input('max mustermann", "max | mustermann')
    (True, 'max mustermann', 'max mustermann')

    >>> find_fixed_input('max - mustermann', 'max – mustermann')
    (True, 'max - mustermann', 'max - mustermann')

    Parameters
    ----------
    fixed_input : str
        String to find in 'text'.
    text : str
        String which may contain the string 'fixed_input'.

    Returns
    -------
    tuple[bool, str, str]
        Returns a boolean if 'fixed_input' was found in the text, the fixed input and the text
        which could be modified.
    """

    extract_entities = False

    if fixed_input in text:
        extract_entities = True
    elif fixed_input in text.replace("|", " "):
        extract_entities = True
        text = text.replace("|", " ")
    elif " ".join(fixed_input.split()) in " ".join(text.replace("|", " ").split()):
        extract_entities = True
        fixed_input = " ".join(fixed_input.split())
        text = " ".join(text.replace("|", " ").split())
    elif fixed_input in " ".join(text.split()):
        extract_entities = True
        text = " ".join(text.split())
    elif fixed_input in text.replace("–", "-"):
        extract_entities = True
        text = text.replace("–", "-")

    return extract_entities, fixed_input, text


def get_answer_positions(answer: str, text: str) -> tuple[int, int]:
    """Get the start and end position of an answer.

    Examples
    --------
    >>> get_answer_positions('name', 'how is your name')
    (12, 16)

    Parameters
    ----------
    answer : str
        Answer string.
    text : str
        Context string which could contain the answer.

    Returns
    -------
    tuple[int, int]
        Returns the answer start and the answer end.
    """

    if answer:
        answer_start = text.find(answer)

        # for non-existing answers in the text
        if answer_start < 0:
            answer_end = -1
        else:
            answer_end = answer_start + len(str(answer))
    else:
        answer_start, answer_end = -1, -1

    return answer_start, answer_end


def get_original_entity(fixed_input: str, tokens: list[str]) -> str:
    """Get the original spelling/notation of the entity (represented with a list of tokens) within
        the original text (= 'fixed_input').

    Examples
    --------
    >>> fixed_input = 'max ᛉ mustermann ᛉ +49 0123 456-7 89'
    >>> tokens = ['+', '49', '0123', '456', '7', '89']
    >>> get_original_entity(fixed_input, tokens)
    '+49 0123 456-7 89'

    Parameters
    ----------
    fixed_input : str
        The original text.
    tokens : list[str]
        List of subtokens which represents an entity together.

    Returns
    -------
    str
        Entity which exists in the 'fixed_input' and contains all elements of 'tokens'.
    """

    tokens = [" ᛉ " if token == "snapaddynewline" else token for token in tokens]

    start_token = tokens[0]
    end_token = tokens[-1]

    def findall(p, s):
        """Yields all the positions of the pattern p in the string s."""
        i = s.find(p)
        while i != -1:
            yield i
            i = s.find(p, i + 1)

    start_indices = [idx for idx in findall(start_token, fixed_input)]
    end_indices = [idx + len(end_token) for idx in findall(end_token, fixed_input)]

    # Step 1: Find all possible combinations of substrings which contain the entity
    possible_combinations = []
    for si in start_indices:
        for ei in end_indices:
            if si < ei:
                possible_combinations.append((si, ei))
    possible_combinations_strings = [
        fixed_input[t[0] : t[1]] for t in possible_combinations
    ]

    # Step 2: Check if all tokens are inside the substrings
    lengths = []
    for comb in possible_combinations_strings:
        token_count = 0
        for token in tokens:
            if token in comb:
                token_count += 1

        if token_count == len(tokens) and comb in fixed_input:
            lengths.append((len(comb), comb))

    # Step 3: Sort the valid substrings by length and select the shortest
    sorted_lengths = sorted(lengths, key=lambda x: (len(x), x))
    if sorted_lengths:
        return sorted_lengths[0][1]
    else:
        # fallback
        return " ".join(tokens)


def group_tags(tags: list[dict[str, str]]) -> list[list[dict]]:
    """Group a list of dicts with the keys 'tag' and 'token' by the same following tags,
        keeping the order.

    Notes
    -----
    The tag 'phoneFax' will be replaced by the tag 'phone'.

    Examples
    --------
    >>> tags = [
        {'tag': 'organization', 'token': 'Imagine'},
        {'tag': 'organization', 'token': 'Company'},
        {'tag': 'word', 'token': 'SNAPADDYNEWLINE'},
        {'tag': 'firstName', 'token': 'Max'},
        {'tag': 'lastName', 'token': 'Mustermann'},
    ]
    >>> group_tags(tags)
    [
        [{'tag': 'organization', 'token': 'Imagine'}, {'tag': 'organization', 'token': 'Company'}],
        [{'tag': 'word', 'token': 'SNAPADDYNEWLINE'}],
        [{'tag': 'firstName', 'token': 'Max'}],
        [{'tag': 'lastName', 'token': 'Mustermann'}]
    ]


    Parameters
    ----------
    tags : list[dict[str, str]]
        List of dicts with the keys 'tag' and 'token'.

    Returns
    -------
    list[list[dict]]
        List of lists of dicts where every inner list represents a group of the same tags.
    """

    entities_order = [dic["tag"] for dic in tags]
    entities_order = [
        "phone" if entity == "phoneFax" else entity for entity in entities_order
    ]
    tag_groups = [list(k) for a, k in itertools.groupby(entities_order)]

    # map grouped tags to original tag/token dict format
    groups = []
    tag_counter = 0
    for tag_group in tag_groups:
        match_group = []
        for _ in tag_group:
            match_group.append(tags[tag_counter])
            tag_counter += 1
        groups.append(match_group)
    return groups


def split_save_jsonl(
    testcases: list[dict],
    dataset_name: str,
    output_dir: str,
    id_name: str = "orig_id",
    n_splits: int = 1,
    seed: int = 42,
    synthetic_all_splits: bool = False,
    synthetic_data_path: str = None,
    synthetic_external_sleeves_path: str = None,
    test_share: float = 0.5,
    train_size: float = 0.8,
) -> tuple[list, list, list]:
    """Split testcases to train, validation and test sets and write them to disk
        in the JSONL format.

    Parameters
    ----------
    testcases : list[dict]
        List of testcases.
    dataset_name : str
        Name of the dataset.
    output_dir : str
        Output directory as path.
    id_name : str, optional
        Name for the id column, by default "orig_id".
    n_splits : int, optional
        Number of splits, by default 1.
    seed: int, optional
        Random seed, by default 42.
    synthetic_all_splits: bool, optional
        Indicates if all splits or only the train split should contain synthetic testcases,
        by default False
    synthetic_data_path: str, optional
        Path to the split files for the appending of synthetic data, e.g. 'synthetic/scan/scan',
        by default None.
    synthetic_external_sleeves_path : str, optional
        Path to external sleeves, e.g. 'knowledge/type-data/raw-sleeves.json', by default None.
    test_share : float, optional
        Test share of the remaining instances after 1-train_size, by default 0.5.
    train_size : float, optional
        Training size, by default 0.8.

    Returns
    -------
    tuple[list, list, list]
        Train, validation and test set.
    """

    folds = shuffle_split(
        testcases,
        id_name=id_name,
        n_splits=n_splits,
        seed=seed,
        test_share=test_share,
        train_size=train_size,
    )

    dataset_dir_name = dataset_name
    dataset_file_name = dataset_name

    for n in range(n_splits):

        # different name for cross validation splits
        if n_splits > 1:
            dataset_dir_name = f"{dataset_name}-cv"
            dataset_file_name = f"{dataset_name}-cv{n}"

        train_inds = folds[n]["train_ids"]
        val_inds = folds[n]["val_ids"]
        test_inds = folds[n]["test_ids"]

        train = [element for element in testcases if element[id_name] in train_inds]
        val = [element for element in testcases if element[id_name] in val_inds]
        test = [element for element in testcases if element[id_name] in test_inds]

        logging.info(
            f"""
            Counts of split {n+1}
            -------------------
            Count of train ids / instances: {len(train_inds)} / {len(train)}
            Count of val ids / instances: {len(val_inds)} / {len(val)}
            Count of test ids / instances: {len(test_inds)} / {len(test)}
            """
        )

        Path(f"{output_dir}/{dataset_dir_name}").mkdir(parents=True, exist_ok=True)

        if synthetic_data_path:
            if synthetic_all_splits:
                if synthetic_external_sleeves_path:
                    logging.WARNING(
                        "Using external sleeves for all splits is not supported!"
                    )
                (
                    synthetic_train,
                    synthetic_val,
                    synthetic_test,
                ) = create_synthetic_testcases(
                    train, val, test, path_to_base_splits=synthetic_data_path
                )

                extended_train = train.copy()
                extended_train.extend(synthetic_train)
                extended_train_inds = set(
                    [element[id_name] for element in extended_train]
                )
                logging.info(f"Adding synthetic data from '{synthetic_data_path}'.")
                logging.info(
                    f"Count of new synthetic train ids / instances: {len(extended_train_inds)} /"
                    + f" {len(extended_train)}'."
                )

                extended_val = val.copy()
                extended_val.extend(synthetic_val)
                extended_val_inds = set([element[id_name] for element in extended_val])
                logging.info(f"Adding synthetic data from '{synthetic_data_path}'.")
                logging.info(
                    f"Count of new synthetic val ids / instances: {len(extended_val_inds)} /"
                    + f" {len(extended_val)}'."
                )

                extended_test = test.copy()
                extended_test.extend(synthetic_test)
                extended_test_inds = set(
                    [element[id_name] for element in extended_test]
                )
                logging.info(f"Adding synthetic data from '{synthetic_data_path}'.")
                logging.info(
                    f"Count of new synthetic test ids / instances: {len(extended_test_inds)} /"
                    + f" {len(extended_test)}'."
                )

                # save synth train
                with open(
                    f"{output_dir}/{dataset_dir_name}/{dataset_file_name}-synth-train.jsonl",
                    "w",
                ) as f:
                    for element in extended_train:
                        f.write(json.dumps(element, ensure_ascii=False))
                        f.write("\n")

                # save synth val
                with open(
                    f"{output_dir}/{dataset_dir_name}/{dataset_file_name}-synth-val.jsonl",
                    "w",
                ) as f:
                    for element in extended_val:
                        f.write(json.dumps(element, ensure_ascii=False))
                        f.write("\n")

                # save synth test
                with open(
                    f"{output_dir}/{dataset_dir_name}/{dataset_file_name}-synth-test.jsonl",
                    "w",
                ) as f:
                    for element in extended_test:
                        f.write(json.dumps(element, ensure_ascii=False))
                        f.write("\n")
            else:
                synthetic_train = create_synthetic_train_testcases(
                    train,
                    path_to_base_splits=synthetic_data_path,
                    synthetic_external_sleeves_path=synthetic_external_sleeves_path,
                )
                extended_train = train.copy()
                extended_train.extend(synthetic_train)
                extended_train_inds = set(
                    [element[id_name] for element in extended_train]
                )
                logging.info(f"Adding synthetic data from '{synthetic_data_path}'.")
                logging.info(
                    f"Count of new synthetic train ids / instances (including the original train):"
                    + f" {len(extended_train_inds)} / {len(extended_train)}'."
                )

            # save synth train (whole version)
            ad = ""
            if synthetic_external_sleeves_path:
                ad = "-raw"
            with open(
                f"{output_dir}/{dataset_dir_name}/{dataset_file_name}-synth-train-whole{ad}.jsonl",
                "w",
            ) as f:
                for element in extended_train:
                    f.write(json.dumps(element, ensure_ascii=False))
                    f.write("\n")

        with open(
            f"{output_dir}/{dataset_dir_name}/{dataset_file_name}-train.jsonl", "w"
        ) as f:
            for element in train:
                f.write(json.dumps(element, ensure_ascii=False))
                f.write("\n")

        with open(
            f"{output_dir}/{dataset_dir_name}/{dataset_file_name}-val.jsonl", "w"
        ) as f:
            for element in val:
                f.write(json.dumps(element, ensure_ascii=False))
                f.write("\n")

        with open(
            f"{output_dir}/{dataset_dir_name}/{dataset_file_name}-test.jsonl", "w"
        ) as f:
            for element in test:
                f.write(json.dumps(element, ensure_ascii=False))
                f.write("\n")

    # return last train, val and test split
    return train, val, test


def save_mlm_jsonl(
    testcases: list[dict], output_dir: str, id_name: str = "orig_id"
) -> list[dict]:
    """Extract unique instances from a testcase list. Everything but the ID (given by 'orig_id')
        and the 'context' will dropped. Testcases will be written to the disk as JSON, JSONL or
        NDJSON file.

    Parameters
    ----------
    testcases : list[dict]
        List of testcases.
    output_dir : str
        Output directory as path.
    id_name : str, optional
        Name for the id column, by default "orig_id".

    Returns
    -------
    list[dict]
        Shortened testcases.
    """
    visited_ids = []
    output = []
    for testcase in testcases:
        if testcase[id_name] not in visited_ids:
            output.append({"id": testcase[id_name], "text": testcase["context"]})
            visited_ids.append(testcase[id_name])

    with open(output_dir, "w+") as f:
        for element in output:
            f.write(json.dumps(element, ensure_ascii=False))
            f.write("\n")

    return testcases


def shuffle_split(
    testcases,
    id_name: str = "orig_id",
    n_splits: int = 1,
    seed: int = 42,
    test_share: float = 0.5,
    train_size: float = 0.8,
) -> dict[int, dict]:
    """Split testcases into train, val and test sets. Val and test set sizes will calculated
        as follows: (1 - 'train_size')/2. No testcase will be split across the sets.
        All testcases will be shuffled. Cross validation can be created by 'n_splits' > 1.

    Parameters
    ----------
    testcases : list[dict]
        List of testcases.
    id_name : str, optional
        Name for the id column, by default "orig_id".
    n_splits : int, optional
        Number of splits, by default 1.
    seed : int, optional
        Random seed, by default 42.
    test_share : float, optional
        Test share of the remaining instances after 1-train_size, by default 0.5.
    train_size : float, optional
        Training size, by default 0.8.

    Returns
    -------
    dict[int, dict]
        IDs of the train, val and test set for every fold.
    """
    random.seed(seed)

    df = pd.DataFrame(testcases)
    unique_df = df.drop_duplicates(subset=id_name)
    unique_df = unique_df.reset_index()

    folds = {}
    for idx, (train_indices, val_test_indices) in enumerate(
        ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed).split(
            unique_df
        )
    ):
        train_ids = unique_df[unique_df.index.isin(train_indices)].orig_id.tolist()
        val_test_ids = unique_df[
            unique_df.index.isin(val_test_indices)
        ].orig_id.tolist()

        val_share = 1 - test_share

        folds[idx] = {
            "train_ids": train_ids,
            "val_ids": val_test_ids[: int(len(val_test_ids) * val_share)],
            "test_ids": val_test_ids[int(len(val_test_ids) * val_share) :],
        }

    return folds


# ============================ #
# synthetic testcase functions #
# ============================ #


def combine_splits(path: str) -> dict:
    """Combine dataset splits to one dict of a list of testcases, identified by the id

    Parameters
    ----------
    path : str
        Path to the splits.

    Returns
    -------
    dict
        Dict of a list of testcases, identified by the id
    """
    # load into one df
    train = pd.read_json(path_or_buf=f"{path}-train.jsonl", lines=True)
    val = pd.read_json(path_or_buf=f"{path}-val.jsonl", lines=True)
    test = pd.read_json(path_or_buf=f"{path}-test.jsonl", lines=True)
    df = train.append(val).append(test).reset_index().drop("index", axis=1)

    # convert to dict of lists of instances
    return df_to_dictlist(df)


def df_to_dictlist(df: pd.DataFrame) -> dict:
    """Convert DataFrame to a dict of a list of testcases, identified by the id

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with testcases.

    Returns
    -------
    dict
        Dict of list of testcases, identified by the id
    """
    df_dict = df.to_dict(orient="records")
    new_df_dict = {}
    for row in df_dict:
        if row["orig_id"] in new_df_dict:
            new_df_dict[row["orig_id"]].append(row)
        else:
            new_df_dict[row["orig_id"]] = [row]

    return new_df_dict


def get_sleeves(df: pd.DataFrame) -> list:
    """Get data sleeves by replacing the contact block in an instance with a placeholder.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with testcases.

    Returns
    -------
    list
        Pool of data sleeves.
    """
    sleeves = []
    for _, row in df.drop_duplicates(subset="orig_id").iterrows():
        sleeves.append(
            row["context"].replace(row["fixed"], "EMPTY-PLACEHOLDER-FOR-FIXED-TEXT")
        )
    return sleeves


def create_synthetic_testcases(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    path_to_base_splits: str = "synthetic/scan/scan",
) -> tuple:
    """Create synthetic testcases for every split (train, val and test) of a dataset.

    Parameters
    ----------
    train : pd.DataFrame
        Train DataFrame.
    val : pd.DataFrame
        Validation DataFrame.
    test : pd.DataFrame
        Test DataFrame.
    path_to_base_splits : str, optional
        Path to the base splits, by default "synthetic/scan/scan"

    Returns
    -------
    tuple
        Synthetic train, validation and test set.
    """
    train_sleeves = get_sleeves(pd.DataFrame(train))
    val_sleeves = get_sleeves(pd.DataFrame(val))
    test_sleeves = get_sleeves(pd.DataFrame(test))

    train_synth_base = df_to_dictlist(
        pd.read_json(path_or_buf=f"{path_to_base_splits}-train.jsonl", lines=True)
    )
    val_synth_base = df_to_dictlist(
        pd.read_json(path_or_buf=f"{path_to_base_splits}-val.jsonl", lines=True)
    )
    test_synth_base = df_to_dictlist(
        pd.read_json(path_or_buf=f"{path_to_base_splits}-test.jsonl", lines=True)
    )

    return (
        encase_synth_with_sleeve(train_synth_base, train_sleeves),
        encase_synth_with_sleeve(val_synth_base, val_sleeves),
        encase_synth_with_sleeve(test_synth_base, test_sleeves),
    )


def create_synthetic_train_testcases(
    train: pd.DataFrame,
    path_to_base_splits: str = "synthetic/scan/scan",
    synthetic_external_sleeves_path: str = None,
) -> list:
    """Create synthetic train testcases.

    Parameters
    ----------
    train : pd.DataFrame
        Train DataFrame.
    path_to_base_splits : str, optional
        Path to the base splits, by default "synthetic/scan/scan"
    synthetic_external_sleeves_path : str, optional
        Path to external sleeves, e.g. 'knowledge/type-data/raw-sleeves.json', by default None.

    Returns
    -------
    list
        List of synthetic train testcases.
    """
    if synthetic_external_sleeves_path:
        with open(synthetic_external_sleeves_path, "r") as f:
            dict_sleeves = json.load(f)
        sleeves = [d["text"] for d in dict_sleeves]
    else:
        sleeves = get_sleeves(pd.DataFrame(train))
    synth_base = combine_splits(path_to_base_splits)
    return encase_synth_with_sleeve(synth_base, sleeves)


def encase_synth_with_sleeve(synth_base: list[dict], sleeves: list) -> list:
    """Encase a an adress block with a random sleeve.

    Parameters
    ----------
    synth_base : list[dict]
        List of testcases instances
    sleeves : list
        List of data sleeves (= texts with a placeholder and without an main adress block)

    Returns
    -------
    list
        Encased synthetic testcases.
    """
    synthetic_testcases = []

    for _, instances in synth_base.items():
        address = instances[0]["context"]
        # grab random imprint sleeve and encase synth base
        sleeve = random.choice(sleeves)
        placeholder_index = sleeve.find("EMPTY-PLACEHOLDER-FOR-FIXED-TEXT")
        new_context = sleeve.replace("EMPTY-PLACEHOLDER-FOR-FIXED-TEXT", address)

        new_instances = []
        for row in instances:
            row["context"] = new_context
            row["answers"] = {
                "text": row["answers"]["text"],
                "answer_start": [
                    i + placeholder_index for i in row["answers"]["answer_start"]
                ],
            }
            new_instances.append(row)

        synthetic_testcases.extend(new_instances)

    return synthetic_testcases


# ========================= #
# testcases to squad format #
# ========================= #


def get_entities_and_text(
    testcase: dict,
    dataset_name: str,
    lowercase: bool = True,
    random_removal: bool = False,
) -> tuple[dict, str]:
    """Extracts questions and answers in form of entities from a testcase. Depending on
        'dataset_name', a different method will be applied.

    Examples
    --------
    >>> get_entities_and_text(testcase, 'expected')
    (
        {
            'organization': {'text': ['imagine company'], 'answer_start': [0]},
            'firstName': {'text': ['max'], 'answer_start': [18]},
            'lastName': {'text': ['mustermann'], 'answer_start': [22]},
            'zip': {'text': ['97070'], 'answer_start': [35]},
            'city': {'text': ['würzburg'], 'answer_start': [-1]},
        },
        'imagine company ᛉ max mustermann ᛉ 97070'
    )

    Parameters
    ----------
    testcase : dict
        Testcase as dict.
    dataset_name : str
        Name of the dataset.
    lowercase : bool
        Converts text to lowercase, by default True.
    random_removal : bool, optional
        Removes ~33% of the entities from the context, by default False.

    Returns
    -------
    tuple[dict, str]
        Entities/Answers in the SQuAD format and context.
    """

    if random_removal:
        probability = 0.33
    else:
        probability = -1

    # general fixed input preprocessing
    fixed_input = testcase["fixed_input"]["text"]
    fixed_input = text_preprocessing.preprocessing(fixed_input, lowercase=lowercase)

    # find context in html and get entities from tags
    if dataset_name.lower().startswith("crawl"):
        # html to text & preprocessing
        text = testcase["raw_input"]["text"]
        text = plainhtml.extract_text(text)
        text = text_preprocessing.preprocessing(text, lowercase=lowercase)

        # find fixed input in extracted html text
        extract_entities, fixed_input, text = find_fixed_input(fixed_input, text)

        if extract_entities:
            fixed_input_start = text.find(fixed_input)
            fixed_input_end = text.find(fixed_input) + len(fixed_input)

            new_fixed_input = text[fixed_input_start:fixed_input_end]

            # group same token entities together
            groups = group_tags(testcase["tags"])

            entities = {}

            for group in groups:
                if lowercase:
                    tokens = [element["token"].lower() for element in group]
                else:
                    tokens = [element["token"] for element in group]
                entity_type = group[0]["tag"]
                entity_content = get_original_entity(new_fixed_input, tokens)
                entity_start = new_fixed_input.find(entity_content) + fixed_input_start

                if entity_content != "word":
                    if entity_start <= fixed_input_end:
                        if random.random() > probability:

                            # remove last empty string
                            if entity_content[-1] == " ":
                                entity_content = entity_content[:-1]

                            entities[entity_type] = {
                                "text": [entity_content],
                                "answer_start": [entity_start],
                            }
                        else:
                            # remove part from text
                            text = (
                                text[:entity_start]
                                + text[entity_start + len(entity_content) :]
                            )
                            entities[entity_type] = {
                                "text": ["EMPTY"],
                                "answer_start": [-1],
                            }
        else:
            entities = {}
            new_fixed_input = ""

        return entities, text, new_fixed_input

    # build context from tags and get the entities from the tags
    else:
        groups = group_tags(testcase["tags"])
        entities = {}

        for group in groups:
            if lowercase:
                tokens = [element["token"].lower() for element in group]
            else:
                tokens = [element["token"] for element in group]
            entity_type = group[0]["tag"]
            entity_content = get_original_entity(fixed_input, tokens)
            entity_content = text_preprocessing.preprocessing(
                entity_content, lowercase=lowercase
            )

            if entity_type != "word":
                if random.random() > probability:

                    # remove last empty string
                    if entity_content[-1] == " ":
                        entity_content = entity_content[:-1]

                    entities[entity_type] = {
                        "text": [entity_content],
                        "answer_start": [fixed_input.find(entity_content)],
                    }
                else:
                    entities[entity_type] = {
                        "text": ["EMPTY"],
                        "answer_start": [-1],
                    }

        return entities, fixed_input, fixed_input


def testcases_to_squad_format(
    testcases: list[dict],
    dataset_name: str,
    allow_empty_answers: bool = False,
    lowercase: bool = True,
    random_removal: bool = False,
) -> list[dict]:
    """Converts testcases to the SQuAD format. Offers the possibility to remove instances.

    Notes
    -----

    The SQuAD Format looks like this:

    {
        'id': '123',
        'title': 'Test title',
        'context': 'This is a test context.',
        'fixed': 'This is a test context.',
        'question': 'What context is this?',
        'answers': {
            'answer_start': [10],
            'text': ['test']
        },
    }

    Parameters
    ----------
    testcases : list[dict]
        List of dicts where a dict represents a testcase.
    dataset_name : str
        Name of the dataset.
    allow_empty_answers : bool, optional
        Includes empty answers, by default False.
    lowercase : bool
        Converts text to lowercase, by default True.
    random_removal : bool, optional
        Removes ~33% of the entities from the context, by default False.

    Returns
    -------
    list[dict]
        Transformed testcases.
    """

    filtered_testcases = []
    transformed_testcases = []
    possible_questions = [
        "city",
        "email",
        "fax",
        "firstName",
        "lastName",
        "mobile",
        "organization",
        "phone",
        "position",
        "street",
        "title",
        "website",
        "zip",
    ]

    if dataset_name == "expected":
        sources = ["CRAWL", "EMAIL", "GRAB", "SCAN"]
    elif dataset_name == "crawl":
        sources = ["CRAWL"]
    else:
        # default
        sources = ["SCAN"]

    filtered_testcases = [
        testcase for testcase in testcases if testcase["source"] in sources
    ]

    logging.info(f"Length '{dataset_name}' testcases: {len(filtered_testcases)}.")

    if allow_empty_answers:
        dataset_name = f"{dataset_name}-na"

    for testcase in tqdm(
        filtered_testcases, desc="Converting testcases to SQuAD format"
    ):
        counter = 0
        entity_id = testcase["id"]

        entities, text, fixed = get_entities_and_text(
            testcase, dataset_name, lowercase=lowercase, random_removal=random_removal
        )

        if entities:
            for question in possible_questions:
                if question not in entities.keys():
                    if allow_empty_answers:
                        instance = {
                            "id": str(entity_id) + "-" + str(counter),
                            "orig_id": entity_id,
                            "title": f"{entity_id}_{dataset_name}_{question}",
                            "context": text + " ᛉ",
                            "fixed": fixed,
                            "question": question,
                            "answers": {
                                "text": ["EMPTY"],
                                "answer_start": [-1],
                            },
                        }
                        transformed_testcases.append(instance)
                        counter += 1
                    else:
                        continue
                else:
                    # double check if question in text
                    if entities[question]["text"][0] in text:
                        instance = {
                            "id": str(entity_id) + "-" + str(counter),
                            "orig_id": entity_id,
                            "title": f"{entity_id}_{dataset_name}_{question}",
                            "context": text + " ᛉ",
                            "fixed": fixed,
                            "question": question,
                            "answers": entities[question],
                        }
                        transformed_testcases.append(instance)
                        counter += 1
                    else:
                        # strange encodings (mostly russian and chinese) will be skipped
                        continue

    return transformed_testcases
