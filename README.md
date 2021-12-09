# contactsQA
Extraction of contact entities from address blocks and imprints with Extractive Question Answering.


## Goal

Input:
```
Dr. Max Mustermann
Hauptstraße 123
97070 Würzburg
```

Output:
```
entities = {
  "city" : "Würzburg",
  "email" : "",
  "fax" : "",
  "firstName" : "Max",
  "lastName" : "Mustermann",
  "mobile" : "",
  "organization" : "",
  "phone" : "",
  "position" : "",
  "street" : "Hauptstraße 123",
  "title" : "Dr.",
  "website" : "",
  "zip" : "97070"
}
```

## Getting started

### Creating a dataset

Due to data protection reasons, no dataset is included in this repository. You need to create a dataset in the SQuAD format, see https://huggingface.co/datasets/squad.
Create the dataset in the `jsonl`-format where one line looks like this:

```javascript
    {
        'id': '123',
        'title': 'mustermanns address',
        'context': 'Meine Adresse ist folgende: \n\nDr. Max Mustermann \nHauptstraße 123 \n97070 Würzburg \n Schicken Sie mir bitte die Rechnung zu.',
        'fixed': 'Dr. Max Mustermann \nHauptstraße 123 \n97070 Würzburg',
        'question': 'firstName',
        'answers': {
            'answer_start': [4],
            'text': ['Max']
        }
    }
```

Questions with no answers should look like this:

```javascript
    {
        'id': '123',
        'title': 'mustermanns address',
        'context': 'Meine Adresse ist folgende: \n\nDr. Max Mustermann \nHauptstraße 123 \n97070 Würzburg \n Schicken Sie mir bitte die Rechnung zu.',
        'fixed': 'Dr. Max Mustermann \nHauptstraße 123 \n97070 Würzburg',
        'question': 'phone',
        'answers': {
            'answer_start': [-1],
            'text': ['EMPTY']
        }
    }
```

Split the dataset into a train-, validation- and test-dataset and save them in a directory with the name `crawl`, `email` or `expected`, like this:

```
├── data
│   ├── crawl
│   │   ├── crawl-test.jsonl
│   │   ├── crawl-train.jsonl
│   │   ├── crawl-val.jsonl
```

If you allow unanswerable questions like in SQuAD v2.0, add a `-na` behind the directory name, like this:

```
├── data
│   ├── crawl-na
│   │   ├── crawl-na-test.jsonl
│   │   ├── crawl-na-train.jsonl
│   │   ├── crawl-na-val.jsonl
```


### Training a model

Example command for training and evaluating a dataset inside the `crawl-na` directory:

```sh
python app/qa-pipeline.py \
--batch_size 4 \
--checkpoint xlm-roberta-base \
--dataset_name crawl \
--dataset_path="../data/" \
--deactivate_map_caching \
--doc_stride 128 \
--epochs 3 \
--gpu_device 0 \
--learning_rate 0.00002 \
--max_answer_length 30 \
--max_length 384 \
--n_best_size 20 \
--n_jobs 8 \
--no_answers \
--overwrite_output_dir;
```

## Virtual Environment Setup

Create and activate the environment (the python version and the environment name can vary at will):

```sh
$ python3.9 -m venv .env
$ source .env/bin/activate
```

To install the project's dependencies, activate the virtual environment and simply run (requires [poetry](https://python-poetry.org/)):

```sh
$ poetry install
```

Alternatively, use the following:

```sh
$ pip install -r requirements.txt
```

Deactivate the environment:

```sh
$ deactivate
```

### Troubleshooting

Common error:

```sh
ModuleNotFoundError: No module named 'setuptools'
```

The solution is to upgrade `setuptools` and then run `poetry install` or `poetry update` afterwards:

```sh
pip install --upgrade setuptools
````
